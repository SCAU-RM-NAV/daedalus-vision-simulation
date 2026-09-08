use crate::capture::driver::{CaptureConfig, CapturedFrameKind};
use crate::capture::{IMAGE_HEIGHT, IMAGE_WIDTH};
use crate::components::{
    Controlled, Infantry, InfantryGimbal, InfantryLaunchOffset, PendingTalosFire, SimHitEventQueue,
    SubscribeAutoAim,
};
use crate::config::SimulationConfig;
use crate::robomaster::prelude::{
    MechanismState, PowerRune, PowerRuneMechanism, PowerRuneRotation, RuneMode, Team,
};
use crate::systems::projectile_launch;
use crate::talos::capture::{
    TalosCaptureContext, TalosCapturePlugin, TalosFrameStamp, advance_talos_frame_stamp,
};
use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use talos_ipc::*;

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct TalosAlgorithmMode(pub u8);

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct TalosGimbalDynamics {
    pub target_yaw_rad: f32,
    pub target_pitch_rad: f32,
    pub yaw_velocity_radps: f32,
    pub pitch_velocity_radps: f32,
    pub yaw_acceleration_radps2: f32,
    pub pitch_acceleration_radps2: f32,
    pub target_yaw_velocity_radps: f32,
    pub target_pitch_velocity_radps: f32,
    pub last_command_seq: u64,
}

impl TalosGimbalDynamics {
    fn mark_command_applied(&mut self, command_seq: u64) {
        self.last_command_seq = command_seq;
    }
}

const MAX_GIMBAL_SPEED_RADPS: f32 = 8.0;
const MAX_GIMBAL_YAW_ACCEL_RADPS2: f32 = 50.0;
const MAX_GIMBAL_PITCH_ACCEL_RADPS2: f32 = 100.0;
const MIN_GIMBAL_ACCEL_RADPS2: f32 = 12.0;
const ZERO_FEEDFORWARD_ACCEL_RADPS2: f32 = 1e-3;
const GIMBAL_POSITION_KP: f32 = 8.0;
const DEFAULT_GIMBAL_PITCH_LIMIT_RAD: f32 = 0.785;

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn step_axis(
    position: f32,
    velocity: f32,
    target: f32,
    requested_velocity: f32,
    requested_accel: f32,
    dt: f32,
) -> (f32, f32) {
    step_pitch_axis(
        position,
        velocity,
        target,
        requested_velocity,
        requested_accel,
        dt,
        DEFAULT_GIMBAL_PITCH_LIMIT_RAD,
        MAX_GIMBAL_PITCH_ACCEL_RADPS2,
    )
}

fn step_yaw_axis(
    position: f32,
    velocity: f32,
    target: f32,
    requested_velocity: f32,
    requested_accel: f32,
    dt: f32,
) -> (f32, f32) {
    let error = wrap_angle(target - position);
    let (desired_velocity, max_acceleration) = position_control_velocity(
        error,
        requested_velocity,
        requested_accel,
        MAX_GIMBAL_YAW_ACCEL_RADPS2,
    );
    let max_delta = max_acceleration * dt;
    let next_velocity = (velocity + (desired_velocity - velocity).clamp(-max_delta, max_delta))
        .clamp(-MAX_GIMBAL_SPEED_RADPS, MAX_GIMBAL_SPEED_RADPS);
    let delta = wrap_angle(target - position);
    let next_position = if delta.abs() <= next_velocity.abs() * dt {
        target
    } else {
        position + next_velocity * dt
    };
    (next_position, next_velocity)
}

fn step_pitch_axis(
    position: f32,
    velocity: f32,
    target: f32,
    requested_velocity: f32,
    requested_accel: f32,
    dt: f32,
    pitch_limit: f32,
    acceleration_limit: f32,
) -> (f32, f32) {
    let target = target.clamp(-pitch_limit, pitch_limit);
    let error = target - position;
    let (desired_velocity, max_acceleration) = position_control_velocity(
        error,
        requested_velocity,
        requested_accel,
        acceleration_limit,
    );
    let max_delta = max_acceleration * dt;
    let next_velocity = (velocity + (desired_velocity - velocity).clamp(-max_delta, max_delta))
        .clamp(-MAX_GIMBAL_SPEED_RADPS, MAX_GIMBAL_SPEED_RADPS);
    let next_position = if error.abs() <= next_velocity.abs() * dt {
        target
    } else {
        position + next_velocity * dt
    };
    (
        next_position.clamp(-pitch_limit, pitch_limit),
        next_velocity,
    )
}

fn position_control_velocity(
    error: f32,
    requested_velocity: f32,
    requested_accel: f32,
    acceleration_limit: f32,
) -> (f32, f32) {
    let requested_accel = requested_accel.abs();
    let max_acceleration = if requested_accel <= ZERO_FEEDFORWARD_ACCEL_RADPS2 {
        MIN_GIMBAL_ACCEL_RADPS2
    } else {
        requested_accel.min(acceleration_limit)
    };
    let feedback_velocity =
        (GIMBAL_POSITION_KP * error).clamp(-MAX_GIMBAL_SPEED_RADPS, MAX_GIMBAL_SPEED_RADPS);
    let braking_velocity = (2.0 * max_acceleration * error.abs()).sqrt();
    let mut desired_velocity = (feedback_velocity + requested_velocity)
        .clamp(-MAX_GIMBAL_SPEED_RADPS, MAX_GIMBAL_SPEED_RADPS);
    desired_velocity = desired_velocity.clamp(-braking_velocity, braking_velocity);
    if error != 0.0 && desired_velocity.signum() != error.signum() {
        desired_velocity = 0.0;
    }
    (desired_velocity, max_acceleration)
}

fn gimbal_local_rotation(yaw: f32, pitch: f32) -> Quat {
    Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0)
}

fn talos_gimbal_frame_alignment(muzzle_local: &Transform) -> Quat {
    muzzle_local.rotation * Quat::from_euler(EulerRot::ZYX, 0.0, 0.0, std::f32::consts::FRAC_PI_2)
}

fn world_command_to_local(
    world_yaw: f32,
    world_pitch: f32,
    gimbal_world_rotation: Quat,
    current_local_yaw: f32,
    current_local_pitch: f32,
    muzzle_local: &Transform,
) -> (f32, f32) {
    // Planner angles use the ROS convention: local +X is the shot direction,
    // +Z is up, and a positive gimbal pitch points down. The model has a fixed
    // muzzle alignment and can retain a non-zero roll, so preserve that roll
    // while solving yaw/pitch instead of subtracting unrelated Euler angles.
    let alignment = talos_gimbal_frame_alignment(muzzle_local);
    let ros_alignment = Quat::from_mat3(&M_ALIGN_MAT3);
    let current_ros_aligned =
        ros_alignment * gimbal_world_rotation * alignment * ros_alignment.inverse();
    let (_, _, current_roll) = current_ros_aligned.to_euler(EulerRot::ZYX);
    let desired_ros_rotation =
        Quat::from_euler(EulerRot::ZYX, world_yaw, world_pitch, current_roll);
    let desired_bevy_aligned = ros_alignment.inverse() * desired_ros_rotation * ros_alignment;
    let current_local = gimbal_local_rotation(current_local_yaw, current_local_pitch);
    let gimbal_parent_world = gimbal_world_rotation * current_local.inverse();
    let desired_local = gimbal_parent_world.inverse() * desired_bevy_aligned * alignment.inverse();
    let (yaw, pitch, _) = desired_local.to_euler(EulerRot::YXZ);
    (wrap_angle(yaw), pitch)
}

fn set_algorithm_mode(keyboard: &ButtonInput<KeyCode>, mode: &mut TalosAlgorithmMode) {
    if keyboard.just_pressed(KeyCode::F6) {
        mode.0 = 1;
    }
    if keyboard.just_pressed(KeyCode::F7) {
        mode.0 = 2;
    }
    if keyboard.just_pressed(KeyCode::F8) {
        mode.0 = 3;
    }
}

#[derive(Resource, Default)]
pub struct ShmSubscriberRes(pub Option<ShmSubscriber>);

#[derive(Resource, Deref, DerefMut)]
pub struct TalosEnabled(pub AtomicBool);

pub struct TalosPluginConfig {
    pub width: u32,
    pub height: u32,
    pub fov_y: f32,
    pub texture_format: TextureFormat,
}

impl Default for TalosPluginConfig {
    fn default() -> Self {
        let config = SimulationConfig::default();
        Self {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
            fov_y: config.camera.fov.to_radians(),
            texture_format: TextureFormat::Rgba8UnormSrgb,
        }
    }
}

#[derive(Default)]
pub struct TalosPlugin {
    pub config: TalosPluginConfig,
}

impl Plugin for TalosPlugin {
    fn build(&self, app: &mut App) {
        let publisher = match ShmPublisher::create() {
            Ok(p) => {
                info!("talos shm created");
                p
            }
            Err(e) => {
                error!("cannot create talos shm: {}", e);
                return;
            }
        };

        let publisher = Arc::new(Mutex::new(publisher));

        let capture_config = CaptureConfig {
            width: self.config.width,
            height: self.config.height,
            texture_format: self.config.texture_format,
            frame_kind: CapturedFrameKind::Rgb8,
        };

        let capture_context = TalosCaptureContext {
            publisher: publisher.clone(),
            fov_y: self.config.fov_y,
        };

        app.init_resource::<TalosFrameStamp>();
        app.init_resource::<TalosAlgorithmMode>();
        app.init_resource::<TalosGimbalDynamics>();

        app.add_plugins(TalosCapturePlugin {
            config: capture_config,
            context: capture_context,
        });

        app.init_resource::<ShmSubscriberRes>();
        app.insert_resource(TalosEnabled(AtomicBool::new(true)));
        app.add_systems(Last, (advance_talos_frame_stamp, heartbeat_system));
        app.add_systems(
            Last,
            process_subscription
                .run_if(|enabled: Res<SubscribeAutoAim>| enabled.load(Ordering::Acquire)),
        );
        app.add_systems(Last, flush_talos_hit_events_system);
        app.add_systems(Last, (talos_hotkeys, sync_rune_mode).chain());
    }
}

fn process_subscription(
    mut subscriber: ResMut<ShmSubscriberRes>,
    mut commands: Commands,
    gimbal: Single<
        (&mut Transform, &GlobalTransform, &mut InfantryGimbal),
        (With<Controlled>, Without<InfantryLaunchOffset>),
    >,
    muzzle_local: Single<&Transform, (With<Controlled>, With<InfantryLaunchOffset>)>,
    time: Res<Time>,
    config: Res<SimulationConfig>,
    mut dynamics: ResMut<TalosGimbalDynamics>,
    mut pending_fire: ResMut<PendingTalosFire>,
) {
    let (mut gimbal_transform, gimbal_global, mut gimbal_data) = gimbal.into_inner();
    let muzzle_local = muzzle_local.into_inner();

    if let Some(cmd) = recv_gimbal_cmd(&mut subscriber) {
        dynamics.mark_command_applied(cmd.command_seq);
        if cmd.control == 0 {
            dynamics.target_yaw_rad = gimbal_data.local_yaw;
            dynamics.target_pitch_rad = gimbal_data.pitch;
            dynamics.yaw_velocity_radps = 0.0;
            dynamics.pitch_velocity_radps = 0.0;
            dynamics.target_yaw_velocity_radps = 0.0;
            dynamics.target_pitch_velocity_radps = 0.0;
            dynamics.yaw_acceleration_radps2 = MAX_GIMBAL_YAW_ACCEL_RADPS2;
            dynamics.pitch_acceleration_radps2 = MAX_GIMBAL_PITCH_ACCEL_RADPS2;
        } else {
            let (local_yaw, local_pitch) = world_command_to_local(
                cmd.yaw_rad,
                cmd.pitch_rad,
                gimbal_global.rotation(),
                gimbal_data.local_yaw,
                gimbal_data.pitch,
                muzzle_local,
            );
            dynamics.target_yaw_rad = wrap_angle(local_yaw);
            dynamics.target_pitch_rad = local_pitch.clamp(
                -config.vehicle.gimbal_pitch_limit,
                config.vehicle.gimbal_pitch_limit,
            );
            dynamics.target_yaw_velocity_radps = cmd.yaw_velocity_radps;
            dynamics.target_pitch_velocity_radps = cmd.pitch_velocity_radps;
            dynamics.yaw_acceleration_radps2 = cmd.yaw_acceleration_radps2;
            dynamics.pitch_acceleration_radps2 = cmd.pitch_acceleration_radps2;
            let pitch_within_limit = local_pitch.abs() <= config.vehicle.gimbal_pitch_limit + 1e-3;
            if cmd.fire == 1 && pitch_within_limit {
                pending_fire.command_seq = cmd.command_seq;
                pending_fire.requested = true;
                commands.queue(|world: &mut World| {
                    world.run_system_once(projectile_launch).unwrap();
                });
            }
        }
    }
    let dt = time.delta_secs().max(1e-4);
    let (yaw, yaw_velocity) = step_yaw_axis(
        gimbal_data.local_yaw,
        dynamics.yaw_velocity_radps,
        dynamics.target_yaw_rad,
        dynamics.target_yaw_velocity_radps,
        dynamics.yaw_acceleration_radps2,
        dt,
    );
    let (pitch, pitch_velocity) = step_pitch_axis(
        gimbal_data.pitch,
        dynamics.pitch_velocity_radps,
        dynamics.target_pitch_rad,
        dynamics.target_pitch_velocity_radps,
        dynamics.pitch_acceleration_radps2,
        dt,
        config.vehicle.gimbal_pitch_limit,
        MAX_GIMBAL_PITCH_ACCEL_RADPS2,
    );
    dynamics.yaw_velocity_radps = yaw_velocity;
    dynamics.pitch_velocity_radps = pitch_velocity;
    gimbal_data.local_yaw = yaw;
    gimbal_data.pitch = pitch;
    gimbal_transform.rotation = gimbal_local_rotation(yaw, pitch);
}

fn flush_talos_hit_events_system(
    context: Option<Res<TalosCaptureContext>>,
    mut events: ResMut<SimHitEventQueue>,
) {
    let Some(context) = context else {
        return;
    };
    if events.0.is_empty() {
        return;
    }
    let Ok(mut publisher) = context.publisher.lock() else {
        return;
    };
    for event in events.0.drain(..) {
        publisher.publish_hit_event(HitEvent {
            event_seq: 0,
            command_seq: event.command_seq,
            hit_timestamp_ns: event.timestamp_ns,
            target_id: event.target_id,
            blade_id: event.blade_id,
            hit_type: event.hit_type,
            correct: u8::from(event.correct),
            outcome: event.outcome,
            reserved0: 0,
            reserved: [0; 24],
        });
    }
}

fn talos_hotkeys(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<TalosAlgorithmMode>,
    mut infantry: Single<&mut Infantry, With<Controlled>>,
) {
    set_algorithm_mode(&keyboard, &mut mode);
    if keyboard.just_pressed(KeyCode::F9) {
        infantry.team = opposite_team(infantry.team);
        info!(
            "Vision detection color is now {}.",
            enemy_color_name(infantry.team)
        );
    }
}

fn opposite_team(team: Team) -> Team {
    match team {
        Team::Red => Team::Blue,
        Team::Blue => Team::Red,
    }
}

fn enemy_color_name(self_team: Team) -> &'static str {
    match self_team {
        Team::Red => "BLUE",
        Team::Blue => "RED",
    }
}

fn sync_rune_mode(
    mode: Res<TalosAlgorithmMode>,
    mut last_mode: Local<u8>,
    mut runes: Query<(&PowerRune, &mut PowerRuneMechanism, &mut PowerRuneRotation)>,
) {
    if *last_mode == mode.0 {
        return;
    }
    *last_mode = mode.0;

    let selected = match mode.0 {
        2 => Some(RuneMode::Small),
        3 => Some(RuneMode::Large),
        _ => None,
    };
    let mut rng = rand::rng();
    for (rune, mut mechanism, mut rotation) in &mut runes {
        if selected == Some(rune.mode()) {
            *mechanism.state_mut() = MechanismState::start(rune.mode(), &mut rng);
            rotation.begin_activation(rune.mode(), &mut rng);
        } else {
            *mechanism.state_mut() = MechanismState::inactive(rune.mode());
            rotation.end_activation();
        }
    }
}

fn heartbeat_system(context: Option<Res<TalosCaptureContext>>) {
    if let Some(ctx) = context {
        if let Ok(mut publisher) = ctx.publisher.lock() {
            publisher.update_heartbeat();
        }
    }
}

pub fn recv_gimbal_cmd(subscriber: &mut ShmSubscriberRes) -> Option<VisionCommand> {
    if subscriber.0.is_none() {
        subscriber.0 = ShmSubscriber::connect().ok();
    }
    let command = subscriber.0.as_mut()?.recv_vision_command();
    if command.is_none() && subscriber.0.is_none() {
        subscriber.0 = ShmSubscriber::connect().ok();
    }
    command
}

pub const M_ALIGN_MAT3: Mat3 = Mat3::from_cols(
    Vec3::new(0.0, -1.0, 0.0), // M[0,0], M[1,0], M[2,0]
    Vec3::new(0.0, 0.0, 1.0),  // M[0,1], M[1,1], M[2,1]
    Vec3::new(-1.0, 0.0, 0.0), // M[0,2], M[1,2], M[2,2]
);

#[inline]
pub fn to_ros(bevy_transform: Transform) -> Transform {
    let new_rotation = to_ros_quat(bevy_transform.rotation);
    let new_translation = to_ros_translation(bevy_transform.translation);
    Transform::from_translation(new_translation).with_rotation(new_rotation)
}

pub fn to_ros_translation(vec3: Vec3) -> Vec3 {
    let align_rot_mat = M_ALIGN_MAT3;
    let new_translation = align_rot_mat * vec3;
    new_translation
}

pub fn to_ros_quat(quat: Quat) -> Quat {
    let align_rot_mat = M_ALIGN_MAT3;
    let align_quat = Quat::from_mat3(&align_rot_mat);
    let new_rotation = align_quat * quat * align_quat.inverse();
    new_rotation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_color_toggle_switches_blue_and_red() {
        assert_eq!(enemy_color_name(Team::Red), "BLUE");
        assert_eq!(enemy_color_name(opposite_team(Team::Red)), "RED");
        assert_eq!(opposite_team(opposite_team(Team::Red)), Team::Red);
    }

    fn pressed(key: KeyCode) -> ButtonInput<KeyCode> {
        let mut keyboard = ButtonInput::default();
        keyboard.press(key);
        keyboard
    }

    #[test]
    fn gimbal_dynamics_obeys_requested_acceleration_without_teleporting() {
        let (position, velocity) = step_axis(0.0, 0.0, 1.0, 8.0, 2.0, 0.1);

        assert!(position > 0.0 && position < 1.0);
        assert!((velocity - 0.2).abs() < 1e-6);
    }

    #[test]
    fn gimbal_dynamics_records_hold_command_sequence() {
        let mut dynamics = TalosGimbalDynamics::default();

        dynamics.mark_command_applied(42);

        assert_eq!(dynamics.last_command_seq, 42);
    }

    #[test]
    fn gimbal_dynamics_caps_requested_speed() {
        let (_, velocity) = step_axis(0.0, 0.0, 3.0, 100.0, 1000.0, 1.0);

        assert!(velocity <= MAX_GIMBAL_SPEED_RADPS);
    }

    #[test]
    fn gimbal_dynamics_closes_position_error_when_feedforward_rate_is_zero() {
        let (position, velocity) = step_axis(0.0, 0.0, 1.0, 0.0, 0.0, 0.1);

        assert!(position > 0.05, "position loop did not move: {position}");
        assert!(
            velocity > 0.5,
            "position loop did not accelerate: {velocity}"
        );
    }

    #[test]
    fn gimbal_pitch_dynamics_never_exceeds_mechanical_limit() {
        let mut position = 0.0;
        let mut velocity = 0.0;

        for _ in 0..8 {
            (position, velocity) = step_axis(position, velocity, 1.2, 8.0, 40.0, 0.1);
        }

        assert!(
            position <= 0.785,
            "pitch escaped its mechanical limit: {position}"
        );
    }

    #[test]
    fn talos_gimbal_pose_uses_the_same_local_rotation_as_manual_control() {
        let rotation = gimbal_local_rotation(0.4, -0.2);
        let (yaw, pitch, roll) = rotation.to_euler(EulerRot::YXZ);

        assert!((yaw - 0.4).abs() < 1e-6);
        assert!((pitch + 0.2).abs() < 1e-6);
        assert!(roll.abs() < 1e-6);
    }

    #[test]
    fn world_command_round_trip_preserves_the_current_model_pose() {
        let muzzle_local = Transform::from_rotation(Quat::from_rotation_x(-1.134));
        let current_local_yaw = 0.35;
        let current_local_pitch = -0.2;
        let gimbal_parent_world = Quat::from_euler(EulerRot::YXZ, -0.4, 0.1, 0.0);
        let gimbal_world =
            gimbal_parent_world * gimbal_local_rotation(current_local_yaw, current_local_pitch);
        let alignment = talos_gimbal_frame_alignment(&muzzle_local);
        let ros_alignment = Quat::from_mat3(&M_ALIGN_MAT3);
        let current_ros = ros_alignment * gimbal_world * alignment * ros_alignment.inverse();
        let (world_yaw, world_pitch, _) = current_ros.to_euler(EulerRot::ZYX);

        let (local_yaw, local_pitch) = world_command_to_local(
            world_yaw,
            world_pitch,
            gimbal_world,
            current_local_yaw,
            current_local_pitch,
            &muzzle_local,
        );

        assert!((wrap_angle(local_yaw - current_local_yaw)).abs() < 1e-5);
        assert!((local_pitch - current_local_pitch).abs() < 1e-5);
    }

    #[test]
    fn world_pitch_command_changes_the_local_pitch_target() {
        let muzzle_local = Transform::from_rotation(Quat::from_rotation_x(-1.134));
        let gimbal_world = Quat::IDENTITY;
        let current_pitch = 0.0;
        let (_, lower) =
            world_command_to_local(0.0, -0.1, gimbal_world, 0.0, current_pitch, &muzzle_local);
        let (_, higher) =
            world_command_to_local(0.0, 0.1, gimbal_world, 0.0, current_pitch, &muzzle_local);

        assert!((higher - lower).abs() > 0.05);
    }

    #[test]
    fn buff_mode_starts_only_matching_runes() {
        let mut world = World::new();
        world.insert_resource(TalosAlgorithmMode(2));
        let small = world
            .spawn((
                PowerRune::new(crate::robomaster::prelude::Team::Blue, RuneMode::Small),
                PowerRuneMechanism::new(RuneMode::Small),
                PowerRuneRotation::new(true),
            ))
            .id();
        let large = world
            .spawn((
                PowerRune::new(crate::robomaster::prelude::Team::Red, RuneMode::Large),
                PowerRuneMechanism::new(RuneMode::Large),
                PowerRuneRotation::new(false),
            ))
            .id();

        let mut schedule = Schedule::default();
        schedule.add_systems(sync_rune_mode);
        schedule.run(&mut world);

        assert!(
            world
                .entity(small)
                .get::<PowerRuneMechanism>()
                .unwrap()
                .state()
                .is_activating()
        );
        assert!(matches!(
            world
                .entity(large)
                .get::<PowerRuneMechanism>()
                .unwrap()
                .state(),
            MechanismState::Inactive {
                mode: RuneMode::Large,
                ..
            }
        ));
    }

    #[test]
    fn mode_hotkeys_publish_auto_aim_small_and_big_buff() {
        let mut mode = TalosAlgorithmMode::default();

        set_algorithm_mode(&pressed(KeyCode::F6), &mut mode);
        assert_eq!(mode.0, 1);
        set_algorithm_mode(&pressed(KeyCode::F7), &mut mode);
        assert_eq!(mode.0, 2);
        set_algorithm_mode(&pressed(KeyCode::F8), &mut mode);
        assert_eq!(mode.0, 3);
    }
}
