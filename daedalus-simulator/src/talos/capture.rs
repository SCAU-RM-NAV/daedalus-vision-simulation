use crate::capture::{
    CameraFov, CaptureBundle, CaptureSource, ImageHandle, compute_camera_intrinsics,
    driver::{
        CaptureConfig, CaptureFrameId, CapturedFrame, CapturedFrameKind, GpuCaptureHandler,
        SnapshotAsync, SnapshotSync,
    },
    setup_capture_camera, setup_preview_window, sync_capture_camera,
};
use crate::components::{
    Controlled, Infantry, InfantryGimbal, InfantryLaunchOffset, ScenarioTarget, SubscribeAutoAim,
};
use crate::config::SimulationConfig;
use crate::robomaster::prelude::{
    Activation, Armor, ArmorRoot, MechanismState, PowerRune, PowerRuneMechanism, PowerRuneRotation,
    RuneMode, Team,
};
use crate::statistic::ProjectileStatistics;
use crate::systems::{GameplaySystems, scenario_linear_velocity_bevy, scenario_spin_rate};
use crate::talos::plugin::{
    TalosAlgorithmMode, TalosGimbalDynamics, to_ros_quat, to_ros_translation,
};
use bevy::ecs::world::DeferredWorld;
use bevy::prelude::*;
use bevy::render::{Extract, ExtractSchedule, RenderApp, RenderSystems};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use talos_ipc::*;

static FRAME_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct TalosFrameStamp {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
}

pub fn advance_talos_frame_stamp(mut stamp: ResMut<TalosFrameStamp>) {
    stamp.frame_seq = FRAME_SEQ.fetch_add(1, Ordering::Relaxed).saturating_add(1);
    stamp.timestamp_ns = now_ns();
}

/// Extracted pose data from MainApp to RenderApp for synchronized publishing
#[derive(Resource, Clone, Default)]
pub struct ExtractedPoseData {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pose: Option<CapturedPoseData>,
    pub valid: bool,
}

/// Pose data captured at frame snapshot time
#[derive(Clone)]
struct CapturedPoseData {
    gimbal_ros: [f32; 3],
    gimbal_quat: [f32; 4],
    camera_rel: [f32; 3],
    camera_rel_quat: [f32; 4],
    yaw_rad: f32,
    pitch_rad: f32,
    yaw_velocity_radps: f32,
    pitch_velocity_radps: f32,
    bullet_speed_mps: f32,
    camp: u8,
    robot_type: u8,
    mode: u8,
    simulation_subscription_enabled: u8,
    last_command_seq: u64,
    projectile_count: u32,
    truth: GroundTruthBatch,
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct TalosSnapshotSync {
    frame_seq: u64,
    timestamp_ns: u64,
    pose: CapturedPoseData,
}

impl SnapshotSync for TalosSnapshotSync {
    fn captured(
        self: Box<Self>,
        world: &mut DeferredWorld,
        _config: &CaptureConfig,
    ) -> Box<dyn SnapshotAsync> {
        let context = world.resource::<TalosCaptureContext>();
        let ctx = context.publisher.clone();

        Box::new(TalosSnapshot {
            ctx,
            fov_y: context.fov_y,
            frame_seq: self.frame_seq,
            timestamp_ns: self.timestamp_ns,
            pose: self.pose,
        })
    }
}

struct TalosSnapshot {
    ctx: Arc<Mutex<ShmPublisher>>,
    fov_y: f32,
    frame_seq: u64,
    timestamp_ns: u64,
    pose: CapturedPoseData,
}

impl SnapshotAsync for TalosSnapshot {
    fn captured(&mut self, frame: CapturedFrame<'_>) {
        if frame.kind != CapturedFrameKind::Rgb8 {
            return;
        }

        let expected_size = (frame.width * frame.height * 3) as usize;
        if frame.data.len() != expected_size {
            warn!(
                "图像大小不匹配: expected {} bytes, got {} bytes",
                expected_size,
                frame.data.len()
            );
            return;
        }

        if frame.width != IMAGE_WIDTH || frame.height != IMAGE_HEIGHT {
            warn!(
                "image reesolution mismatched: expected {}x{}, got {}x{}",
                IMAGE_WIDTH, IMAGE_HEIGHT, frame.width, frame.height
            );
            return;
        }

        if let Ok(mut publisher) = self.ctx.lock() {
            let mut packet = FrameData::default();
            packet.image = ImageMeta {
                frame_seq: self.frame_seq,
                timestamp_ns: self.timestamp_ns,
                width: IMAGE_WIDTH,
                height: IMAGE_HEIGHT,
                buffer_id: 0,
                format: 0,
                reserved: [0; 6],
            };
            packet.camera =
                camera_calibration(&self.pose, self.frame_seq, self.timestamp_ns, self.fov_y);
            packet.gimbal_world = Pose {
                frame_seq: self.frame_seq,
                timestamp_ns: self.timestamp_ns,
                position_m: self.pose.gimbal_ros,
                quaternion_wxyz: self.pose.gimbal_quat,
                reserved: [0; 20],
            };
            packet.feedback = feedback_from_pose(&self.pose, self.frame_seq, self.timestamp_ns);
            packet.truth = self.pose.truth;
            let _ = publisher.try_publish_frame(packet, frame.data);
        }
    }
}

#[derive(Default)]
struct TalosSnapshotCreator {}

impl GpuCaptureHandler for TalosSnapshotCreator {
    fn captured(
        &self,
        world: &World,
        _frame_id: Option<CaptureFrameId>,
    ) -> Option<Box<dyn SnapshotSync>> {
        // Timestamp, frame sequence and pose must come from the same ExtractSchedule snapshot.
        let extracted = world.get_resource::<ExtractedPoseData>()?;
        if !extracted.valid {
            return None;
        }
        let pose = extracted.pose.clone()?;

        Some(Box::new(TalosSnapshotSync {
            frame_seq: extracted.frame_seq,
            timestamp_ns: extracted.timestamp_ns,
            pose,
        }))
    }
}

#[derive(Resource, Clone, Deref, DerefMut)]
pub struct TalosCaptureContextShared(pub Arc<Mutex<ShmPublisher>>);

#[derive(Resource, Clone)]
pub struct TalosCaptureContext {
    pub publisher: Arc<Mutex<ShmPublisher>>,
    pub fov_y: f32,
}

pub struct TalosCapturePlugin {
    pub config: CaptureConfig,
    pub context: TalosCaptureContext,
}

impl Plugin for TalosCapturePlugin {
    fn build(&self, app: &mut App) {
        let capture = CaptureBundle::color(
            app,
            self.config.clone(),
            vec![Box::new(TalosSnapshotCreator::default())],
        );
        let render_target_handle = capture.color_target().unwrap().clone();

        app.add_plugins(capture)
            .insert_resource(ImageHandle(render_target_handle))
            .insert_resource(CameraFov(self.context.fov_y))
            .insert_resource(self.context.clone())
            .add_systems(Startup, setup_capture_camera)
            .add_systems(Startup, setup_preview_window)
            .add_systems(
                Update,
                sync_capture_camera
                    .after(GameplaySystems::Camera)
                    .before(RenderSystems::Render),
            );

        app.sub_app_mut(RenderApp)
            .insert_resource(TalosCaptureContextShared(self.context.publisher.clone()))
            .insert_resource(self.context.clone())
            .insert_resource(ExtractedPoseData::default())
            .add_systems(ExtractSchedule, extract_pose_data);
    }
}

/// Extract pose data from MainApp to RenderApp
fn extract_pose_data(
    mut pose_data: ResMut<ExtractedPoseData>,
    frame_stamp: Extract<Res<TalosFrameStamp>>,
    camera: Extract<Query<&GlobalTransform, With<CaptureSource>>>,
    gimbal: Extract<Query<&GlobalTransform, (With<Controlled>, With<InfantryGimbal>)>>,
    gimbal_data: Extract<Query<&InfantryGimbal, With<Controlled>>>,
    infantry: Extract<Query<&Infantry, With<Controlled>>>,
    armors: Extract<Query<(Entity, &GlobalTransform, &Armor), With<ArmorRoot>>>,
    parents: Extract<Query<&ChildOf>>,
    scenario_targets: Extract<Query<(&ScenarioTarget, &GlobalTransform)>>,
    runes: Extract<
        Query<(
            Entity,
            &GlobalTransform,
            &Transform,
            &PowerRune,
            &PowerRuneMechanism,
            &PowerRuneRotation,
        )>,
    >,
    statistics: Extract<Res<ProjectileStatistics>>,
    config: Extract<Res<SimulationConfig>>,
    muzzle_offset: Extract<
        Query<(&GlobalTransform, &Transform), (With<InfantryLaunchOffset>, With<Controlled>)>,
    >,
    mode: Extract<Res<TalosAlgorithmMode>>,
    dynamics: Extract<Res<TalosGimbalDynamics>>,
    subscription: Extract<Res<SubscribeAutoAim>>,
) {
    pose_data.frame_seq = frame_stamp.frame_seq;
    pose_data.timestamp_ns = frame_stamp.timestamp_ns;

    let Ok(cam_transform) = camera.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(gimbal_transform) = gimbal.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok((muzzle_global, muzzle_local)) = muzzle_offset.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(gimbal_data) = gimbal_data.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };
    let Ok(infantry) = infantry.single() else {
        pose_data.pose = None;
        pose_data.valid = false;
        return;
    };

    pose_data.pose = Some(captured_pose_data(
        cam_transform,
        gimbal_transform,
        muzzle_global,
        muzzle_local,
        gimbal_data,
        infantry,
        mode.0,
        u8::from(subscription.load(Ordering::Acquire)),
        &dynamics,
        config.projectile.speed,
        statistics.launch_count,
        capture_ground_truth(
            &armors,
            &runes,
            &parents,
            &scenario_targets,
            pose_data.frame_seq,
            pose_data.timestamp_ns,
        ),
    ));
    pose_data.valid = true;
}

fn captured_pose_data(
    cam_transform: &GlobalTransform,
    gimbal_transform: &GlobalTransform,
    muzzle_global: &GlobalTransform,
    muzzle_local: &Transform,
    gimbal_data: &InfantryGimbal,
    infantry: &Infantry,
    mode: u8,
    simulation_subscription_enabled: u8,
    dynamics: &TalosGimbalDynamics,
    bullet_speed_mps: f32,
    projectile_count: u32,
    truth: GroundTruthBatch,
) -> CapturedPoseData {
    let raw_camera_rel = cam_transform.reparented_to(gimbal_transform);
    let gimbal_frame_alignment = talos_gimbal_frame_alignment(muzzle_local);
    let cam_rel = camera_in_talos_gimbal_frame(raw_camera_rel, gimbal_frame_alignment);
    let _ = muzzle_global.reparented_to(gimbal_transform);

    let gimbal_rot = gimbal_transform.rotation() * gimbal_frame_alignment;

    let gimbal_ros = to_ros_translation(gimbal_transform.translation());
    let gimbal_rot = to_ros_quat(gimbal_rot);
    let camera = to_ros_translation(cam_rel.translation);
    let camera_quat = to_ros_quat(cam_rel.rotation);

    CapturedPoseData {
        gimbal_ros: [gimbal_ros.x, gimbal_ros.y, gimbal_ros.z],
        gimbal_quat: [gimbal_rot.w, gimbal_rot.x, gimbal_rot.y, gimbal_rot.z],
        camera_rel: [camera.x, camera.y, camera.z],
        camera_rel_quat: [camera_quat.w, camera_quat.x, camera_quat.y, camera_quat.z],
        yaw_rad: gimbal_data.local_yaw,
        pitch_rad: gimbal_data.pitch,
        yaw_velocity_radps: dynamics.yaw_velocity_radps,
        pitch_velocity_radps: dynamics.pitch_velocity_radps,
        bullet_speed_mps,
        camp: team_to_u8(infantry.team),
        robot_type: u8::from(matches!(
            infantry.config.armor,
            crate::robomaster::prelude::ArmorSpec::Large(_)
        )),
        mode,
        simulation_subscription_enabled,
        last_command_seq: dynamics.last_command_seq,
        projectile_count,
        truth,
    }
}

fn talos_gimbal_frame_alignment(muzzle_local: &Transform) -> Quat {
    muzzle_local.rotation * Quat::from_euler(EulerRot::ZYX, 0.0, 0.0, PI / 2.0)
}

fn camera_in_talos_gimbal_frame(
    raw_camera_rel: Transform,
    gimbal_frame_alignment: Quat,
) -> Transform {
    let inverse_alignment = gimbal_frame_alignment.inverse();
    Transform::from_translation(inverse_alignment * raw_camera_rel.translation)
        .with_rotation(inverse_alignment * raw_camera_rel.rotation)
}

fn feedback_from_pose(
    pose: &CapturedPoseData,
    frame_seq: u64,
    timestamp_ns: u64,
) -> GimbalFeedback {
    GimbalFeedback {
        frame_seq,
        timestamp_ns,
        yaw_rad: pose.yaw_rad,
        pitch_rad: pose.pitch_rad,
        yaw_velocity_radps: pose.yaw_velocity_radps,
        pitch_velocity_radps: pose.pitch_velocity_radps,
        bullet_speed_mps: pose.bullet_speed_mps,
        projectile_count: pose.projectile_count,
        camp: pose.camp,
        robot_type: pose.robot_type,
        mode: pose.mode,
        simulation_subscription_enabled: pose.simulation_subscription_enabled,
        reserved0: 0,
        last_command_seq: pose.last_command_seq,
        reserved: [0; 8],
    }
}

fn camera_calibration(
    pose: &CapturedPoseData,
    frame_seq: u64,
    timestamp_ns: u64,
    fov_y: f32,
) -> CameraCalibration {
    let intrinsics = compute_camera_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, fov_y);
    let quat = Quat::from_xyzw(
        pose.camera_rel_quat[1],
        pose.camera_rel_quat[2],
        pose.camera_rel_quat[3],
        pose.camera_rel_quat[0],
    );
    let matrix = Mat3::from_quat(quat) * opencv_optical_to_gimbal();
    CameraCalibration {
        frame_seq,
        timestamp_ns,
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        intrinsics: [
            intrinsics.fx as f32,
            intrinsics.fy as f32,
            intrinsics.cx as f32,
            intrinsics.cy as f32,
        ],
        distortion: [0.0; 8],
        r_camera2gimbal_row_major: [
            matrix.x_axis.x,
            matrix.y_axis.x,
            matrix.z_axis.x,
            matrix.x_axis.y,
            matrix.y_axis.y,
            matrix.z_axis.y,
            matrix.x_axis.z,
            matrix.y_axis.z,
            matrix.z_axis.z,
        ],
        t_camera2gimbal_m: pose.camera_rel,
        reserved: [0; 136],
    }
}

fn opencv_optical_to_gimbal() -> Mat3 {
    // OpenCV optical: x right, y down, z forward. Talos gimbal: x forward, y left, z up.
    Mat3::from_cols(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_calibration_preserves_non_identity_camera_to_gimbal_rotation() {
        let pose = CapturedPoseData {
            camera_rel_quat: {
                let rotation = Quat::from_rotation_x(0.3);
                [rotation.w, rotation.x, rotation.y, rotation.z]
            },
            ..default_pose_data()
        };
        let calibration = camera_calibration(&pose, 7, 9, 1.0);

        assert_ne!(
            calibration.r_camera2gimbal_row_major,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(calibration.frame_seq, 7);
        assert_eq!(calibration.timestamp_ns, 9);
    }

    #[test]
    fn gimbal_feedback_reports_configured_projectile_speed() {
        let pose = CapturedPoseData {
            bullet_speed_mps: 31.5,
            ..default_pose_data()
        };

        assert_eq!(feedback_from_pose(&pose, 7, 9).bullet_speed_mps, 31.5);
    }

    #[test]
    fn camera_calibration_composes_with_published_gimbal_pose() {
        let gimbal_rotation = Quat::from_euler(EulerRot::YXZ, 0.4, -0.2, 0.0);
        let muzzle_local = Transform::from_rotation(Quat::from_rotation_x(0.25));
        let talos_frame_rotation =
            muzzle_local.rotation * Quat::from_euler(EulerRot::ZYX, 0.0, 0.0, PI / 2.0);
        let camera_in_talos_gimbal = Quat::from_euler(EulerRot::YXZ, -0.15, 0.1, 0.0);
        let camera_rotation = gimbal_rotation * talos_frame_rotation * camera_in_talos_gimbal;
        let gimbal_global = GlobalTransform::from(Transform::from_rotation(gimbal_rotation));
        let camera_global = GlobalTransform::from(Transform::from_rotation(camera_rotation));
        let gimbal_data = InfantryGimbal::default();
        let infantry = Infantry::new(
            Team::Blue,
            crate::robomaster::prelude::INFANTRY_THREE_CONFIG,
        );

        let pose = captured_pose_data(
            &camera_global,
            &gimbal_global,
            &gimbal_global,
            &muzzle_local,
            &gimbal_data,
            &infantry,
            1,
            1,
            &TalosGimbalDynamics::default(),
            25.0,
            0,
            GroundTruthBatch::default(),
        );
        let calibration = camera_calibration(&pose, 7, 9, 1.0);
        let published_gimbal = Quat::from_xyzw(
            pose.gimbal_quat[1],
            pose.gimbal_quat[2],
            pose.gimbal_quat[3],
            pose.gimbal_quat[0],
        );
        let camera_to_gimbal = Mat3::from_cols(
            Vec3::new(
                calibration.r_camera2gimbal_row_major[0],
                calibration.r_camera2gimbal_row_major[3],
                calibration.r_camera2gimbal_row_major[6],
            ),
            Vec3::new(
                calibration.r_camera2gimbal_row_major[1],
                calibration.r_camera2gimbal_row_major[4],
                calibration.r_camera2gimbal_row_major[7],
            ),
            Vec3::new(
                calibration.r_camera2gimbal_row_major[2],
                calibration.r_camera2gimbal_row_major[5],
                calibration.r_camera2gimbal_row_major[8],
            ),
        );
        let optical_to_gimbal = Mat3::from_cols(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let expected_camera = Mat3::from_quat(to_ros_quat(camera_rotation)) * optical_to_gimbal;

        assert!(
            (Mat3::from_quat(published_gimbal) * camera_to_gimbal)
                .abs_diff_eq(expected_camera, 1e-5),
            "camera-to-gimbal calibration is not expressed in the published gimbal frame"
        );
    }

    #[test]
    fn scenario_armor_truth_velocity_combines_vertical_and_spin_motion() {
        let mut target = ScenarioTarget::new(crate::components::ScenarioTargetKind::Infantry);
        target.profile.enabled = true;
        target.profile.vertical_mode = crate::components::VerticalMotionMode::Sine;
        target.profile.vertical_amplitude_m = 1.0;
        target.profile.vertical_peak_speed_mps = 2.0;
        target.profile.spin_mode = crate::components::SpinMode::Constant;
        target.profile.spin_speed_radps = 3.0;

        let (velocity, yaw_rate) =
            scenario_armor_velocity_bevy(&target, Vec3::ZERO, Vec3::new(0.0, 0.0, -2.0));

        assert!(velocity.abs_diff_eq(Vec3::new(-6.0, 2.0, 0.0), 1e-6));
        assert_eq!(yaw_rate, 3.0);
    }

    fn default_pose_data() -> CapturedPoseData {
        CapturedPoseData {
            gimbal_ros: [0.0; 3],
            gimbal_quat: [1.0, 0.0, 0.0, 0.0],
            camera_rel: [0.0; 3],
            camera_rel_quat: [1.0, 0.0, 0.0, 0.0],
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            yaw_velocity_radps: 0.0,
            pitch_velocity_radps: 0.0,
            bullet_speed_mps: 0.0,
            camp: 0,
            robot_type: 0,
            mode: 0,
            simulation_subscription_enabled: 0,
            last_command_seq: 0,
            projectile_count: 0,
            truth: GroundTruthBatch::default(),
        }
    }
}

fn team_to_u8(team: Team) -> u8 {
    match team {
        Team::Red => 0,
        Team::Blue => 1,
    }
}

fn activation_to_u8(activation: &Activation) -> u8 {
    match activation {
        Activation::Deactivated => 0,
        Activation::Activating => 1,
        Activation::Activated => 2,
        Activation::Completed => 3,
    }
}

fn mechanism_state_to_u8(state: &MechanismState) -> u8 {
    match state {
        MechanismState::Inactive { .. } => 0,
        MechanismState::Activating(_) => 1,
        MechanismState::Activated { .. } => 2,
        MechanismState::Failed { .. } => 3,
    }
}

fn rune_mode_to_u8(mode: RuneMode) -> u8 {
    match mode {
        RuneMode::Small => 2,
        RuneMode::Large => 3,
    }
}

fn capture_ground_truth(
    armors: &Query<(Entity, &GlobalTransform, &Armor), With<ArmorRoot>>,
    runes: &Query<(
        Entity,
        &GlobalTransform,
        &Transform,
        &PowerRune,
        &PowerRuneMechanism,
        &PowerRuneRotation,
    )>,
    parents: &Query<&ChildOf>,
    scenario_targets: &Query<(&ScenarioTarget, &GlobalTransform)>,
    frame_seq: u64,
    timestamp_ns: u64,
) -> GroundTruthBatch {
    let mut batch = GroundTruthBatch::default();
    batch.frame_seq = frame_seq;
    batch.timestamp_ns = timestamp_ns;
    for (entity, transform, armor) in armors.iter() {
        if batch.target_count as usize >= GROUND_TRUTH_MAX_TARGETS {
            break;
        }
        let position = to_ros_translation(transform.translation());
        let quaternion = to_ros_quat(transform.rotation());
        let (velocity_mps, yaw_rate_radps) =
            scenario_motion_for_armor(entity, transform.translation(), parents, scenario_targets);
        let index = batch.target_count as usize;
        batch.targets[index] = GroundTruthTarget {
            frame_seq,
            timestamp_ns,
            target_id: entity.to_bits() as u32,
            team: team_to_u8(armor.team),
            armor_label: armor.label as u8,
            robot_type: u8::from(matches!(
                armor.spec,
                crate::robomaster::prelude::ArmorSpec::Large(_)
            )),
            is_outpost: u8::from(armor.label == crate::robomaster::prelude::ArmorLabel::Outpost),
            position_m: [position.x, position.y, position.z],
            quaternion_wxyz: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
            velocity_mps,
            yaw_rate_radps,
            reserved: [0; 8],
        };
        batch.target_count += 1;
    }
    for (entity, global, local, rune, mechanism, rotation) in runes.iter() {
        if batch.rune_count as usize >= GROUND_TRUTH_MAX_RUNES {
            break;
        }
        let position = to_ros_translation(global.translation());
        let quaternion = to_ros_quat(global.rotation());
        let axis = Dir3::from_xyz(-1.0, 0.0, -1.0).expect("rune axis is normalized");
        let (rotation_axis, angle) = local.rotation.to_axis_angle();
        let current_angle = angle * rotation_axis.dot(*axis).signum();
        let controller = rotation.controller();
        let (sine_amplitude, sine_omega, relative_time, sine_offset) = controller
            .variable_params()
            .map(|(amplitude, omega, time)| (amplitude, omega, time, 2.090 - amplitude))
            .unwrap_or((0.0, 0.0, 0.0, 0.0));
        let mut activations = [0; 5];
        for (index, activation) in mechanism.state().target_states().iter().enumerate() {
            activations[index] = activation_to_u8(activation);
        }
        let index = batch.rune_count as usize;
        batch.runes[index] = GroundTruthRune {
            frame_seq,
            timestamp_ns,
            rune_id: entity.to_bits() as u32,
            team: team_to_u8(rune.team()),
            mode: rune_mode_to_u8(rune.mode()),
            mechanism_state: mechanism_state_to_u8(mechanism.state()),
            direction: if controller.is_clockwise() { 1 } else { -1 },
            center_m: [position.x, position.y, position.z],
            quaternion_wxyz: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
            radius_m: 0.0,
            angle_rad: current_angle,
            angular_velocity_radps: 0.0,
            sine_amplitude,
            sine_omega,
            sine_phase: 0.0,
            sine_offset,
            relative_time_s: relative_time,
            active_blade_id: -1,
            target_activations: activations,
            reserved: [0; 35],
        };
        batch.rune_count += 1;
    }
    batch
}

fn scenario_motion_for_armor(
    mut entity: Entity,
    armor_position: Vec3,
    parents: &Query<&ChildOf>,
    scenario_targets: &Query<(&ScenarioTarget, &GlobalTransform)>,
) -> ([f32; 3], f32) {
    loop {
        if let Ok((target, target_transform)) = scenario_targets.get(entity) {
            let (velocity_bevy, yaw_rate_radps) = scenario_armor_velocity_bevy(
                target,
                target_transform.translation(),
                armor_position,
            );
            let velocity = to_ros_translation(velocity_bevy);
            return ([velocity.x, velocity.y, velocity.z], yaw_rate_radps);
        }
        let Ok(parent) = parents.get(entity) else {
            return ([0.0; 3], 0.0);
        };
        entity = parent.parent();
    }
}

fn scenario_armor_velocity_bevy(
    target: &ScenarioTarget,
    target_position: Vec3,
    armor_position: Vec3,
) -> (Vec3, f32) {
    if !target.profile.enabled {
        return (Vec3::ZERO, 0.0);
    }
    let yaw_rate_radps = scenario_spin_rate(target);
    let angular_velocity = Vec3::Y * yaw_rate_radps;
    let linear_velocity = scenario_linear_velocity_bevy(target);
    (
        linear_velocity + angular_velocity.cross(armor_position - target_position),
        yaw_rate_radps,
    )
}
