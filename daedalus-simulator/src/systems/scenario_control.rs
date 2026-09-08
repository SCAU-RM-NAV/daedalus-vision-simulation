use avian3d::prelude::{AngularVelocity, CustomPositionIntegration, LinearVelocity, RigidBody};
use bevy::prelude::*;
use bevy_inspector_egui::bevy_egui::{EguiContexts, EguiPrimaryContextPass};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::components::{
    ActiveSlapper, ArmorPair, HorizontalMotionMode, InfantryChassis, InfantryLaunchOffset,
    ScenarioArmorGeometry, ScenarioChassisGeometry, ScenarioProfile, ScenarioTarget,
    ScenarioTargetKind, SpinDirection, SpinMode, VerticalMotionMode,
};
use crate::config::SimulationConfig;
use crate::robomaster::prelude::ArmorRoot;
use crate::systems::ControllerState;

#[derive(Resource)]
struct ScenarioUiState {
    selected_target: ScenarioTargetKind,
    event_sequence: u64,
    last_status: String,
}

impl Default for ScenarioUiState {
    fn default() -> Self {
        Self {
            selected_target: ScenarioTargetKind::Infantry,
            event_sequence: 0,
            last_status: "Select a target and enable scripted motion.".to_string(),
        }
    }
}

pub struct ScenarioControlPlugin {
    pub gui_enabled: bool,
}

impl Plugin for ScenarioControlPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ScenarioUiState>()
            .add_systems(
                Update,
                (
                    initialize_scenario_armor_geometry,
                    initialize_scenario_chassis_geometry,
                    sync_scenario_armor_heights,
                    scenario_spin_hotkeys,
                )
                    .chain(),
            )
            .add_systems(
                FixedUpdate,
                (
                    drive_scenario_targets.after(crate::systems::remote_vehicle_controls),
                    drive_scenario_chassis_spin.after(drive_scenario_targets),
                )
                    .chain(),
            );

        if self.gui_enabled {
            app.add_systems(EguiPrimaryContextPass, scenario_control_panel);
        }
    }
}

fn requested_spin_mode(keyboard: &ButtonInput<KeyCode>) -> Option<SpinMode> {
    if keyboard.just_pressed(KeyCode::F10) {
        Some(SpinMode::Constant)
    } else if keyboard.just_pressed(KeyCode::F11) {
        Some(SpinMode::Variable)
    } else if keyboard.just_pressed(KeyCode::F12) {
        Some(SpinMode::Off)
    } else {
        None
    }
}

fn scenario_spin_hotkeys(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<ScenarioUiState>,
    mut targets: Query<&mut ScenarioTarget>,
) {
    let Some(mode) = requested_spin_mode(&keyboard) else {
        return;
    };
    let Some(mut target) = targets
        .iter_mut()
        .find(|target| target.kind == state.selected_target)
    else {
        state.last_status = "Spin hotkey: selected scenario target was not found".to_string();
        return;
    };

    target.profile.spin_mode = mode;
    if mode != SpinMode::Off {
        target.profile.enabled = true;
    }
    let mode_name = mode.label();
    match record_parameter_event(&mut state, &target, "spin_hotkey") {
        Ok(()) => {
            state.last_status = format!(
                "Armor spin {}: {} (F10 constant / F11 sine / F12 off)",
                target.kind.label(),
                mode_name
            );
        }
        Err(error) => {
            state.last_status = format!("Armor spin {} applied; record failed: {error}", mode_name);
        }
    }
}

fn ancestor_scenario_target(
    mut entity: Entity,
    parents: &Query<&ChildOf>,
    targets: &Query<(), With<ScenarioTarget>>,
) -> Option<Entity> {
    loop {
        if targets.contains(entity) {
            return Some(entity);
        }
        entity = parents.get(entity).ok()?.parent();
    }
}

fn normalized_xz(translation: Vec3) -> Option<Vec2> {
    Vec2::new(translation.x, translation.z).try_normalize()
}

fn pair_dot(a: Vec3, b: Vec3) -> Option<f32> {
    Some(normalized_xz(a)?.dot(normalized_xz(b)?))
}

fn assign_armor_pairs(roots: &[(Entity, Vec3)]) -> Option<Vec<(Entity, ArmorPair, Vec3)>> {
    if roots.len() != 4 {
        return None;
    }
    const PAIRINGS: [((usize, usize), (usize, usize)); 3] =
        [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))];

    let mut best: Option<(((usize, usize), (usize, usize)), f32, f32)> = None;
    for pairing in PAIRINGS {
        let first_dot = pair_dot(roots[pairing.0.0].1, roots[pairing.0.1].1)?;
        let second_dot = pair_dot(roots[pairing.1.0].1, roots[pairing.1.1].1)?;
        let score = first_dot + second_dot;
        if best
            .as_ref()
            .is_none_or(|(_, best_score, _)| score < *best_score)
        {
            best = Some((pairing, score, first_dot.min(second_dot)));
        }
    }

    let (pairing, _, worst_dot) = best?;
    if worst_dot > -0.85 {
        return None;
    }
    let first_height = (roots[pairing.0.0].1.y + roots[pairing.0.1].1.y) * 0.5;
    let second_height = (roots[pairing.1.0].1.y + roots[pairing.1.1].1.y) * 0.5;
    let (pair_a, pair_b) = if first_height <= second_height {
        (pairing.0, pairing.1)
    } else {
        (pairing.1, pairing.0)
    };

    Some(
        [pair_a.0, pair_a.1]
            .into_iter()
            .map(|index| (roots[index].0, ArmorPair::A, roots[index].1))
            .chain(
                [pair_b.0, pair_b.1]
                    .into_iter()
                    .map(|index| (roots[index].0, ArmorPair::B, roots[index].1)),
            )
            .collect(),
    )
}

fn initialize_scenario_armor_geometry(
    mut commands: Commands,
    armors: Query<(Entity, &Transform), (With<ArmorRoot>, Without<ScenarioArmorGeometry>)>,
    parents: Query<&ChildOf>,
    targets: Query<(), With<ScenarioTarget>>,
    mut target_profiles: Query<&mut ScenarioTarget>,
) {
    let mut by_target: HashMap<Entity, Vec<(Entity, Vec3)>> = HashMap::new();
    for (entity, transform) in &armors {
        let Some(target) = ancestor_scenario_target(entity, &parents, &targets) else {
            continue;
        };
        by_target
            .entry(target)
            .or_default()
            .push((entity, transform.translation));
    }

    for (target, roots) in by_target {
        let Some(pairs) = assign_armor_pairs(&roots) else {
            continue;
        };
        let mut pair_radius_sum_m = [0.0; 2];
        let mut pair_counts = [0_u32; 2];
        for (_, pair, translation) in &pairs {
            pair_radius_sum_m[pair.index()] += Vec2::new(translation.x, translation.z).length();
            pair_counts[pair.index()] += 1;
        }
        if let Ok(mut target_profile) = target_profiles.get_mut(target) {
            if !target_profile.profile.armor_radii_initialized {
                target_profile.profile.armor_radii_m = std::array::from_fn(|index| {
                    pair_radius_sum_m[index] / pair_counts[index].max(1) as f32
                });
                target_profile.profile.armor_radii_initialized = true;
            }
        }
        for (entity, pair, original_translation) in pairs {
            commands.entity(entity).insert(ScenarioArmorGeometry {
                target,
                pair,
                original_translation,
            });
        }
    }
}

fn initialize_scenario_chassis_geometry(
    mut commands: Commands,
    chassis: Query<(Entity, &Transform), (With<InfantryChassis>, Without<ScenarioChassisGeometry>)>,
    parents: Query<&ChildOf>,
    targets: Query<(), With<ScenarioTarget>>,
) {
    for (entity, transform) in &chassis {
        let Some(target) = ancestor_scenario_target(entity, &parents, &targets) else {
            continue;
        };
        commands.entity(entity).insert(ScenarioChassisGeometry {
            target,
            original_rotation: transform.rotation,
        });
    }
}

fn armor_translation_at_radius(original_translation: Vec3, radius_m: f32) -> Vec3 {
    let radial_direction = Vec2::new(original_translation.x, original_translation.z)
        .try_normalize()
        .unwrap_or(Vec2::X);
    let radius_m = radius_m.clamp(0.01, 2.0);
    Vec3::new(
        radial_direction.x * radius_m,
        original_translation.y,
        radial_direction.y * radius_m,
    )
}

fn sync_scenario_armor_heights(
    targets: Query<&ScenarioTarget>,
    mut armors: Query<(&mut Transform, &ScenarioArmorGeometry)>,
) {
    for (mut transform, geometry) in &mut armors {
        let Ok(target) = targets.get(geometry.target) else {
            continue;
        };
        let mut translation = armor_translation_at_radius(
            geometry.original_translation,
            target.profile.armor_radii_m[geometry.pair.index()],
        );
        translation.y += target.profile.armor_height_offsets_m[geometry.pair.index()];
        transform.translation = translation;
    }
}

fn vertical_displacement(profile: &ScenarioTarget) -> f32 {
    let profile = &profile.profile;
    match profile.vertical_mode {
        VerticalMotionMode::Static => 0.0,
        VerticalMotionMode::Sine => {
            let amplitude = profile.vertical_amplitude_m.abs();
            if amplitude <= f32::EPSILON || profile.vertical_peak_speed_mps <= f32::EPSILON {
                return 0.0;
            }
            let omega = profile.vertical_peak_speed_mps / amplitude;
            amplitude * (omega * profile.elapsed_s + profile.vertical_phase_rad).sin()
        }
    }
}

pub(crate) fn scenario_vertical_velocity(profile: &ScenarioTarget) -> f32 {
    let profile = &profile.profile;
    match profile.vertical_mode {
        VerticalMotionMode::Static => 0.0,
        VerticalMotionMode::Sine => {
            let amplitude = profile.vertical_amplitude_m.abs();
            if amplitude <= f32::EPSILON || profile.vertical_peak_speed_mps <= f32::EPSILON {
                return 0.0;
            }
            let omega = profile.vertical_peak_speed_mps / amplitude;
            profile.vertical_peak_speed_mps
                * (omega * profile.elapsed_s + profile.vertical_phase_rad).cos()
        }
    }
}

fn ordered_range(range: [f32; 2]) -> (f32, f32) {
    (range[0].min(range[1]), range[0].max(range[1]))
}

fn polynomial_loop_state(profile: &ScenarioTarget) -> (f32, f32) {
    let phase = std::f32::consts::TAU
        * profile.profile.polynomial_frequency_hz.abs()
        * profile.profile.elapsed_s;
    let progress = 0.5 * (1.0 - phase.cos());
    let progress_velocity =
        std::f32::consts::PI * profile.profile.polynomial_frequency_hz.abs() * phase.sin();
    (progress, progress_velocity)
}

fn polynomial_axis(range: [f32; 2], power: u32, progress: f32) -> (f32, f32) {
    let (minimum, maximum) = ordered_range(range);
    let power = power.clamp(1, 8);
    let progress_power = progress.powi(power as i32);
    let derivative = power as f32 * progress.powi(power.saturating_sub(1) as i32);
    (
        minimum + (maximum - minimum) * progress_power,
        (maximum - minimum) * derivative,
    )
}

fn polynomial_world_xy(profile: &ScenarioTarget) -> (Vec2, Vec2) {
    let (progress, progress_velocity) = polynomial_loop_state(profile);
    let (x, dx_dprogress) = polynomial_axis(
        profile.profile.polynomial_x_range_m,
        profile.profile.polynomial_x_power,
        progress,
    );
    let (y, dy_dprogress) = polynomial_axis(
        profile.profile.polynomial_y_range_m,
        profile.profile.polynomial_y_power,
        progress,
    );
    (
        Vec2::new(x, y),
        Vec2::new(dx_dprogress, dy_dprogress) * progress_velocity,
    )
}

fn scenario_horizontal_position_bevy(target: &ScenarioTarget, origin: Vec3) -> Vec3 {
    match target.profile.horizontal_mode {
        HorizontalMotionMode::StaticOffset => {
            let [world_x, world_y] = target.profile.world_xy_offset_m;
            origin + Vec3::new(-world_y, 0.0, -world_x)
        }
        HorizontalMotionMode::PolynomialLoop => {
            let (world_xy, _) = polynomial_world_xy(target);
            Vec3::new(-world_xy.y, origin.y, -world_xy.x)
        }
    }
}

pub(crate) fn scenario_horizontal_velocity_bevy(target: &ScenarioTarget) -> Vec3 {
    match target.profile.horizontal_mode {
        HorizontalMotionMode::StaticOffset => Vec3::ZERO,
        HorizontalMotionMode::PolynomialLoop => {
            let (_, world_velocity) = polynomial_world_xy(target);
            Vec3::new(-world_velocity.y, 0.0, -world_velocity.x)
        }
    }
}

pub(crate) fn scenario_linear_velocity_bevy(target: &ScenarioTarget) -> Vec3 {
    scenario_horizontal_velocity_bevy(target) + Vec3::Y * scenario_vertical_velocity(target)
}

pub(crate) fn scenario_spin_rate(profile: &ScenarioTarget) -> f32 {
    let profile = &profile.profile;
    let direction = profile.spin_direction.sign();
    match profile.spin_mode {
        SpinMode::Off => 0.0,
        SpinMode::Constant => direction * profile.spin_speed_radps,
        SpinMode::Variable => {
            let min_speed = profile.spin_min_speed_radps.max(0.05);
            let max_speed = profile.spin_max_speed_radps.max(min_speed);
            let mean_speed = (min_speed + max_speed) * 0.5;
            let amplitude = (max_speed - min_speed) * 0.5;
            let phase = std::f32::consts::TAU * profile.spin_frequency_hz.abs() * profile.elapsed_s;
            direction * (mean_speed + amplitude * phase.cos())
        }
    }
}

fn spin_yaw(profile: &ScenarioTarget) -> f32 {
    profile.profile.spin_angle_rad
}

fn has_scripted_translation(profile: &ScenarioProfile) -> bool {
    profile.horizontal_mode != HorizontalMotionMode::StaticOffset
        || profile
            .world_xy_offset_m
            .iter()
            .any(|value| value.abs() > 1e-6)
        || profile.vertical_mode != VerticalMotionMode::Static
        || profile.vertical_offset_m.abs() > 1e-6
}

fn drive_scenario_targets(
    mut commands: Commands,
    time: Res<Time<Fixed>>,
    controller: Option<Res<ControllerState>>,
    config: Option<Res<SimulationConfig>>,
    active_muzzle: Query<&GlobalTransform, (With<ActiveSlapper>, With<InfantryLaunchOffset>)>,
    mut targets: Query<(
        Entity,
        &mut Transform,
        &RigidBody,
        &mut ScenarioTarget,
        Option<&ActiveSlapper>,
        Option<&mut LinearVelocity>,
        Option<&mut AngularVelocity>,
    )>,
) {
    let remote_input = controller
        .as_ref()
        .map(|controller| controller.remote)
        .unwrap_or_default();
    let manual_max_speed = config
        .as_ref()
        .map(|config| config.vehicle.max_speed)
        .unwrap_or(8.0);
    for (
        entity,
        mut transform,
        rigid_body,
        mut target,
        is_active,
        linear_velocity,
        angular_velocity,
    ) in &mut targets
    {
        if !target.profile.enabled {
            if target.was_scripted && *rigid_body == RigidBody::Kinematic {
                commands.entity(entity).insert((
                    RigidBody::Dynamic,
                    LinearVelocity::ZERO,
                    AngularVelocity::ZERO,
                ));
                commands
                    .entity(entity)
                    .remove::<CustomPositionIntegration>();
            }
            if target.was_scripted {
                target.was_scripted = false;
                target.restore_chassis = true;
            }
            continue;
        }

        // Armor/chassis spin does not require a kinematic vehicle root. Keep the root dynamic so
        // I/J/K/L continues through VehicleDynamic + Forces with the same acceleration, inertia
        // and collision response as a non-spinning enemy. Only scripted translation owns the root
        // transform and therefore needs kinematic integration.
        if !has_scripted_translation(&target.profile) {
            if *rigid_body == RigidBody::Kinematic {
                commands.entity(entity).insert(RigidBody::Dynamic);
                commands
                    .entity(entity)
                    .remove::<CustomPositionIntegration>();
            }
            target.was_scripted = false;
            target.origin_translation = None;
            target.manual_offset_bevy = Vec3::ZERO;
            let previous_spin_rate = scenario_spin_rate(&target);
            target.profile.elapsed_s += time.delta_secs();
            let current_spin_rate = scenario_spin_rate(&target);
            target.profile.spin_angle_rad +=
                (previous_spin_rate + current_spin_rate) * 0.5 * time.delta_secs();
            match angular_velocity {
                Some(mut value) => value.0 = Vec3::ZERO,
                None => {
                    commands.entity(entity).insert(AngularVelocity::ZERO);
                }
            }
            continue;
        }

        if *rigid_body != RigidBody::Kinematic {
            commands
                .entity(entity)
                .insert((RigidBody::Kinematic, CustomPositionIntegration));
        }
        let origin_translation = *target
            .origin_translation
            .get_or_insert(transform.translation);
        target.was_scripted = true;
        let previous_spin_rate = scenario_spin_rate(&target);
        target.profile.elapsed_s += time.delta_secs();
        let current_spin_rate = scenario_spin_rate(&target);
        target.profile.spin_angle_rad +=
            (previous_spin_rate + current_spin_rate) * 0.5 * time.delta_secs();

        let mut manual_velocity = Vec3::ZERO;
        if is_active.is_some()
            && remote_input.movement != Vec2::ZERO
            && let Ok(muzzle) = active_muzzle.single()
        {
            let barrel = muzzle.rotation() * Vec3::Y;
            let forward = Vec3::new(barrel.x, 0.0, barrel.z).normalize_or_zero();
            if forward != Vec3::ZERO {
                let right = forward.cross(Vec3::Y).normalize_or_zero();
                manual_velocity = (forward * remote_input.movement.y
                    + right * remote_input.movement.x)
                    .normalize_or_zero()
                    * manual_max_speed
                    * remote_input.boost_multiplier();
                target.manual_offset_bevy += manual_velocity * time.delta_secs();
            }
        }

        let linear = scenario_linear_velocity_bevy(&target) + manual_velocity;
        let angular = Vec3::Y * current_spin_rate;
        match linear_velocity {
            Some(mut value) => value.0 = linear,
            None => {
                commands.entity(entity).insert(LinearVelocity(linear));
            }
        }
        match angular_velocity {
            Some(mut value) => value.0 = angular,
            None => {
                commands.entity(entity).insert(AngularVelocity(angular));
            }
        }
        transform.translation = scenario_horizontal_position_bevy(&target, origin_translation)
            + target.manual_offset_bevy
            + Vec3::Y * (target.profile.vertical_offset_m + vertical_displacement(&target));
    }
}

fn drive_scenario_chassis_spin(
    mut targets: Query<&mut ScenarioTarget>,
    mut chassis: Query<(
        &mut Transform,
        &mut InfantryChassis,
        &ScenarioChassisGeometry,
    )>,
) {
    for (mut transform, mut chassis_state, geometry) in &mut chassis {
        let Ok(mut target) = targets.get_mut(geometry.target) else {
            continue;
        };
        if target.profile.enabled {
            transform.rotation =
                geometry.original_rotation * Quat::from_rotation_y(spin_yaw(&target));
        } else if target.restore_chassis {
            restore_chassis_pose(
                &mut transform,
                &mut chassis_state,
                geometry.original_rotation,
            );
            target.restore_chassis = false;
        }
    }
}

fn restore_chassis_pose(
    transform: &mut Transform,
    chassis: &mut InfantryChassis,
    original_rotation: Quat,
) {
    let (yaw, pitch, roll) = original_rotation.to_euler(EulerRot::YXZ);
    transform.rotation = original_rotation;
    chassis.yaw = yaw;
    chassis.pitch = pitch;
    chassis.roll = roll;
    chassis.yaw_velocity = 0.0;
}

fn record_parameter_event(
    state: &mut ScenarioUiState,
    target: &ScenarioTarget,
    action: &str,
) -> std::io::Result<()> {
    state.event_sequence += 1;
    let timestamp_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let profile = &target.profile;
    let event = serde_json::json!({
        "event_seq": state.event_sequence,
        "timestamp_ns": timestamp_ns,
        "action": action,
        "target": target.kind.label(),
        "enabled": profile.enabled,
        "armor_height_offsets_m": profile.armor_height_offsets_m,
        "armor_radii_m": profile.armor_radii_m,
        "horizontal": {
            "mode": profile.horizontal_mode.label(),
            "world_xy_offset_m": profile.world_xy_offset_m,
            "polynomial_x_range_m": profile.polynomial_x_range_m,
            "polynomial_y_range_m": profile.polynomial_y_range_m,
            "polynomial_x_power": profile.polynomial_x_power,
            "polynomial_y_power": profile.polynomial_y_power,
            "polynomial_frequency_hz": profile.polynomial_frequency_hz,
        },
        "vertical": {
            "mode": profile.vertical_mode.label(),
            "offset_m": profile.vertical_offset_m,
            "amplitude_m": profile.vertical_amplitude_m,
            "peak_speed_mps": profile.vertical_peak_speed_mps,
            "phase_rad": profile.vertical_phase_rad,
        },
        "spin": {
            "mode": profile.spin_mode.label(),
            "direction": profile.spin_direction.label(),
            "constant_speed_radps": profile.spin_speed_radps,
            "min_speed_radps": profile.spin_min_speed_radps,
            "max_speed_radps": profile.spin_max_speed_radps,
            "frequency_hz": profile.spin_frequency_hz,
        },
    });
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("parameter_events.jsonl")?;
    writeln!(file, "{event}")
}

fn target_mut<'a>(
    targets: &'a mut Query<&mut ScenarioTarget>,
    kind: ScenarioTargetKind,
) -> Option<Mut<'a, ScenarioTarget>> {
    targets.iter_mut().find(|target| target.kind == kind)
}

fn scenario_control_panel(
    mut contexts: EguiContexts,
    mut state: ResMut<ScenarioUiState>,
    mut targets: Query<&mut ScenarioTarget>,
    window: Single<&Window>,
) {
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };

    // The off-screen Talos render target reports a 1.0 native Egui scale. Use the desktop
    // window's DPI scale explicitly so a 310-point panel is 310 logical pixels on HiDPI X11.
    context.set_pixels_per_point(window.scale_factor());

    // The Egui context can inherit the fixed 1440x1080 Talos capture size even though the
    // preview window is smaller. Build the panel inside the actual window rectangle so its
    // right edge is not clipped on 1280x720/HiDPI desktop windows.
    let window_rect = egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(window.width(), window.height()),
    );
    let mut window_ui = egui::Ui::new(
        context.clone(),
        "scenario_control_window".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(window_rect),
    );
    egui::Panel::right("scenario_control_panel")
        .resizable(true)
        .default_size(310.0)
        .show_inside(&mut window_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Scenario Control");
            ui.label("All height labels use Talos world Z. The simulation applies them on Bevy Y.");
            ui.separator();

            ui.label("Target");
            for kind in ScenarioTargetKind::ALL {
                ui.radio_value(&mut state.selected_target, kind, kind.label());
            }

            let Some(mut target) = target_mut(&mut targets, state.selected_target) else {
                ui.colored_label(egui::Color32::YELLOW, "Target asset is still loading.");
                return;
            };

            let mut changed = false;
            changed |= ui
                .checkbox(&mut target.profile.enabled, "Enable scripted target")
                .changed();

            ui.separator();
            ui.heading("Armor pair height");
            ui.label("Pair A: lower baseline opposite pair");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut target.profile.armor_height_offsets_m[0])
                        .speed(0.005)
                        .range(-1.0..=1.0)
                        .suffix(" m"),
                )
                .changed();
            ui.label("Pair B: upper baseline opposite pair");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut target.profile.armor_height_offsets_m[1])
                        .speed(0.005)
                        .range(-1.0..=1.0)
                        .suffix(" m"),
                )
                .changed();
            ui.small("Changing a pair height is a geometry jump. Reset the visual tracker before using it as a clean experiment.");

            ui.separator();
            ui.heading("Armor pair radius");
            if target.profile.armor_radii_initialized {
                ui.label("Pair A: lower baseline opposite pair");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.armor_radii_m[0])
                            .speed(0.002)
                            .range(0.01..=2.0)
                            .suffix(" m"),
                    )
                    .changed();
                ui.label("Pair B: upper baseline opposite pair");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.armor_radii_m[1])
                            .speed(0.002)
                            .range(0.01..=2.0)
                            .suffix(" m"),
                    )
                    .changed();
                ui.small("Changing a radius moves armor rendering, collision and Talos truth together. Reset the visual tracker before evaluating the new geometry.");
            } else {
                ui.small("Armor geometry is loading; radii will initialize from the target asset.");
            }

            ui.separator();
            ui.heading("World XY motion");
            egui::ComboBox::from_id_salt("horizontal_motion_mode")
                .selected_text(target.profile.horizontal_mode.label())
                .show_ui(ui, |ui| {
                    for mode in HorizontalMotionMode::ALL {
                        changed |= ui
                            .selectable_value(&mut target.profile.horizontal_mode, mode, mode.label())
                            .changed();
                    }
                });
            if target.profile.horizontal_mode == HorizontalMotionMode::StaticOffset {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.world_xy_offset_m[0])
                            .speed(0.01)
                            .range(-20.0..=20.0)
                            .prefix("X offset ")
                            .suffix(" m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.world_xy_offset_m[1])
                            .speed(0.01)
                            .range(-20.0..=20.0)
                            .prefix("Y offset ")
                            .suffix(" m"),
                    )
                    .changed();
            } else {
                ui.label("X range");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_x_range_m[0])
                            .speed(0.01)
                            .range(-30.0..=30.0)
                            .suffix(" m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_x_range_m[1])
                            .speed(0.01)
                            .range(-30.0..=30.0)
                            .suffix(" m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_x_power)
                            .range(1..=8)
                            .prefix("X power "),
                    )
                    .changed();
                ui.label("Y range");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_y_range_m[0])
                            .speed(0.01)
                            .range(-30.0..=30.0)
                            .suffix(" m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_y_range_m[1])
                            .speed(0.01)
                            .range(-30.0..=30.0)
                            .suffix(" m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_y_power)
                            .range(1..=8)
                            .prefix("Y power "),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.polynomial_frequency_hz)
                            .speed(0.01)
                            .range(0.01..=5.0)
                            .prefix("cycle ")
                            .suffix(" Hz"),
                    )
                    .changed();
            }

            ui.separator();
            ui.heading("World Z motion");
            egui::ComboBox::from_id_salt("vertical_motion_mode")
                .selected_text(target.profile.vertical_mode.label())
                .show_ui(ui, |ui| {
                    for mode in VerticalMotionMode::ALL {
                        changed |= ui
                            .selectable_value(&mut target.profile.vertical_mode, mode, mode.label())
                            .changed();
                    }
                });
            changed |= ui
                .add(
                    egui::DragValue::new(&mut target.profile.vertical_offset_m)
                        .speed(0.005)
                        .range(-2.0..=2.0)
                        .prefix("offset ")
                        .suffix(" m"),
                )
                .changed();
            if target.profile.vertical_mode == VerticalMotionMode::Sine {
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.vertical_amplitude_m)
                            .speed(0.005)
                            .range(0.0..=2.0)
                            .prefix("amplitude ")
                            .suffix(" m"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.vertical_peak_speed_mps)
                            .speed(0.01)
                            .range(0.0..=10.0)
                            .prefix("peak speed ")
                            .suffix(" m/s"),
                    )
                    .changed();
            }

            ui.separator();
            ui.heading("Armor spin");
            egui::ComboBox::from_id_salt("spin_mode")
                .selected_text(target.profile.spin_mode.label())
                .show_ui(ui, |ui| {
                    for mode in SpinMode::ALL {
                        changed |= ui
                            .selectable_value(&mut target.profile.spin_mode, mode, mode.label())
                            .changed();
                    }
                });
            if target.profile.spin_mode != SpinMode::Off {
                egui::ComboBox::from_id_salt("spin_direction")
                    .selected_text(target.profile.spin_direction.label())
                    .show_ui(ui, |ui| {
                        for direction in SpinDirection::ALL {
                            changed |= ui
                                .selectable_value(
                                    &mut target.profile.spin_direction,
                                    direction,
                                    direction.label(),
                                )
                                .changed();
                        }
                    });
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.spin_speed_radps)
                            .speed(0.05)
                            .range(0.05..=30.0)
                            .prefix("constant ")
                            .suffix(" rad/s"),
                    )
                    .changed();
            }
            if target.profile.spin_mode == SpinMode::Variable {
                let min_speed_limit = target.profile.spin_max_speed_radps.max(0.05);
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.spin_max_speed_radps)
                            .speed(0.05)
                            .range(0.1..=30.0)
                            .prefix("high speed ")
                            .suffix(" rad/s"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.spin_min_speed_radps)
                            .speed(0.05)
                            .range(0.05..=min_speed_limit)
                            .prefix("low speed ")
                            .suffix(" rad/s"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut target.profile.spin_frequency_hz)
                            .speed(0.01)
                            .range(0.01..=10.0)
                            .prefix("cycle ")
                            .suffix(" Hz"),
                    )
                    .changed();
            }

            if ui.button("Reset motion phase").clicked() {
                target.profile.elapsed_s = 0.0;
                target.profile.spin_angle_rad = 0.0;
                changed = true;
            }

            if changed {
                target.profile.polynomial_x_range_m.sort_by(f32::total_cmp);
                target.profile.polynomial_y_range_m.sort_by(f32::total_cmp);
                target.profile.polynomial_x_power = target.profile.polynomial_x_power.clamp(1, 8);
                target.profile.polynomial_y_power = target.profile.polynomial_y_power.clamp(1, 8);
                target.profile.polynomial_frequency_hz =
                    target.profile.polynomial_frequency_hz.clamp(0.01, 5.0);
                target.profile.spin_min_speed_radps = target
                    .profile
                    .spin_min_speed_radps
                    .clamp(0.05, target.profile.spin_max_speed_radps.max(0.05));
                target.profile.spin_max_speed_radps = target.profile.spin_max_speed_radps.max(0.05);
                match record_parameter_event(&mut state, &target, "ui_update") {
                    Ok(()) => state.last_status = "Applied and recorded in parameter_events.jsonl".to_string(),
                    Err(error) => state.last_status = format!("Could not record parameter event: {error}"),
                }
            }
            ui.separator();
            ui.small(&state.last_status);
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn armor_pairs_follow_geometry_not_asset_root_names() {
        let vehicle = [
            (
                Entity::from_raw_u32(1).unwrap(),
                Vec3::new(-0.00015, -0.07542, -0.21971),
            ),
            (
                Entity::from_raw_u32(2).unwrap(),
                Vec3::new(-0.00015, -0.07542, 0.20382),
            ),
            (
                Entity::from_raw_u32(3).unwrap(),
                Vec3::new(0.21102, -0.06142, -0.00794),
            ),
            (
                Entity::from_raw_u32(4).unwrap(),
                Vec3::new(-0.21132, -0.06142, -0.00794),
            ),
        ];
        let hero = [
            (
                Entity::from_raw_u32(1).unwrap(),
                Vec3::new(-0.00159, 0.01780, -0.27742),
            ),
            (
                Entity::from_raw_u32(2).unwrap(),
                Vec3::new(-0.21012, -0.00220, -0.00773),
            ),
            (
                Entity::from_raw_u32(3).unwrap(),
                Vec3::new(0.00027, 0.01780, 0.26051),
            ),
            (
                Entity::from_raw_u32(4).unwrap(),
                Vec3::new(0.20880, -0.00220, -0.00918),
            ),
        ];

        let vehicle_pairs = assign_armor_pairs(&vehicle).unwrap();
        let hero_pairs = assign_armor_pairs(&hero).unwrap();
        let pair_of = |pairs: &Vec<(Entity, ArmorPair, Vec3)>, entity| {
            pairs.iter().find(|(id, _, _)| *id == entity).unwrap().1
        };
        assert_eq!(
            pair_of(&vehicle_pairs, vehicle[0].0),
            pair_of(&vehicle_pairs, vehicle[1].0)
        );
        assert_ne!(
            pair_of(&vehicle_pairs, vehicle[0].0),
            pair_of(&vehicle_pairs, vehicle[2].0)
        );
        assert_eq!(
            pair_of(&hero_pairs, hero[0].0),
            pair_of(&hero_pairs, hero[2].0)
        );
        assert_ne!(
            pair_of(&hero_pairs, hero[0].0),
            pair_of(&hero_pairs, hero[1].0)
        );
    }

    #[test]
    fn configured_pair_radius_preserves_height_and_sets_xz_distance() {
        let original = Vec3::new(0.12, -0.08, -0.16);

        let translated = armor_translation_at_radius(original, 0.32);

        assert!((translated.y - original.y).abs() < 1e-6);
        assert!((Vec2::new(translated.x, translated.z).length() - 0.32).abs() < 1e-6);
        assert!(
            Vec2::new(translated.x, translated.z)
                .normalize()
                .dot(Vec2::new(original.x, original.z).normalize())
                > 0.99999
        );
    }

    #[test]
    fn sine_vertical_motion_uses_peak_speed_as_the_derivative_bound() {
        let mut target = ScenarioTarget::new(ScenarioTargetKind::Infantry);
        target.profile.vertical_mode = VerticalMotionMode::Sine;
        target.profile.vertical_amplitude_m = 0.5;
        target.profile.vertical_peak_speed_mps = 2.0;
        target.profile.elapsed_s = std::f32::consts::FRAC_PI_4 * 0.5 / 2.0;

        assert!(
            (vertical_displacement(&target) - 0.5 * std::f32::consts::FRAC_PI_4.sin()).abs() < 1e-6
        );
    }

    #[test]
    fn polynomial_xy_loop_respects_world_ranges_powers_and_axis_mapping() {
        let mut target = ScenarioTarget::new(ScenarioTargetKind::Infantry);
        target.profile.horizontal_mode = HorizontalMotionMode::PolynomialLoop;
        target.profile.polynomial_x_range_m = [0.0, 4.0];
        target.profile.polynomial_y_range_m = [-2.0, 2.0];
        target.profile.polynomial_x_power = 1;
        target.profile.polynomial_y_power = 2;
        target.profile.polynomial_frequency_hz = 1.0;
        target.profile.elapsed_s = 0.25;

        let position = scenario_horizontal_position_bevy(&target, Vec3::new(0.0, 1.0, 0.0));
        let velocity = scenario_horizontal_velocity_bevy(&target);

        assert!(position.abs_diff_eq(Vec3::new(1.0, 1.0, -2.0), 1e-6));
        assert!(velocity.abs_diff_eq(
            Vec3::new(
                -4.0 * std::f32::consts::PI,
                0.0,
                -4.0 * std::f32::consts::PI
            ),
            1e-5
        ));
    }

    #[test]
    fn variable_spin_starts_fast_then_smoothly_slows_without_stalling() {
        let mut target = ScenarioTarget::new(ScenarioTargetKind::Hero);
        target.profile.spin_mode = SpinMode::Variable;
        target.profile.spin_min_speed_radps = 2.0;
        target.profile.spin_max_speed_radps = 6.0;
        target.profile.spin_frequency_hz = 1.0;

        assert!((scenario_spin_rate(&target) - 6.0).abs() < 1e-6);
        target.profile.elapsed_s = 0.5;
        assert!((scenario_spin_rate(&target) - 2.0).abs() < 1e-6);
        target.profile.elapsed_s = 1.0;
        assert!((scenario_spin_rate(&target) - 6.0).abs() < 1e-6);

        target.profile.spin_min_speed_radps = 0.0;
        target.profile.elapsed_s = 0.5;
        assert!(scenario_spin_rate(&target) >= 0.05);
    }

    #[test]
    fn spin_hotkeys_select_constant_variable_and_off_modes() {
        let mode_for = |key| {
            let mut keyboard = ButtonInput::<KeyCode>::default();
            keyboard.press(key);
            requested_spin_mode(&keyboard)
        };

        assert_eq!(mode_for(KeyCode::F10), Some(SpinMode::Constant));
        assert_eq!(mode_for(KeyCode::F11), Some(SpinMode::Variable));
        assert_eq!(mode_for(KeyCode::F12), Some(SpinMode::Off));
    }

    #[test]
    fn spin_only_profile_does_not_claim_vehicle_translation() {
        let mut profile = ScenarioProfile {
            enabled: true,
            spin_mode: SpinMode::Variable,
            ..default()
        };
        assert!(!has_scripted_translation(&profile));

        profile.world_xy_offset_m = [0.1, 0.0];
        assert!(has_scripted_translation(&profile));
        profile.world_xy_offset_m = [0.0, 0.0];
        profile.vertical_mode = VerticalMotionMode::Sine;
        assert!(has_scripted_translation(&profile));
        profile.vertical_mode = VerticalMotionMode::Static;
        profile.horizontal_mode = HorizontalMotionMode::PolynomialLoop;
        assert!(has_scripted_translation(&profile));
    }

    #[test]
    fn hot_spin_parameter_updates_preserve_the_accumulated_angle() {
        let mut target = ScenarioTarget::new(ScenarioTargetKind::Hero);
        target.profile.spin_angle_rad = 3.2;
        target.profile.spin_min_speed_radps = 1.0;
        target.profile.spin_max_speed_radps = 9.0;

        assert_eq!(spin_yaw(&target), 3.2);
    }

    #[test]
    fn scripted_target_exposes_the_same_kinematics_to_physics() {
        let mut app = App::new();
        app.insert_resource(Time::<Fixed>::from_hz(100.0));
        app.add_systems(FixedUpdate, drive_scenario_targets);

        let mut target = ScenarioTarget::new(ScenarioTargetKind::Infantry);
        target.profile.enabled = true;
        target.profile.vertical_mode = VerticalMotionMode::Sine;
        target.profile.vertical_amplitude_m = 1.0;
        target.profile.vertical_peak_speed_mps = 2.0;
        target.profile.spin_mode = SpinMode::Constant;
        target.profile.spin_speed_radps = 3.0;
        let entity = app
            .world_mut()
            .spawn((Transform::default(), RigidBody::Dynamic, target))
            .id();

        app.world_mut().run_schedule(FixedUpdate);

        assert_eq!(
            app.world().get::<RigidBody>(entity),
            Some(&RigidBody::Kinematic)
        );
        assert_eq!(
            app.world().get::<LinearVelocity>(entity),
            Some(&LinearVelocity(Vec3::Y * 2.0))
        );
        assert_eq!(
            app.world().get::<AngularVelocity>(entity),
            Some(&AngularVelocity(Vec3::Y * 3.0))
        );
        assert!(
            app.world()
                .get::<CustomPositionIntegration>(entity)
                .is_some()
        );
    }

    #[test]
    fn restoring_a_scripted_chassis_also_resets_controller_state() {
        let original_rotation = Quat::from_euler(EulerRot::YXZ, 0.2, -0.1, 0.05);
        let mut transform = Transform::from_rotation(Quat::from_rotation_y(1.2));
        let mut chassis = InfantryChassis {
            yaw: 1.2,
            yaw_velocity: 4.0,
            pitch: 0.3,
            roll: -0.2,
        };

        restore_chassis_pose(&mut transform, &mut chassis, original_rotation);

        assert_eq!(transform.rotation, original_rotation);
        assert!((chassis.yaw - 0.2).abs() < 1e-6);
        assert!((chassis.pitch + 0.1).abs() < 1e-6);
        assert!((chassis.roll - 0.05).abs() < 1e-6);
        assert_eq!(chassis.yaw_velocity, 0.0);
    }
}
