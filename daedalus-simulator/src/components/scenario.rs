use bevy::prelude::*;

/// Identifies a selectable scripted target in the scenario control panel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioTargetKind {
    Infantry,
    Hero,
}

impl ScenarioTargetKind {
    pub const ALL: [Self; 2] = [Self::Infantry, Self::Hero];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Infantry => "Infantry",
            Self::Hero => "Hero",
        }
    }
}

/// A vertical trajectory uses the configured peak velocity instead of a frequency, which makes
/// the control panel directly expressible in metres and metres per second.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerticalMotionMode {
    Static,
    Sine,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HorizontalMotionMode {
    StaticOffset,
    PolynomialLoop,
}

impl HorizontalMotionMode {
    pub const ALL: [Self; 2] = [Self::StaticOffset, Self::PolynomialLoop];

    pub const fn label(self) -> &'static str {
        match self {
            Self::StaticOffset => "Static offset",
            Self::PolynomialLoop => "Polynomial loop",
        }
    }
}

impl VerticalMotionMode {
    pub const ALL: [Self; 2] = [Self::Static, Self::Sine];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Static => "Static",
            Self::Sine => "Sine",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpinMode {
    Off,
    Constant,
    Variable,
}

impl SpinMode {
    pub const ALL: [Self; 3] = [Self::Off, Self::Constant, Self::Variable];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Constant => "Constant",
            Self::Variable => "Variable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpinDirection {
    CounterClockwise,
    Clockwise,
}

impl SpinDirection {
    pub const ALL: [Self; 2] = [Self::CounterClockwise, Self::Clockwise];

    pub const fn label(self) -> &'static str {
        match self {
            Self::CounterClockwise => "CCW",
            Self::Clockwise => "CW",
        }
    }

    pub const fn sign(self) -> f32 {
        match self {
            Self::CounterClockwise => 1.0,
            Self::Clockwise => -1.0,
        }
    }
}

/// The four vehicle armor roots are grouped into two geometrically opposite pairs. Pair A is the
/// lower baseline pair when the asset has staggered armor heights; pair B is the other pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArmorPair {
    A,
    B,
}

impl ArmorPair {
    pub const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
        }
    }
}

/// Runtime-controlled target parameters. `height_offset_m` is local to each armor pair; all
/// vertical movement is in Bevy Y and becomes ROS/Talos world Z at capture time.
#[derive(Clone, Debug)]
pub struct ScenarioProfile {
    pub enabled: bool,
    pub armor_height_offsets_m: [f32; 2],
    pub armor_radii_m: [f32; 2],
    pub armor_radii_initialized: bool,
    pub world_xy_offset_m: [f32; 2],
    pub horizontal_mode: HorizontalMotionMode,
    pub polynomial_x_range_m: [f32; 2],
    pub polynomial_y_range_m: [f32; 2],
    pub polynomial_x_power: u32,
    pub polynomial_y_power: u32,
    pub polynomial_frequency_hz: f32,
    pub vertical_offset_m: f32,
    pub vertical_mode: VerticalMotionMode,
    pub vertical_amplitude_m: f32,
    pub vertical_peak_speed_mps: f32,
    pub vertical_phase_rad: f32,
    pub spin_mode: SpinMode,
    pub spin_direction: SpinDirection,
    pub spin_speed_radps: f32,
    pub spin_min_speed_radps: f32,
    pub spin_max_speed_radps: f32,
    pub spin_frequency_hz: f32,
    pub elapsed_s: f32,
    pub spin_angle_rad: f32,
}

impl Default for ScenarioProfile {
    fn default() -> Self {
        Self {
            enabled: false,
            armor_height_offsets_m: [0.0, 0.0],
            armor_radii_m: [0.0, 0.0],
            armor_radii_initialized: false,
            world_xy_offset_m: [0.0, 0.0],
            horizontal_mode: HorizontalMotionMode::StaticOffset,
            polynomial_x_range_m: [-2.0, 2.0],
            polynomial_y_range_m: [-2.0, 2.0],
            polynomial_x_power: 1,
            polynomial_y_power: 1,
            polynomial_frequency_hz: 0.2,
            vertical_offset_m: 0.0,
            vertical_mode: VerticalMotionMode::Static,
            vertical_amplitude_m: 0.0,
            vertical_peak_speed_mps: 0.0,
            vertical_phase_rad: 0.0,
            spin_mode: SpinMode::Off,
            spin_direction: SpinDirection::CounterClockwise,
            spin_speed_radps: 6.0,
            spin_min_speed_radps: 2.0,
            spin_max_speed_radps: 6.0,
            spin_frequency_hz: 0.5,
            elapsed_s: 0.0,
            spin_angle_rad: 0.0,
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct ScenarioTarget {
    pub kind: ScenarioTargetKind,
    pub profile: ScenarioProfile,
    pub origin_translation: Option<Vec3>,
    pub was_scripted: bool,
    pub restore_chassis: bool,
    /// Manual I/J/K/L displacement accumulated while scripted motion is enabled.
    pub manual_offset_bevy: Vec3,
}

impl ScenarioTarget {
    pub fn new(kind: ScenarioTargetKind) -> Self {
        Self {
            kind,
            profile: ScenarioProfile::default(),
            origin_translation: None,
            was_scripted: false,
            restore_chassis: false,
            manual_offset_bevy: Vec3::ZERO,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ScenarioArmorGeometry {
    pub target: Entity,
    pub pair: ArmorPair,
    pub original_translation: Vec3,
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ScenarioChassisGeometry {
    pub target: Entity,
    pub original_rotation: Quat,
}
