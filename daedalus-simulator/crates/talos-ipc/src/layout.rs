use core::mem::{offset_of, size_of};

pub const IMAGE_WIDTH: u32 = 1440;
pub const IMAGE_HEIGHT: u32 = 1080;
pub const IMAGE_CHANNELS: u32 = 3;
pub const IMAGE_SIZE: usize = (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS) as usize;
pub const IMAGE_POOL_SIZE: usize = IMAGE_SIZE * 3;
pub const SHM_MAGIC: u32 = 0x5441_4c05;
pub const SHM_VERSION: u32 = 3;
pub const SHM_NAME_META: &str = "talos_ipc_meta";
pub const SHM_NAME_IMAGE_POOL: &str = "talos_ipc_image_pool";
pub const TRIPLE_NEW: u8 = 0x80;
pub const TRIPLE_INDEX_MASK: u8 = 0x03;
pub const GROUND_TRUTH_MAX_TARGETS: usize = 16;
pub const GROUND_TRUTH_MAX_RUNES: usize = 4;
pub const HIT_EVENT_CAPACITY: usize = 256;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct ShmHeader {
    pub magic: u32,
    pub version: u32,
    pub meta_bytes: u32,
    pub image_width: u32,
    pub image_height: u32,
    pub image_channels: u32,
    pub created_ns: u64,
    pub heartbeat_ns: u64,
    pub reserved: [u8; 24],
}

impl Default for ShmHeader {
    fn default() -> Self {
        Self {
            magic: SHM_MAGIC,
            version: SHM_VERSION,
            meta_bytes: size_of::<ShmMetaRegion>() as u32,
            image_width: IMAGE_WIDTH,
            image_height: IMAGE_HEIGHT,
            image_channels: IMAGE_CHANNELS,
            created_ns: 0,
            heartbeat_ns: 0,
            reserved: [0; 24],
        }
    }
}
const _: () = assert!(size_of::<ShmHeader>() == 64);

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageMeta {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
    pub buffer_id: u8,
    pub format: u8,
    pub reserved: [u8; 6],
}
const _: () = assert!(size_of::<ImageMeta>() == 32);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Pose {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub position_m: [f32; 3],
    pub quaternion_wxyz: [f32; 4],
    pub reserved: [u8; 20],
}
const _: () = assert!(size_of::<Pose>() == 64);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct CameraCalibration {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub width: u32,
    pub height: u32,
    pub intrinsics: [f32; 4],
    pub distortion: [f32; 8],
    pub r_camera2gimbal_row_major: [f32; 9],
    pub t_camera2gimbal_m: [f32; 3],
    pub reserved: [u8; 136],
}
impl Default for CameraCalibration {
    fn default() -> Self {
        Self {
            frame_seq: 0,
            timestamp_ns: 0,
            width: 0,
            height: 0,
            intrinsics: [0.0; 4],
            distortion: [0.0; 8],
            r_camera2gimbal_row_major: [0.0; 9],
            t_camera2gimbal_m: [0.0; 3],
            reserved: [0; 136],
        }
    }
}
const _: () = assert!(size_of::<CameraCalibration>() == 256);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct GimbalFeedback {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub yaw_velocity_radps: f32,
    pub pitch_velocity_radps: f32,
    pub bullet_speed_mps: f32,
    pub projectile_count: u32,
    pub camp: u8,
    pub robot_type: u8,
    pub mode: u8,
    pub simulation_subscription_enabled: u8,
    pub reserved0: u32,
    pub last_command_seq: u64,
    pub reserved: [u8; 8],
}
const _: () = assert!(size_of::<GimbalFeedback>() == 64);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct GroundTruthTarget {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub target_id: u32,
    pub team: u8,
    pub armor_label: u8,
    pub robot_type: u8,
    pub is_outpost: u8,
    pub position_m: [f32; 3],
    pub quaternion_wxyz: [f32; 4],
    pub velocity_mps: [f32; 3],
    pub yaw_rate_radps: f32,
    pub reserved: [u8; 8],
}
const _: () = assert!(size_of::<GroundTruthTarget>() == 128);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct GroundTruthRune {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub rune_id: u32,
    pub team: u8,
    pub mode: u8,
    pub mechanism_state: u8,
    pub direction: i8,
    pub center_m: [f32; 3],
    pub quaternion_wxyz: [f32; 4],
    pub radius_m: f32,
    pub angle_rad: f32,
    pub angular_velocity_radps: f32,
    pub sine_amplitude: f32,
    pub sine_omega: f32,
    pub sine_phase: f32,
    pub sine_offset: f32,
    pub relative_time_s: f32,
    pub active_blade_id: i32,
    pub target_activations: [u8; 5],
    pub reserved: [u8; 35],
}
impl Default for GroundTruthRune {
    fn default() -> Self {
        Self {
            frame_seq: 0,
            timestamp_ns: 0,
            rune_id: 0,
            team: 0,
            mode: 0,
            mechanism_state: 0,
            direction: 0,
            center_m: [0.0; 3],
            quaternion_wxyz: [0.0; 4],
            radius_m: 0.0,
            angle_rad: 0.0,
            angular_velocity_radps: 0.0,
            sine_amplitude: 0.0,
            sine_omega: 0.0,
            sine_phase: 0.0,
            sine_offset: 0.0,
            relative_time_s: 0.0,
            active_blade_id: -1,
            target_activations: [0; 5],
            reserved: [0; 35],
        }
    }
}
const _: () = assert!(size_of::<GroundTruthRune>() == 128);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct GroundTruthBatch {
    pub frame_seq: u64,
    pub timestamp_ns: u64,
    pub target_count: u32,
    pub rune_count: u32,
    pub targets: [GroundTruthTarget; GROUND_TRUTH_MAX_TARGETS],
    pub runes: [GroundTruthRune; GROUND_TRUTH_MAX_RUNES],
    pub reserved: [u8; 48],
}

impl Default for GroundTruthBatch {
    fn default() -> Self {
        Self {
            frame_seq: 0,
            timestamp_ns: 0,
            target_count: 0,
            rune_count: 0,
            targets: [GroundTruthTarget::default(); GROUND_TRUTH_MAX_TARGETS],
            runes: [GroundTruthRune::default(); GROUND_TRUTH_MAX_RUNES],
            reserved: [0; 48],
        }
    }
}
const _: () = assert!(size_of::<GroundTruthBatch>() == 2688);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameData {
    pub image: ImageMeta,
    pub camera: CameraCalibration,
    pub gimbal_world: Pose,
    pub feedback: GimbalFeedback,
    pub truth: GroundTruthBatch,
}
const _: () = assert!(size_of::<FrameData>() == 3136);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct FrameTripleBuffer {
    pub state: u8,
    pub write_index: u8,
    pub read_index: u8,
    pub reserved0: [u8; 5],
    pub consumer_ack_frame_seq: u64,
    pub reserved1: [u8; 48],
    pub slots: [FrameData; 3],
}
impl Default for FrameTripleBuffer {
    fn default() -> Self {
        Self {
            state: 1,
            write_index: 0,
            read_index: 2,
            reserved0: [0; 5],
            consumer_ack_frame_seq: 0,
            reserved1: [0; 48],
            slots: [FrameData::default(); 3],
        }
    }
}
const _: () = assert!(size_of::<FrameTripleBuffer>() == 9472);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct VisionCommand {
    pub frame_seq: u64,
    pub command_seq: u64,
    pub timestamp_ns: u64,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub yaw_velocity_radps: f32,
    pub pitch_velocity_radps: f32,
    pub yaw_acceleration_radps2: f32,
    pub pitch_acceleration_radps2: f32,
    pub distance_m: f32,
    pub control: u8,
    pub fire: u8,
    pub reserved: [u8; 10],
}

impl VisionCommand {
    pub fn is_valid(&self) -> bool {
        self.frame_seq != 0
            && self.command_seq != 0
            && self.timestamp_ns != 0
            && self.control <= 1
            && self.fire <= 1
            && [
                self.yaw_rad,
                self.pitch_rad,
                self.yaw_velocity_radps,
                self.pitch_velocity_radps,
                self.yaw_acceleration_radps2,
                self.pitch_acceleration_radps2,
                self.distance_m,
            ]
            .iter()
            .all(|value| value.is_finite())
    }
}
const _: () = assert!(size_of::<VisionCommand>() == 64);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct CommandTripleBuffer {
    pub state: u8,
    pub write_index: u8,
    pub read_index: u8,
    pub reserved: [u8; 61],
    pub slots: [VisionCommand; 3],
}
impl Default for CommandTripleBuffer {
    fn default() -> Self {
        Self {
            state: 1,
            write_index: 0,
            read_index: 2,
            reserved: [0; 61],
            slots: [VisionCommand::default(); 3],
        }
    }
}
const _: () = assert!(size_of::<CommandTripleBuffer>() == 256);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct HitEvent {
    pub event_seq: u64,
    pub command_seq: u64,
    pub hit_timestamp_ns: u64,
    pub target_id: u32,
    pub blade_id: i32,
    pub hit_type: u8,
    pub correct: u8,
    pub outcome: u8,
    pub reserved0: u8,
    pub reserved: [u8; 24],
}
const _: () = assert!(size_of::<HitEvent>() == 64);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct HitEventRing {
    pub write_seq: u64,
    pub consumer_ack_seq: u64,
    pub overflow_count: u64,
    pub reserved: [u8; 40],
    pub events: [HitEvent; HIT_EVENT_CAPACITY],
}

impl Default for HitEventRing {
    fn default() -> Self {
        Self {
            write_seq: 0,
            consumer_ack_seq: 0,
            overflow_count: 0,
            reserved: [0; 40],
            events: [HitEvent::default(); HIT_EVENT_CAPACITY],
        }
    }
}
const _: () = assert!(size_of::<HitEventRing>() == 16448);

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShmMetaRegion {
    pub header: ShmHeader,
    pub frame: FrameTripleBuffer,
    pub command: CommandTripleBuffer,
    pub hit_events: HitEventRing,
}
const _: () = assert!(size_of::<ShmMetaRegion>() == 26240);
const _: () = assert!(offset_of!(ShmMetaRegion, frame) == 64);
const _: () = assert!(offset_of!(ShmMetaRegion, command) == 9536);
const _: () = assert!(offset_of!(ShmMetaRegion, hit_events) == 9792);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_layout_is_stable_for_cpp_consumers() {
        assert_eq!(SHM_VERSION, 3);
        assert_eq!(size_of::<ShmHeader>(), 64);
        assert_eq!(size_of::<FrameData>(), 3136);
        assert_eq!(size_of::<VisionCommand>(), 64);
        assert_eq!(size_of::<ShmMetaRegion>(), 26240);
    }

    #[test]
    fn vision_command_rejects_unsequenced_commands() {
        assert!(!VisionCommand::default().is_valid());
    }
}
