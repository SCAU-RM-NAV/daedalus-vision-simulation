use crate::layout::*;
use crate::shm::{ShmError, ShmRegion};
use crate::triple_buffer::{load_u64, publish_slot, store_u64};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ShmPublisher {
    meta_region: ShmRegion,
    image_pool: ShmRegion,
    current_buffer_id: u8,
    last_published_frame_seq: u64,
}

impl ShmPublisher {
    pub fn create() -> Result<Self, ShmError> {
        let mut meta_region = ShmRegion::create(SHM_NAME_META, size_of::<ShmMetaRegion>())?;
        let image_pool = ShmRegion::create(SHM_NAME_IMAGE_POOL, IMAGE_POOL_SIZE)?;
        unsafe {
            let meta = meta_region.as_mut::<ShmMetaRegion>();
            *meta = ShmMetaRegion::default();
            meta.header.created_ns = Self::now_ns();
            meta.header.heartbeat_ns = meta.header.created_ns;
        }
        Ok(Self {
            meta_region,
            image_pool,
            current_buffer_id: 0,
            last_published_frame_seq: 0,
        })
    }

    pub fn try_publish_frame(&mut self, mut data: FrameData, rgb: &[u8]) -> bool {
        if rgb.len() != IMAGE_SIZE
            || !frame_is_valid(&data)
            || data.image.frame_seq <= self.last_published_frame_seq
        {
            return false;
        }
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            if self.last_published_frame_seq != 0
                && load_u64(&meta.frame.consumer_ack_frame_seq) < self.last_published_frame_seq
            {
                return false;
            }
            let buffer_id = self.current_buffer_id;
            self.current_buffer_id = (self.current_buffer_id + 1) % 3;
            let dst = self
                .image_pool
                .as_ptr()
                .add(buffer_id as usize * IMAGE_SIZE);
            std::ptr::copy_nonoverlapping(rgb.as_ptr(), dst, IMAGE_SIZE);
            data.image.buffer_id = buffer_id;
            let slot = meta.frame.write_index as usize;
            meta.frame.slots[slot] = data;
            publish_slot(&mut meta.frame.state, &mut meta.frame.write_index);
            self.last_published_frame_seq = data.image.frame_seq;
        }
        true
    }

    pub fn publish_hit_event(&mut self, mut event: HitEvent) {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            let next = load_u64(&meta.hit_events.write_seq).saturating_add(1);
            let acknowledged = load_u64(&meta.hit_events.consumer_ack_seq);
            if next.saturating_sub(acknowledged) > HIT_EVENT_CAPACITY as u64 {
                let overflow = load_u64(&meta.hit_events.overflow_count).saturating_add(1);
                store_u64(&mut meta.hit_events.overflow_count, overflow);
            }
            event.event_seq = next;
            meta.hit_events.events[(next as usize) % HIT_EVENT_CAPACITY] = event;
            store_u64(&mut meta.hit_events.write_seq, next);
        }
    }

    pub fn update_heartbeat(&mut self) {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            meta.header.heartbeat_ns = Self::now_ns();
        }
    }

    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0)
    }
}

pub fn frame_is_valid(data: &FrameData) -> bool {
    let seq = data.image.frame_seq;
    let timestamp = data.image.timestamp_ns;
    seq != 0
        && timestamp != 0
        && data.image.width == IMAGE_WIDTH
        && data.image.height == IMAGE_HEIGHT
        && data.image.format == 0
        && data.camera.frame_seq == seq
        && data.camera.timestamp_ns == timestamp
        && data.camera.width == IMAGE_WIDTH
        && data.camera.height == IMAGE_HEIGHT
        && data.gimbal_world.frame_seq == seq
        && data.gimbal_world.timestamp_ns == timestamp
        && data.feedback.frame_seq == seq
        && data.feedback.timestamp_ns == timestamp
        && data.truth.frame_seq == seq
        && data.truth.timestamp_ns == timestamp
        && data.truth.target_count as usize <= GROUND_TRUTH_MAX_TARGETS
        && data.truth.rune_count as usize <= GROUND_TRUTH_MAX_RUNES
        && data.camera.intrinsics[0].is_finite()
        && data.camera.intrinsics[0] > 0.0
        && data.camera.intrinsics[1].is_finite()
        && data.camera.intrinsics[1] > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_mismatched_frame_sequences() {
        let mut frame = FrameData::default();
        frame.image.frame_seq = 9;
        frame.image.timestamp_ns = 10;
        frame.image.width = IMAGE_WIDTH;
        frame.image.height = IMAGE_HEIGHT;
        frame.camera.frame_seq = 9;
        frame.camera.timestamp_ns = 10;
        frame.camera.width = IMAGE_WIDTH;
        frame.camera.height = IMAGE_HEIGHT;
        frame.camera.intrinsics = [1.0, 1.0, 0.0, 0.0];
        frame.gimbal_world.frame_seq = 9;
        frame.gimbal_world.timestamp_ns = 10;
        frame.feedback.frame_seq = 9;
        frame.feedback.timestamp_ns = 10;
        frame.truth.frame_seq = 9;
        frame.truth.timestamp_ns = 10;
        assert!(frame_is_valid(&frame));
        frame.feedback.frame_seq = 8;
        assert!(!frame_is_valid(&frame));
    }
}
