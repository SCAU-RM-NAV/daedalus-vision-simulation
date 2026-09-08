use crate::layout::*;
use crate::shm::{ShmError, ShmRegion};
use crate::triple_buffer::consume_slot;

pub struct ShmSubscriber {
    meta_region: ShmRegion,
}

impl ShmSubscriber {
    pub fn connect() -> Result<Self, ShmError> {
        let meta_region = ShmRegion::open(SHM_NAME_META, size_of::<ShmMetaRegion>())?;
        unsafe {
            let meta = meta_region.as_ref::<ShmMetaRegion>();
            if meta.header.magic != SHM_MAGIC
                || meta.header.version != SHM_VERSION
                || meta.header.meta_bytes != size_of::<ShmMetaRegion>() as u32
                || meta.header.image_width != IMAGE_WIDTH
                || meta.header.image_height != IMAGE_HEIGHT
                || meta.header.image_channels != IMAGE_CHANNELS
            {
                return Err(ShmError::InvalidSize);
            }
        }
        Ok(Self { meta_region })
    }

    pub fn recv_vision_command(&mut self) -> Option<VisionCommand> {
        unsafe {
            let meta = self.meta_region.as_mut::<ShmMetaRegion>();
            let slot =
                consume_slot(&mut meta.command.state, &mut meta.command.read_index)? as usize;
            let command = meta.command.slots[slot];
            command.is_valid().then_some(command)
        }
    }
}
