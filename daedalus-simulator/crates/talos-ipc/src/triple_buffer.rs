use crate::layout::{TRIPLE_INDEX_MASK, TRIPLE_NEW};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

#[inline]
pub fn load_u8(value: &u8) -> u8 {
    unsafe { (&*(value as *const u8 as *const AtomicU8)).load(Ordering::Acquire) }
}

#[inline]
pub fn swap_u8(value: &mut u8, next: u8) -> u8 {
    unsafe { (&*(value as *mut u8 as *mut AtomicU8)).swap(next, Ordering::AcqRel) }
}

#[inline]
pub fn compare_exchange_u8(value: &mut u8, expected: u8, next: u8) -> Result<u8, u8> {
    unsafe {
        (&*(value as *mut u8 as *mut AtomicU8)).compare_exchange_weak(
            expected,
            next,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
    }
}

#[inline]
pub fn load_u64(value: &u64) -> u64 {
    unsafe { (&*(value as *const u64 as *const AtomicU64)).load(Ordering::Acquire) }
}

#[inline]
pub fn store_u64(value: &mut u64, next: u64) {
    unsafe { (&*(value as *mut u64 as *mut AtomicU64)).store(next, Ordering::Release) }
}

pub fn consume_slot(state: &mut u8, read_index: &mut u8) -> Option<u8> {
    let expected = load_u8(state);
    if expected & TRIPLE_NEW == 0 {
        return None;
    }
    let slot = expected & TRIPLE_INDEX_MASK;
    let previous_read = *read_index;
    compare_exchange_u8(state, expected, previous_read).ok()?;
    *read_index = slot;
    Some(slot)
}

pub fn publish_slot(state: &mut u8, write_index: &mut u8) -> u8 {
    let slot = *write_index;
    let old = swap_u8(state, slot | TRIPLE_NEW);
    *write_index = old & TRIPLE_INDEX_MASK;
    slot
}
