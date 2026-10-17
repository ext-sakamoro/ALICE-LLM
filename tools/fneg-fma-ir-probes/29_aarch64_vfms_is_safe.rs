// expect-ir: clean
// expect-syn: clean
// targets: aarch64
#[cfg(target_arch = "aarch64")]
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    use std::arch::aarch64::{vdupq_n_f32, vfmsq_f32, vgetq_lane_f32};
    unsafe { vgetq_lane_f32::<0>(vfmsq_f32(vdupq_n_f32(c), vdupq_n_f32(a), vdupq_n_f32(b))) }
}
