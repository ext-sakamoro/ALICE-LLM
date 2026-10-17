// expect-ir: flag
// expect-syn: flag
// targets: x86_64
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
pub unsafe fn probe(a: f32, b: f32, c: f32) -> f32 {
    use std::arch::x86_64::*;
    _mm_cvtss_f32(_mm_mul_ps(_mm_fmadd_ps(_mm_set1_ps(a), _mm_set1_ps(b), _mm_set1_ps(c)), _mm_set1_ps(-1.0)))
}
