// expect-ir: flag
// expect-syn: flag
// targets: any
use std::ops::Neg;

pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c).neg()
}
