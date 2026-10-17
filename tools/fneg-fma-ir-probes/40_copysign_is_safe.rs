// expect-ir: clean
// expect-syn: clean
// targets: any
// lowers to fneg(fabs(fma)): fabs blocks the sink
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c).copysign(-1.0)
}
