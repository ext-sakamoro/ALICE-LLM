// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f64 {
    -f64::from(a.mul_add(b, c))
}
