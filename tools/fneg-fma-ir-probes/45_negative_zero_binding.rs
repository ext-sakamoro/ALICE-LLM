// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    let z = -0.0f32;
    z - a.mul_add(b, c)
}
