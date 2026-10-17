// expect-ir: flag
// expect-syn: miss
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    let t = (a.mul_add(b, c), b);
    -t.0
}
