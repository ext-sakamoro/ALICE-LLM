// expect-ir: clean
// expect-syn: clean
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    let t = a.mul_add(b, c);
    let t = t + 1.0;
    -t
}
