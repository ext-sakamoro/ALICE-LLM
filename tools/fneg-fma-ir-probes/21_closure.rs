// expect-ir: flag
// expect-syn: miss
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    let m = |x: f32| x.mul_add(b, c);
    -m(a)
}
