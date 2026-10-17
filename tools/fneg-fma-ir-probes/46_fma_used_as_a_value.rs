// expect-ir: flag
// expect-syn: refused
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    let f: fn(f32, f32, f32) -> f32 = f32::mul_add;
    -f(a, b, c)
}
