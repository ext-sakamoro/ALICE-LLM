// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(k: bool, a: f32, b: f32, c: f32) -> f32 {
    -(if k { a.mul_add(b, c) } else { b.mul_add(a, c) })
}
