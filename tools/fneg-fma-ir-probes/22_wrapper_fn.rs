// expect-ir: flag
// expect-syn: miss
// targets: any
fn m(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    -m(a, b, c)
}
