// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(k: u8, a: f32, b: f32, c: f32) -> f32 {
    -(match k {
        0 => a.mul_add(b, c),
        _ => b.mul_add(c, a),
    })
}
