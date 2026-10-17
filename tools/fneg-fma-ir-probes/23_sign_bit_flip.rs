// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    f32::from_bits(a.mul_add(b, c).to_bits() ^ 0x8000_0000)
}
