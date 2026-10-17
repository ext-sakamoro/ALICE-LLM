// expect-ir: flag
// expect-syn: flag
// targets: any
const SIGN: u32 = 1 << 31;

pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    f32::from_bits(a.mul_add(b, c).to_bits() ^ SIGN)
}
