// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(a: f64, b: f64, c: f64) -> f64 {
    -a.mul_add(b, c)
}
