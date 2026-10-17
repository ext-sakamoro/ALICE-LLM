// expect-ir: flag
// expect-syn: flag
// targets: any
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    -{
        let s = a * 2.0;
        s.mul_add(b, c)
    }
}
