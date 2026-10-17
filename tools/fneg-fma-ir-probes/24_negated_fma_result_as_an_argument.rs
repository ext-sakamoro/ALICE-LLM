// expect-ir: flag
// expect-syn: flag
// targets: any
// the shape found in ALICE-Physics sim_field.rs ScalarField3D::splat
pub fn probe(d: f32, r: f32) -> f32 {
    let t = d.mul_add(-r, 1.0);
    t * t * 2.0f32.mul_add(-t, 3.0)
}
