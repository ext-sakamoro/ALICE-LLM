//! Oracle for `lawmath::ln_1p` (Kahan's form over `alice-det-math`'s `ln`)
//!
//! The reference is `f64::ln_1p` of the same `f32` input, rounded once to
//! `f32`: the platform's double-precision `log1p` is accurate to well under
//! one `f64` ulp, which is about 2^-29 of an `f32` ulp, so its rounding to
//! `f32` is the correctly rounded result except in ties that the sample
//! sets below do not hit. It is a reference, not the code under test.
//!
//! The ranges are the ones the call sites see: softplus of `exp(a)` for
//! `a` in `[-20, 20]` (src/llama3.rs, src/gpu.rs: arguments in
//! `[2.06e-9, 4.85e8]`), bigram counts (src/speculative_dspark.rs:
//! non-negative integers), plus the edges (`|x| < 1e-6`, `x` near -1, ±0,
//! NaN, ±inf).

use alice_llm::lawmath::ln_1p;

/// Distance in units in the last place of `f32` between `got` and the
/// correctly rounded `want` (0 when both are the same value)
fn ulps(got: f32, want: f32) -> u32 {
    if got.to_bits() == want.to_bits() {
        return 0;
    }
    let key = |x: f32| {
        let b = x.to_bits() as i32;
        if b < 0 {
            i32::MIN - b
        } else {
            b
        }
    };
    key(got).abs_diff(key(want))
}

#[expect(
    clippy::disallowed_methods,
    reason = "reference value, not the code under test"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "rounding the f64 reference to f32 is the point"
)]
fn reference(x: f32) -> f32 {
    f64::from(x).ln_1p() as f32
}

fn worst(xs: impl Iterator<Item = f32>) -> (u32, f32, usize) {
    let mut worst = (0, 0.0, 0);
    for x in xs {
        let u = ulps(ln_1p(x), reference(x));
        if u > worst.0 {
            worst.0 = u;
            worst.1 = x;
        }
        worst.2 += 1;
    }
    worst
}

/// `count` points spaced evenly in log scale over `[lo, hi]` (both > 0)
fn log_spaced(lo: f32, hi: f32, count: u32) -> impl Iterator<Item = f32> {
    let (a, b) = (lo.to_bits(), hi.to_bits());
    let step = ((b - a) / count).max(1);
    (0..=count)
        .map(move |i| f32::from_bits(a + i * step))
        .filter(move |x| *x <= hi)
}

/// The accuracy bound this module states for `ln_1p`, in `f32` ulps
const BOUND_ULPS: u32 = 2;

#[test]
fn tiny_arguments_keep_full_relative_accuracy() {
    let pos = worst(log_spaced(1e-30, 1e-6, 20_000));
    let neg = worst(log_spaced(1e-30, 1e-6, 20_000).map(|x| -x));
    assert!(
        pos.2 > 10_000 && neg.2 > 10_000,
        "sampled {} / {}",
        pos.2,
        neg.2
    );
    assert!(pos.0 <= BOUND_ULPS, "{} ulps at {:e}", pos.0, pos.1);
    assert!(neg.0 <= BOUND_ULPS, "{} ulps at {:e}", neg.0, neg.1);
}

#[test]
fn the_softplus_range_is_within_the_bound() {
    let w = worst(log_spaced(2.06e-9, 4.85e8, 200_000));
    assert!(w.2 > 100_000, "sampled {}", w.2);
    assert!(w.0 <= BOUND_ULPS, "{} ulps at {:e}", w.0, w.1);
}

#[test]
fn bigram_counts_are_within_the_bound() {
    let w =
        worst((0..=200_000_u32).map(|n| {
            f32::from(u16::try_from(n % 65_536).unwrap()) + (n / 65_536) as f32 * 65_536.0
        }));
    assert!(w.0 <= BOUND_ULPS, "{} ulps at {}", w.0, w.1);
}

#[test]
fn arguments_near_minus_one_are_within_the_bound() {
    // -1 + k * 2^-24 for k = 1..2^16, and -0.999.. -0.5
    let near = (1..=65_536_u32).map(|k| -1.0 + f32::from_bits(0x3380_0000) * k as f32);
    let w = worst(near.chain(log_spaced(0.5, 0.999_999, 20_000).map(|x| -x)));
    assert!(w.0 <= BOUND_ULPS, "{} ulps at {:e}", w.0, w.1);
}

#[test]
fn the_edges_are_exact() {
    assert_eq!(ln_1p(0.0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(ln_1p(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(ln_1p(-1.0), f32::NEG_INFINITY);
    assert!(ln_1p(-1.5).is_nan());
    assert!(ln_1p(f32::NEG_INFINITY).is_nan());
    assert_eq!(ln_1p(f32::INFINITY), f32::INFINITY);
    assert!(ln_1p(f32::NAN).is_nan());
    // the largest finite argument does not overflow on the way
    assert_eq!(ulps(ln_1p(f32::MAX), reference(f32::MAX)), 0);
    // the smallest subnormal is its own ln_1p
    let tiny = f32::from_bits(1);
    assert_eq!(ln_1p(tiny).to_bits(), tiny.to_bits());
}
