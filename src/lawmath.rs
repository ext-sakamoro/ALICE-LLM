//! The transcendental functions and max / min the inference code computes
//! with (ALICE-Law G1 / G2)
//!
//! Every call goes through this module. With the default features each
//! function is `alice-det-math`'s, which returns the same bits on every
//! target, so the logits, the sampled tokens and the KV cache contents are
//! the same on every machine for the same model and input.
//!
//! The `platform-math` feature (off by default) replaces them with the
//! standard library's, which calls the platform libm: faster on some targets,
//! but the bits then depend on the OS, CPU and compiler. A build with it is
//! not Law-bearing: it reports another semantics identifier
//! ([`crate::semantics`]) and the KV cache reader refuses files written by
//! the other mode.
//!
//! `ln_1p` has no `alice-det-math` counterpart in 0.5; [`ln_1p`] computes it
//! here from `alice-det-math`'s `ln` with Kahan's form, which keeps the
//! relative accuracy of `ln(1 + x)` for small `x` that the plain form loses.

#[cfg(not(feature = "platform-math"))]
mod imp {
    #[inline]
    pub fn exp(x: f32) -> f32 {
        alice_det_math::exp(x)
    }
    #[inline]
    pub fn ln(x: f32) -> f32 {
        alice_det_math::ln(x)
    }
    #[inline]
    pub fn tanh(x: f32) -> f32 {
        alice_det_math::tanh(x)
    }
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        alice_det_math::powf(x, y)
    }
    #[inline]
    pub fn sin_cos(x: f32) -> (f32, f32) {
        alice_det_math::sin_cos(x)
    }
    #[inline]
    pub fn sin(x: f32) -> f32 {
        alice_det_math::sin(x)
    }
    #[inline]
    pub fn cos(x: f32) -> f32 {
        alice_det_math::cos(x)
    }
    #[inline]
    pub fn powi(x: f32, n: i32) -> f32 {
        alice_det_math::powi(x, n)
    }
}

#[cfg(feature = "platform-math")]
#[expect(
    clippy::disallowed_methods,
    reason = "platform-math is the non-Law build: it calls the platform libm by design"
)]
mod imp {
    #[inline]
    pub fn exp(x: f32) -> f32 {
        x.exp()
    }
    #[inline]
    pub fn ln(x: f32) -> f32 {
        x.ln()
    }
    #[inline]
    pub fn tanh(x: f32) -> f32 {
        x.tanh()
    }
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        x.powf(y)
    }
    #[inline]
    pub fn sin_cos(x: f32) -> (f32, f32) {
        x.sin_cos()
    }
    #[inline]
    pub fn sin(x: f32) -> f32 {
        x.sin()
    }
    #[inline]
    pub fn cos(x: f32) -> f32 {
        x.cos()
    }
    #[inline]
    pub fn powi(x: f32, n: i32) -> f32 {
        x.powi(n)
    }
}

/// `e^x`
#[inline]
#[must_use]
pub fn exp(x: f32) -> f32 {
    imp::exp(x)
}

/// Natural logarithm (`ln(0) = -inf`, `ln(x < 0) = NaN`)
#[inline]
#[must_use]
pub fn ln(x: f32) -> f32 {
    imp::ln(x)
}

/// Hyperbolic tangent
#[inline]
#[must_use]
pub fn tanh(x: f32) -> f32 {
    imp::tanh(x)
}

/// `x^y`
#[inline]
#[must_use]
pub fn powf(x: f32, y: f32) -> f32 {
    imp::powf(x, y)
}

/// `(sin x, cos x)`
#[inline]
#[must_use]
pub fn sin_cos(x: f32) -> (f32, f32) {
    imp::sin_cos(x)
}

/// `sin x`
#[inline]
#[must_use]
pub fn sin(x: f32) -> f32 {
    imp::sin(x)
}

/// `cos x`
#[inline]
#[must_use]
pub fn cos(x: f32) -> f32 {
    imp::cos(x)
}

/// `x^n` by repeated squaring
///
/// The multiplication order is fixed by `alice-det-math`'s source
/// (`f32::powi` lowers to a compiler-rt routine of unspecified association
/// order); `powi(2.0, n)` is exact wherever `2^n` is representable.
#[inline]
#[must_use]
pub fn powi(x: f32, n: i32) -> f32 {
    imp::powi(x, n)
}

/// `ln(1 + x)`, accurate for small `|x|`
///
/// Kahan's form: with `u = 1 + x` rounded, `u == 1` means `x` is below half
/// an ulp of 1 and `ln(1 + x) = x` to within the rounding of `x` itself;
/// otherwise `ln(u) * (x / (u - 1))` corrects the rounding of `u`, because
/// `u - 1` is exact and `x / (u - 1)` is the factor by which `u - 1` missed
/// `x`. The multiplication is done after the division so that a large `x`
/// does not overflow. Uses only `+`, `-`, `*`, `/` and [`ln`].
///
/// `ln_1p(-1) = -inf`, `ln_1p(x < -1) = NaN`, `ln_1p(inf) = inf`,
/// `ln_1p(±0) = ±0`, NaN gives NaN. Measured against `f64::ln_1p` rounded to
/// `f32` (tests/lawmath_ln_1p.rs): at most 2 ulps over `|x| < 1e-6`, at most
/// 1 ulp over the softplus range `[2.06e-9, 4.85e8]`, the bigram counts and
/// `x` near -1.
#[inline]
#[must_use]
pub fn ln_1p(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }
    let u = 1.0 + x;
    if u == 1.0 {
        return x;
    }
    if u == 0.0 {
        return f32::NEG_INFINITY;
    }
    ln(u) * (x / (u - 1.0))
}

/// `max` with a defined zero sign (`+0` for an equal pair of zeros) and a
/// NaN operand giving the other operand
#[inline]
#[must_use]
pub fn max(a: f32, b: f32) -> f32 {
    alice_det_math::max(a, b)
}

/// `min` with a defined zero sign (`-0` for an equal pair of zeros) and a
/// NaN operand giving the other operand
#[inline]
#[must_use]
pub fn min(a: f32, b: f32) -> f32 {
    alice_det_math::min(a, b)
}

/// `x` restricted to `[low, high]`: a NaN in any argument gives the
/// canonical NaN (`f32::clamp` passes a NaN `x` through with its payload),
/// and with `low > high` the low bound wins (`f32::clamp` panics)
#[inline]
#[must_use]
pub fn clamp(x: f32, low: f32, high: f32) -> f32 {
    alice_det_math::metric::clamp(x, low, high)
}

/// [`max`] for `f64`
#[inline]
#[must_use]
pub fn max64(a: f64, b: f64) -> f64 {
    alice_det_math::max64(a, b)
}

/// [`min`] for `f64`
#[inline]
#[must_use]
pub fn min64(a: f64, b: f64) -> f64 {
    alice_det_math::min64(a, b)
}
