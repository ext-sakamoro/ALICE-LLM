//! The one place this crate reads how much wall time has passed.
//!
//! Every timing the crate reports (per-layer and prefill / decode milliseconds, kernel
//! microseconds, the server's request latency) is a diagnostic: it is printed or put in a
//! statistics field and never feeds back into a generated token, a logit or any other
//! computed value. `Instant::elapsed` is banned in `clippy.toml` so that a wall-clock read
//! cannot slip into the computation unseen; this function is the single allowed read.

use std::time::{Duration, Instant};

/// The wall time since `start`, for diagnostics only (see the module documentation).
#[must_use]
#[inline]
pub fn elapsed(start: Instant) -> Duration {
    #[allow(
        clippy::disallowed_methods,
        reason = "the crate's single wall-clock read for diagnostic timings, never part of a computed value"
    )]
    start.elapsed()
}
