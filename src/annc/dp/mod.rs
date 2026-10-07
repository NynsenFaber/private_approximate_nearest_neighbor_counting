//! Differentially private ANNC (DP-ANNC) with the truncated Laplace mechanism.
//!
//! Theorem 13: privatize the counters of an ANNC structure from [`crate::annc`],
//! and publish them together with the filters. Each of the four algorithms gives
//! one private structure: [`DpTop1`] (Algorithm 1, "DPTop-1"), [`DpCloseTop1`],
//! [`DpTensorCloseTop1`] and [`DpTensorTop1`].
//!
//! # Privacy argument (Section 4.2)
//!
//! * The partition function `Q` — the Gaussian filters and the threshold — is drawn
//!   independently of the data, so it can be published as is.
//! * Every point is stored in at most one bucket, so two neighbouring data sets
//!   (add/remove one point) produce histograms that differ by one in a single
//!   counter: the sensitivity is `1` (`2` under substitution).
//! * The counters are released with the [`TruncatedLaplace`] mechanism, so the
//!   release is `(epsilon, delta)`-DP and every counter is off by at most
//!   `A = O(log(1/delta) / epsilon)`.
//!
//! A query sums the released counters of the buckets it selects, so its additive
//! error is at most `A` per summed counter. Theorem 13 states this as `A * K` with
//! `K = E[|I(q)|]`; with the balanced `theta = rho` this is
//! `O(log(1/delta) / epsilon * n^{rho + o(1)})` (Theorem 1).
//!
//! # Two deviations from the paper, both deliberate
//!
//! **Sparse release.** Theorem 13 noises all `m` counters. For the tensorized
//! algorithms `m = m_sub^t` is `5.3e9` at `n = 10^6`, so only the non-empty buckets
//! are noised and stored, and a bucket is dropped unless its noisy value exceeds
//! `1 + A`. That threshold keeps the released *key set* private, and it is sound
//! because the truncated Laplace noise is *bounded*: a bucket holding one point is
//! noised to at most `1 + A`, so it is suppressed with certainty, whether or not
//! that point is in the data set. Everything else is post-processing of an
//! `(epsilon, delta)`-DP value. See [`TruncatedLaplace::suppression_threshold`].
//!
//! **Budget range.** Theorem 13 is stated for `epsilon <= 1`, which is what its
//! `O(log(1/delta) / epsilon)` error form assumes. The mechanism itself is
//! `(epsilon, delta)`-DP at any `epsilon > 0` (Geng et al.), so larger budgets are
//! accepted.
//!
//! # Caveats for real deployments
//!
//! * The noise comes from a `StdRng` seeded by the caller, so that experiments are
//!   reproducible. A real release must seed from OS entropy and keep the seed secret.
//! * Sampling uses `f64` arithmetic and is exposed to the floating-point attacks of
//!   Mironov (CCS 2012).
//! * Every [`LsfCounter::release`](crate::annc::LsfCounter::release) spends a full
//!   `(epsilon, delta)` budget; publishing several releases requires composition.

mod close_top1;
mod private_counter;
mod tensor_close_top1;
mod tensor_top1;
mod top1;
pub mod truncated_laplace;

pub use close_top1::DpCloseTop1;
pub use private_counter::{DpCountOutcome, DpLsfCounter};
pub use tensor_close_top1::DpTensorCloseTop1;
pub use tensor_top1::DpTensorTop1;
pub use top1::DpTop1;
pub use truncated_laplace::TruncatedLaplace;
