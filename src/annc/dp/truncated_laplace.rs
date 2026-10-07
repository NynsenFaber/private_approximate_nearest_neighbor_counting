//! Truncated Laplace mechanism (Geng, Ding, Guo, Kumar, AISTATS 2020).
//!
//! This is the `make_private` mechanism of Algorithm 1 / Theorem 13 of the paper.
//! It adds noise drawn from a Laplace distribution *truncated* to `[-A, A]` with
//!
//! ```text
//! b = sensitivity / epsilon,     A = b * ln(1 + (e^epsilon - 1) / (2 delta)),
//! ```
//!
//! which is the optimal (minimum noise) `(epsilon, delta)`-DP additive mechanism for a
//! query of the given sensitivity. Because the noise is *bounded*, every released
//! counter is off by at most `A = O(log(1/delta) / epsilon)`, which is exactly the
//! per-counter error assumed by Theorem 13 of the paper.

use rand::Rng;
use std::collections::HashMap;
use std::hash::Hash;

/// A calibrated truncated Laplace mechanism.
#[derive(Debug, Clone, Copy)]
pub struct TruncatedLaplace {
    /// Privacy budget `epsilon`.
    pub epsilon: f64,
    /// Failure probability `delta`; the mechanism requires `delta > 0`.
    pub delta: f64,
    /// `L1` sensitivity of the histogram query (1 for add/remove, 2 for substitution).
    pub sensitivity: f64,
    /// Scale `b = sensitivity / epsilon` of the underlying Laplace distribution.
    pub scale: f64,
    /// Support bound `A`: samples always lie in `[-A, A]`.
    pub bound: f64,
}

impl TruncatedLaplace {
    /// Calibrates the mechanism to `(epsilon, delta)`-DP for the given sensitivity.
    pub fn new(epsilon: f64, delta: f64, sensitivity: f64) -> Result<Self, String> {
        // NaN fails every comparison, so testing the positive form also rejects it.
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(format!(
                "epsilon must be finite and positive, got {epsilon}"
            ));
        }
        if !(delta > 0.0 && delta < 1.0) {
            return Err(format!(
                "delta must lie in (0, 1) - the truncated Laplace mechanism cannot give \
                 pure DP - got {delta}"
            ));
        }
        if !(sensitivity.is_finite() && sensitivity > 0.0) {
            return Err(format!(
                "sensitivity must be finite and positive, got {sensitivity}"
            ));
        }
        let scale = sensitivity / epsilon;
        let bound = scale * (1.0 + (epsilon.exp() - 1.0) / (2.0 * delta)).ln();
        if !bound.is_finite() {
            return Err(format!(
                "noise bound overflowed for epsilon = {epsilon}, delta = {delta}"
            ));
        }
        Ok(TruncatedLaplace {
            epsilon,
            delta,
            sensitivity,
            scale,
            bound,
        })
    }

    /// Draws one noise sample in `[-A, A]` by inverting the CDF.
    ///
    /// With `c = 1 - e^{-A/b}` the CDF of the truncated Laplace is
    /// `F(x) = (2c)^{-1}(e^{x/b} - (1 - c))` for `x <= 0` and symmetric above,
    /// so `F^{-1}(u) = b ln(2cu + 1 - c)` for `u <= 1/2`.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = rng.gen();
        let c = 1.0 - (-self.bound / self.scale).exp();
        let noise = if u <= 0.5 {
            let arg = (2.0 * c * u + 1.0 - c).max(f64::MIN_POSITIVE);
            self.scale * arg.ln()
        } else {
            let arg = (1.0 - c * (2.0 * u - 1.0)).max(f64::MIN_POSITIVE);
            -self.scale * arg.ln()
        };
        // Guards against round-off at the very ends of the support.
        noise.clamp(-self.bound, self.bound)
    }

    /// Counters whose noisy value does not exceed this threshold are suppressed.
    ///
    /// Suppressing at `1 + A` is what makes the *sparse* release private. A counter
    /// that is `0` on one dataset and `1` on a neighbouring one would be noised to
    /// at most `1 + A`, hence suppressed in both cases: the set of released keys
    /// therefore carries no information about a single individual, and the released
    /// values are `(epsilon, delta)`-DP by the guarantee of the mechanism followed
    /// by post-processing.
    pub fn suppression_threshold(&self) -> f64 {
        1.0 + self.bound
    }

    /// Expected absolute noise `E|Z|` added to a released counter.
    pub fn mean_absolute_noise(&self) -> f64 {
        let ratio = self.bound / self.scale;
        let exp_neg = (-ratio).exp();
        self.scale * (1.0 - exp_neg * (1.0 + ratio)) / (1.0 - exp_neg)
    }
}

/// Releases a sparse histogram under `(epsilon, delta)`-differential privacy.
///
/// Only non-empty buckets are noised and stored; buckets whose noisy value falls
/// below [`TruncatedLaplace::suppression_threshold`] are dropped, which makes the
/// released *key set* independent of any single data point (see the discussion on
/// [`TruncatedLaplace::suppression_threshold`]). Returns the released map together
/// with the number of suppressed buckets.
pub fn privatize_histogram<K, I, R>(
    mechanism: &TruncatedLaplace,
    counts: I,
    rng: &mut R,
) -> (HashMap<K, f64>, usize)
where
    K: Eq + Hash,
    I: IntoIterator<Item = (K, u64)>,
    R: Rng + ?Sized,
{
    let threshold = mechanism.suppression_threshold();
    let mut released = HashMap::new();
    let mut suppressed = 0usize;
    for (key, count) in counts {
        let noisy = count as f64 + mechanism.sample(rng);
        if noisy > threshold {
            released.insert(key, noisy);
        } else {
            suppressed += 1;
        }
    }
    (released, suppressed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// The support bound must match the closed form of Geng et al.
    #[test]
    fn test_bound_formula() {
        let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
        let expected = (1.0f64 + (1.0f64.exp() - 1.0) / 2e-6).ln();
        assert!((mechanism.bound - expected).abs() < 1e-9);
        // Larger epsilon or larger delta means less noise.
        let looser = TruncatedLaplace::new(2.0, 1e-6, 1.0).unwrap();
        assert!(looser.bound < mechanism.bound);
        let larger_delta = TruncatedLaplace::new(1.0, 1e-3, 1.0).unwrap();
        assert!(larger_delta.bound < mechanism.bound);
    }

    /// Pure DP is not achievable with this mechanism, and inputs are validated.
    #[test]
    fn test_invalid_parameters() {
        assert!(TruncatedLaplace::new(1.0, 0.0, 1.0).is_err());
        assert!(TruncatedLaplace::new(0.0, 1e-6, 1.0).is_err());
        assert!(TruncatedLaplace::new(-1.0, 1e-6, 1.0).is_err());
        assert!(TruncatedLaplace::new(1.0, 1e-6, 0.0).is_err());
        // e^1000 overflows, so the support bound would be infinite.
        assert!(TruncatedLaplace::new(1000.0, 1e-6, 1.0).is_err());
    }

    /// Samples stay inside the support and are centred, with roughly Laplace spread.
    #[test]
    fn test_sample_distribution() {
        let mechanism = TruncatedLaplace::new(0.5, 1e-6, 1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<f64> = (0..200_000).map(|_| mechanism.sample(&mut rng)).collect();
        for sample in &samples {
            assert!(sample.abs() <= mechanism.bound + 1e-12);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.05, "sample mean {mean} is not centred");
        let mean_abs = samples.iter().map(|s| s.abs()).sum::<f64>() / samples.len() as f64;
        assert!(
            (mean_abs - mechanism.mean_absolute_noise()).abs() < 0.05,
            "E|Z| = {mean_abs}, expected {}",
            mechanism.mean_absolute_noise()
        );
    }

    /// A bucket holding a single point can never be released: this is what protects
    /// the key set of the sparse histogram.
    #[test]
    fn test_singleton_buckets_are_always_suppressed() {
        let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let counts: Vec<(u32, u64)> = (0..1000).map(|i| (i, 1)).collect();
        let (released, suppressed) = privatize_histogram(&mechanism, counts, &mut rng);
        assert!(released.is_empty());
        assert_eq!(suppressed, 1000);
    }

    /// Counters well above the threshold survive and stay within `A` of the truth.
    #[test]
    fn test_large_buckets_are_released_with_bounded_error() {
        let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(11);
        let truth = 10_000u64;
        let counts: Vec<(u32, u64)> = (0..100).map(|i| (i, truth)).collect();
        let (released, suppressed) = privatize_histogram(&mechanism, counts, &mut rng);
        assert_eq!(suppressed, 0);
        assert_eq!(released.len(), 100);
        for value in released.values() {
            assert!((value - truth as f64).abs() <= mechanism.bound);
        }
    }
}
