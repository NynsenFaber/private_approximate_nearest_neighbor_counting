//! Differentially private approximate near neighbour counting (DP-ANNC).
//!
//! This is the composition of Algorithm 3 (turn a space partitioning ANN structure
//! into a counting structure by replacing each list with its size) and Theorem 13
//! (privatize the resulting histogram with the truncated Laplace mechanism).
//!
//! Privacy argument, following Section 4.2 of the paper:
//!
//! * the partition `Q` — the Gaussian filters and the thresholds — is drawn
//!   independently of the data, so it can be published as is;
//! * every point is stored in exactly one bucket, hence two neighbouring data sets
//!   (add/remove one point) produce histograms that differ by one in a single
//!   counter: the sensitivity is `1`;
//! * the counters are released with the truncated Laplace mechanism, so the release
//!   is `(epsilon, delta)`-DP and every counter is off by at most
//!   `A = O(log(1/delta)/epsilon)`.
//!
//! A query sums the released counters over the inspected buckets, so its additive
//! error is at most `A * |I(q)|`, matching Theorem 13 with `K = E[|I(q)|]`.
//!
//! # Two deviations from the paper, both deliberate
//!
//! **Sparse release.** Theorem 13 noises all `m` counters. Here `m = m_sub^t` is
//! `5.3e9` at `n = 10^6`, so only the non-empty buckets are noised and stored, and a
//! bucket is dropped unless its noisy value exceeds `1 + A`. That threshold is what
//! keeps the released *key set* private, and it is sound precisely because the
//! truncated Laplace noise is *bounded*: a bucket holding one point is noised to at
//! most `1 + A`, so it is suppressed with certainty, whether or not that point is
//! in the data set. Everything else is post-processing of an `(epsilon, delta)`-DP
//! value. See [`crate::dp::truncated_laplace::TruncatedLaplace::suppression_threshold`].
//!
//! **Budget range.** Theorem 13 is stated for `epsilon <= 1`, which is what its
//! `O(log(1/delta)/epsilon)` error form assumes. The mechanism itself is
//! `(epsilon, delta)`-DP at any `epsilon > 0` (Geng et al.), so larger budgets are
//! accepted and reported; the experiments sweep up to `epsilon = 8` to show where
//! the private answer meets the non-private accuracy floor of the same partition.

use super::bucket_index::BucketIndex;
use super::close_top1::FilterSet;
use super::probe::{candidate_filters, product_size, BucketKey};
use super::tensor_close_top1::Parameters;
use crate::dp::truncated_laplace::{privatize_histogram, TruncatedLaplace};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Outcome of a private counting query.
#[derive(Debug, Clone)]
pub struct DpCountOutcome {
    /// Differentially private estimate of `|S ∩ B(q, alpha)|`.
    pub estimate: f64,
    /// Size `|I(q)|` of the Cartesian product `B_1 x ... x B_t` the query covers.
    pub probed_buckets: usize,
    /// Number of released counters the estimate actually summed. Empty and
    /// suppressed buckets contribute nothing, so this is the number of noise terms.
    pub matched_buckets: usize,
}

/// A published DP-ANNC structure: filters plus noisy counters, no input points.
pub struct DpAnnc {
    /// Resolved parameters of the underlying partition.
    pub params: Parameters,
    /// The privacy mechanism used to release the counters.
    pub mechanism: TruncatedLaplace,
    /// Number of counters that were suppressed because their noisy value fell
    /// below the suppression threshold.
    pub suppressed_buckets: usize,
    filters: Vec<FilterSet>,
    noisy_counts: BucketIndex<f64>,
}

impl DpAnnc {
    /// Releases a bucket histogram under `(epsilon, delta)`-DP.
    ///
    /// Called by [`super::tensor_close_top1::TensorCloseTop1::release`].
    pub fn release<I>(
        params: Parameters,
        filters: Vec<FilterSet>,
        counts: I,
        mechanism: TruncatedLaplace,
        seed: u64,
    ) -> Self
    where
        I: IntoIterator<Item = (BucketKey, u64)>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let (noisy_counts, suppressed_buckets) = privatize_histogram(&mechanism, counts, &mut rng);
        DpAnnc {
            params,
            mechanism,
            suppressed_buckets,
            filters,
            noisy_counts: BucketIndex::from_map(noisy_counts),
        }
    }

    /// Number of released (non suppressed) counters.
    pub fn released_buckets(&self) -> usize {
        self.noisy_counts.len()
    }

    /// Private estimate of the number of points at inner product `>= alpha`.
    ///
    /// With probability at least `2/3` the answer lies between
    /// `(1 - o(1)) |S ∩ B(q, alpha)| - O(A K)` and `|S ∩ B(q, beta)| + O(A K)`.
    pub fn query(&self, query: &[f64]) -> DpCountOutcome {
        let levels = candidate_filters(self.filters.iter(), query);
        let mut estimate = 0.0f64;
        let matched_buckets = self.noisy_counts.for_each_match(&levels, |_, noisy| {
            estimate += noisy;
            true
        });

        DpCountOutcome {
            estimate,
            probed_buckets: product_size(&levels),
            matched_buckets,
        }
    }

    /// Worst case additive error of a query, `A` times the number of counters it
    /// summed. This is the `A * |I(q)|` term of Theorem 13, tightened by the fact
    /// that only released counters carry noise in the sparse implementation.
    pub fn error_bound(&self, counters: usize) -> f64 {
        self.mechanism.bound * counters as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{plant_neighbours, PlantConfig};
    use crate::tensor_data_structures::tensor_close_top1::{Config, TensorCloseTop1};
    use crate::utils::{generate_unit_sphere_vectors, random_unit_vector};
    use rand::SeedableRng;

    fn structure(seed: u64) -> (TensorCloseTop1, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let query = random_unit_vector(16, &mut rng);
        let mut data = generate_unit_sphere_vectors(400, 16, seed);
        data.extend(plant_neighbours(
            &query,
            &PlantConfig {
                count: 300,
                similarity: 0.95,
                tightness: 0.9999,
            },
            &mut rng,
        ));
        let config = Config {
            alpha: 0.9,
            beta: 0.5,
            t: Some(3),
            m_sub: Some(64),
            seed,
            ..Config::default()
        };
        (TensorCloseTop1::build(data, &config).unwrap(), query)
    }

    /// The private estimate must stay within the additive error bound of the exact
    /// answer of the same partition.
    #[test]
    fn test_estimate_within_error_bound() {
        let (structure, query) = structure(101);
        let exact = structure.count(&query);
        let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
        let private = structure.into_private(mechanism, 202);
        let noisy = private.query(&query);

        assert_eq!(noisy.probed_buckets, exact.probed_buckets);
        // Suppressed buckets lose their whole (small) count, released ones are off
        // by at most A: both are covered by A per non-empty bucket of the product.
        let error = (noisy.estimate - exact.count as f64).abs();
        let bound = private.error_bound(exact.matched_buckets)
            + private.mechanism.suppression_threshold() * exact.matched_buckets as f64;
        assert!(
            error <= bound + 1e-9,
            "error {error} exceeds the bound {bound}"
        );
    }

    /// A larger privacy budget means less noise, hence a smaller error bound and,
    /// in expectation, a more accurate answer.
    #[test]
    fn test_more_budget_means_less_noise() {
        let tight = TruncatedLaplace::new(0.1, 1e-6, 1.0).unwrap();
        let loose = TruncatedLaplace::new(4.0, 1e-6, 1.0).unwrap();
        assert!(loose.bound < tight.bound);

        let (structure, query) = structure(103);
        let counted = structure.count(&query);
        let exact = counted.count as f64;
        let private = structure.into_private(loose, 204);
        let outcome = private.query(&query);
        let bound = private.error_bound(counted.matched_buckets)
            + private.mechanism.suppression_threshold() * counted.matched_buckets as f64;
        assert!((outcome.estimate - exact).abs() <= bound);
    }

    /// Releasing the structure drops the input points: only filters and noisy
    /// counters survive, and suppressed buckets are not published.
    #[test]
    fn test_release_is_sparse() {
        let (structure, _query) = structure(105);
        let occupied = structure.occupied_buckets();
        let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
        let private = structure.into_private(mechanism, 206);
        assert_eq!(
            private.released_buckets() + private.suppressed_buckets,
            occupied
        );
        // Every released counter is above the suppression threshold.
        for (_, value) in private.noisy_counts.iter() {
            assert!(*value > private.mechanism.suppression_threshold());
        }
    }
}
