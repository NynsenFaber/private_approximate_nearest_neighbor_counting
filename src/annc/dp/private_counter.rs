//! [`DpLsfCounter`], the published structure behind the four DP-ANNC algorithms.

use super::truncated_laplace::{privatize_histogram, TruncatedLaplace};
use crate::lsf::bucket_index::BucketIndex;
use crate::lsf::filters::FilterSet;
use crate::lsf::input::prepare_query;
use crate::lsf::probe::{candidate_filters, product_size, BucketKey};
use crate::lsf::{Algorithm, Parameters};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::marker::PhantomData;

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

/// A published `(epsilon, delta)`-DP ANNC structure built by algorithm `A`:
/// filters plus noisy counters, no input points.
///
/// Obtained with [`LsfCounter::release`](crate::annc::LsfCounter::release) or
/// [`LsfCounter::into_private`](crate::annc::LsfCounter::into_private). Use it
/// through the aliases [`DpTop1`](super::DpTop1), [`DpCloseTop1`](super::DpCloseTop1),
/// [`DpTensorCloseTop1`](super::DpTensorCloseTop1) and
/// [`DpTensorTop1`](super::DpTensorTop1).
pub struct DpLsfCounter<A> {
    params: Parameters,
    mechanism: TruncatedLaplace,
    suppressed_buckets: usize,
    filters: Vec<FilterSet>,
    pub(crate) noisy_counts: BucketIndex<f64>,
    algorithm: PhantomData<A>,
}

impl<A: Algorithm> DpLsfCounter<A> {
    /// Releases a bucket histogram under `(epsilon, delta)`-DP: noises every
    /// non-empty counter and suppresses the ones at or below `1 + A`.
    pub(crate) fn release<I>(
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
        DpLsfCounter {
            params,
            mechanism,
            suppressed_buckets,
            filters,
            noisy_counts: BucketIndex::from_map(noisy_counts),
            algorithm: PhantomData,
        }
    }

    /// Private estimate of the number of points at inner product `>= alpha`: the
    /// sum of the released counters of the buckets the query selects.
    ///
    /// With probability at least `2/3` the answer lies between
    /// `(1 - o(1)) |S ∩ B(q, alpha)| - O(A K)` and `|S ∩ B(q, beta)| + O(A K)`
    /// (Theorem 13).
    ///
    /// A query that is not a unit vector is normalized.
    ///
    /// # Panics
    ///
    /// If the query's dimension differs from the points', or it has a non-finite
    /// coordinate, or it is the zero vector.
    pub fn query(&self, query: &[f64]) -> DpCountOutcome {
        let query = prepare_query(query, self.params.d);
        let levels = candidate_filters(self.filters.iter(), &query);
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

    /// Worst case additive noise of a query that summed `counters` released
    /// counters: `A` times that number. This is the `A * |I(q)|` term of
    /// Theorem 13, tightened by the fact that only released counters carry noise.
    pub fn error_bound(&self, counters: usize) -> f64 {
        self.mechanism.bound * counters as f64
    }

    /// Resolved parameters of the underlying partition.
    pub fn params(&self) -> &Parameters {
        &self.params
    }

    /// The privacy mechanism used to release the counters.
    pub fn mechanism(&self) -> &TruncatedLaplace {
        &self.mechanism
    }

    /// Number of released (non suppressed) counters.
    pub fn released_buckets(&self) -> usize {
        self.noisy_counts.len()
    }

    /// Number of non-empty buckets whose noisy value fell at or below the
    /// suppression threshold and were therefore not released.
    pub fn suppressed_buckets(&self) -> usize {
        self.suppressed_buckets
    }
}

/// Properties every DP-ANNC algorithm must have, checked by each algorithm's tests.
#[cfg(test)]
pub(crate) fn assert_dp_contract<A: Algorithm>() {
    use crate::annc::LsfCounter;
    use crate::test_support::{config, planted_dataset};

    let (data, query) = planted_dataset(400, 300, 0.9999, 101);
    let counter = LsfCounter::<A>::build(&data, &config::<A>(101)).unwrap();
    let exact = counter.count(&query);
    let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
    let private = counter.release(mechanism, 202);
    let noisy = private.query(&query);
    assert_eq!(noisy.probed_buckets, exact.probed_buckets);

    // Suppressed buckets lose their whole (small) count, released ones are off by
    // at most A: both are covered by 1 + 2A per non-empty bucket of the product.
    let bound = private.error_bound(exact.matched_buckets)
        + mechanism.suppression_threshold() * exact.matched_buckets as f64;
    let error = (noisy.estimate - exact.count as f64).abs();
    assert!(
        error <= bound + 1e-9,
        "error {error} exceeds the bound {bound}"
    );
    assert!(noisy.estimate > 0.0, "the planted cluster was suppressed");

    // Sparse release: every non-empty bucket is either released above the
    // threshold or suppressed, and nothing else is published.
    assert_eq!(
        private.released_buckets() + private.suppressed_buckets(),
        counter.occupied_buckets()
    );
    for (_, value) in private.noisy_counts.iter() {
        assert!(*value > mechanism.suppression_threshold());
    }

    // The noise is reproducible from the seed, whether or not the counter is consumed.
    let again = counter.into_private(mechanism, 202).query(&query);
    assert_eq!(again.estimate, noisy.estimate);
}
