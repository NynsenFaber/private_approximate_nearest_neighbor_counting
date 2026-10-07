//! [`LsfCounter`], the counting structure behind the four ANNC algorithms.

use super::dp::{DpLsfCounter, TruncatedLaplace};
use crate::anns::LsfIndex;
use crate::lsf::bucket_index::BucketIndex;
use crate::lsf::filters::FilterSet;
use crate::lsf::input::{prepare_points, prepare_query};
use crate::lsf::partition::Partition;
use crate::lsf::probe::{candidate_filters, product_size};
use crate::lsf::{Algorithm, Config, Parameters};
use std::marker::PhantomData;

/// Outcome of an exact (non private) `(alpha, beta)`-ANN counting query.
#[derive(Debug, Clone)]
pub struct CountOutcome {
    /// Number of points stored in the buckets the query selects.
    pub count: u64,
    /// Size `|I(q)|` of the Cartesian product `B_1 x ... x B_t` the query covers.
    pub probed_buckets: usize,
    /// Number of *non-empty* buckets of that product that were actually visited.
    pub matched_buckets: usize,
}

/// An `(alpha, beta)`-ANNC structure (Definition 3) obtained from algorithm `A` by
/// Algorithm 3: every bucket is replaced by its size, so no input point is kept.
///
/// Use it through the aliases [`Top1Counter`](super::Top1Counter),
/// [`CloseTop1Counter`](super::CloseTop1Counter),
/// [`TensorCloseTop1Counter`](super::TensorCloseTop1Counter) and
/// [`TensorTop1Counter`](super::TensorTop1Counter). The methods are the same for
/// all four.
pub struct LsfCounter<A> {
    params: Parameters,
    filters: Vec<FilterSet>,
    pub(crate) counts: BucketIndex<u64>,
    stored_points: usize,
    algorithm: PhantomData<A>,
}

impl<A: Algorithm> LsfCounter<A> {
    /// Builds the counting structure over `data`, one `Vec<f64>` per point. None of
    /// the points is kept.
    ///
    /// All points must have the same dimension, finite coordinates and a non-zero
    /// norm, otherwise an error names the first offending point. Points that are not
    /// unit vectors are normalized (on a copy), with a warning on stderr. See
    /// [`crate::lsf::input`].
    pub fn build(data: &[Vec<f64>], config: &Config) -> Result<Self, String> {
        let data = prepare_points(data)?;
        Partition::build::<A>(&data, config).map(Self::from_partition)
    }

    /// Algorithm 3 `construction`: keeps the filters and replaces each list of
    /// points by its size.
    fn from_partition(partition: Partition) -> Self {
        LsfCounter {
            params: partition.params,
            filters: partition.filters,
            counts: partition.buckets.map(|bucket| bucket.len() as u64),
            stored_points: partition.stored_points,
            algorithm: PhantomData,
        }
    }

    /// Algorithm 3 `query`: the sum of the counters of the buckets the query
    /// selects. Every point is stored at most once, so none is counted twice.
    ///
    /// A query that is not a unit vector is normalized.
    ///
    /// # Panics
    ///
    /// If the query's dimension differs from the points', or it has a non-finite
    /// coordinate, or it is the zero vector.
    pub fn count(&self, query: &[f64]) -> CountOutcome {
        let query = prepare_query(query, self.params.d);
        let levels = candidate_filters(self.filters.iter(), &query);
        let mut count = 0u64;
        let matched_buckets = self.counts.for_each_match(&levels, |_, size| {
            count += size;
            true
        });

        CountOutcome {
            count,
            probed_buckets: product_size(&levels),
            matched_buckets,
        }
    }

    /// Releases the counters under `(epsilon, delta)`-differential privacy
    /// (Theorem 13). `seed` drives the noise.
    ///
    /// Every call spends a full `(epsilon, delta)` budget: drawing several releases
    /// is fine to measure the error over independent noise, but publishing more
    /// than one must be accounted for by composition.
    pub fn release(&self, mechanism: TruncatedLaplace, seed: u64) -> DpLsfCounter<A> {
        let counts = self.counts.iter().map(|(key, &size)| (key.clone(), size));
        DpLsfCounter::release(
            self.params.clone(),
            self.filters.clone(),
            counts,
            mechanism,
            seed,
        )
    }

    /// Same as [`LsfCounter::release`], consuming the exact counters.
    pub fn into_private(self, mechanism: TruncatedLaplace, seed: u64) -> DpLsfCounter<A> {
        let LsfCounter {
            params,
            filters,
            counts,
            ..
        } = self;
        DpLsfCounter::release(params, filters, counts.into_iter_entries(), mechanism, seed)
    }

    /// Resolved parameters.
    pub fn params(&self) -> &Parameters {
        &self.params
    }

    /// Number of input points that ended up in a bucket.
    pub fn stored_points(&self) -> usize {
        self.stored_points
    }

    /// Number of non-empty buckets, i.e. of counters.
    pub fn occupied_buckets(&self) -> usize {
        self.counts.len()
    }
}

/// Algorithm 3 applied to a built search index: the points are dropped and each
/// bucket keeps only its size.
impl<A: Algorithm> From<LsfIndex<A>> for LsfCounter<A> {
    fn from(index: LsfIndex<A>) -> Self {
        Self::from_partition(index.into_partition())
    }
}

/// Properties every ANNC algorithm must have, checked by each algorithm's tests.
#[cfg(test)]
pub(crate) fn assert_count_contract<A: Algorithm>() {
    use crate::data::exact_count;
    use crate::lsf::probe::ProbeIter;
    use crate::test_support::{config, planted_dataset, scaled_by_4};
    use crate::utils::{dot_product, normalize_vector};

    let (data, query) = planted_dataset(600, 20, 0.99, 15);
    let config = config::<A>(16);
    let index = LsfIndex::<A>::build(data.clone(), &config).unwrap();

    // Split the points in the selected buckets into the ones the problem allows to
    // count (inner product >= beta) and the far points that inflate the answer.
    let levels = index.candidate_filters(&query);
    let mut within_beta = 0u64;
    let mut far_points = 0u64;
    for key in ProbeIter::new(&levels) {
        for &point_id in index.partition.buckets.get(&key).into_iter().flatten() {
            if dot_product(&query, index.point(point_id as usize)) >= config.beta {
                within_beta += 1;
            } else {
                far_points += 1;
            }
        }
    }
    assert!(within_beta <= exact_count(&data, &query, config.beta));

    let stored_points = index.stored_points();
    let from_index = LsfCounter::from(index);
    let counted = from_index.count(&query);
    // The count is exactly the size of the selected buckets: near plus far points.
    assert_eq!(counted.count, within_beta + far_points);
    assert!(counted.count > 0, "the planted cluster was not counted");
    assert!(counted.count <= stored_points as u64);
    assert_eq!(counted.probed_buckets, product_size(&levels));
    assert_eq!(from_index.stored_points(), stored_points);

    // Building the counter directly gives the same structure.
    let direct = LsfCounter::<A>::build(&data, &config).unwrap();
    assert_eq!(direct.count(&query).count, counted.count);
    assert_eq!(direct.occupied_buckets(), from_index.occupied_buckets());

    // Points and queries off the sphere are normalized (on a copy): scaling them
    // by 4 gives the counter of the normalized data.
    let mut normalized = data.clone();
    normalized
        .iter_mut()
        .for_each(|point| normalize_vector(point));
    let reference = LsfCounter::<A>::build(&normalized, &config).unwrap();
    let scaled = LsfCounter::<A>::build(&scaled_by_4(&data), &config).unwrap();
    let scaled_query: Vec<f64> = query.iter().map(|x| 4.0 * x).collect();
    assert_eq!(
        scaled.count(&scaled_query).count,
        reference.count(&query).count
    );

    // Malformed data sets are errors naming the offending point.
    let mut with_nan = data.clone();
    with_nan[7][0] = f64::NAN;
    let error = LsfCounter::<A>::build(&with_nan, &config).err().unwrap();
    assert!(error.contains("point 7 has the non-finite"), "{error}");
    let mut with_zero = data;
    with_zero[3].iter_mut().for_each(|x| *x = 0.0);
    let error = LsfCounter::<A>::build(&with_zero, &config).err().unwrap();
    assert!(error.contains("point 3 is the zero vector"), "{error}");
}
