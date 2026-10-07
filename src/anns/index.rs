//! [`LsfIndex`], the search structure behind the four ANNS algorithms.

use crate::lsf::filters::FilterSet;
use crate::lsf::input::{check_points, normalize_points, prepare_query};
use crate::lsf::partition::Partition;
use crate::lsf::probe::product_size;
use crate::lsf::{Algorithm, Config, Parameters};
use crate::utils::dot_product;
use std::marker::PhantomData;

/// Outcome of an `(alpha, beta)`-ANN query.
#[derive(Debug, Clone)]
pub struct AnnOutcome {
    /// Index of a point at inner product `>= beta` from the query, if one was found.
    pub point: Option<usize>,
    /// Size `|I(q)|` of the Cartesian product `B_1 x ... x B_t` the query covers.
    pub probed_buckets: usize,
    /// Number of *non-empty* buckets of that product that were actually visited.
    pub matched_buckets: usize,
    /// Number of points whose inner product with the query was evaluated.
    pub inspected_points: usize,
}

/// An `(alpha, beta)`-ANN index (Definition 2) built by algorithm `A`.
///
/// Use it through the aliases [`Top1`](super::Top1), [`CloseTop1`](super::CloseTop1),
/// [`TensorCloseTop1`](super::TensorCloseTop1) and [`TensorTop1`](super::TensorTop1),
/// whose documentation describes each algorithm. The methods are the same for all
/// four.
pub struct LsfIndex<A> {
    pub(crate) partition: Partition,
    data: Vec<Vec<f64>>,
    algorithm: PhantomData<A>,
}

impl<A: Algorithm> LsfIndex<A> {
    /// Builds the index over `data`, one `Vec<f64>` per point.
    ///
    /// All points must have the same dimension, finite coordinates and a non-zero
    /// norm, otherwise an error names the first offending point. Points that are not
    /// unit vectors are normalized in place, with a warning on stderr. See
    /// [`crate::lsf::input`].
    pub fn build(mut data: Vec<Vec<f64>>, config: &Config) -> Result<Self, String> {
        let not_unit = check_points(&data)?;
        normalize_points(&mut data, &not_unit);
        let partition = Partition::build::<A>(&data, config)?;
        Ok(LsfIndex {
            partition,
            data,
            algorithm: PhantomData,
        })
    }

    /// Solves `(alpha, beta)`-ANN: returns a point at inner product `>= beta`, if
    /// one is found in the buckets the query selects.
    ///
    /// A query that is not a unit vector is normalized.
    ///
    /// # Panics
    ///
    /// If the query's dimension differs from the points', or it has a non-finite
    /// coordinate, or it is the zero vector.
    pub fn query(&self, query: &[f64]) -> AnnOutcome {
        let query = &*prepare_query(query, self.params().d);
        let levels = self.partition.candidate_filters(query);
        let beta = self.params().beta;
        let mut inspected_points = 0usize;
        let mut found = None;

        let matched_buckets = self.partition.buckets.for_each_match(&levels, |_, bucket| {
            for &point_id in bucket {
                inspected_points += 1;
                if dot_product(query, &self.data[point_id as usize]) >= beta {
                    found = Some(point_id as usize);
                    return false;
                }
            }
            true
        });

        AnnOutcome {
            point: found,
            probed_buckets: product_size(&levels),
            matched_buckets,
            inspected_points,
        }
    }

    /// Resolved parameters.
    pub fn params(&self) -> &Parameters {
        &self.partition.params
    }

    /// Number of input points that ended up in a bucket.
    pub fn stored_points(&self) -> usize {
        self.partition.stored_points
    }

    /// Number of non-empty buckets.
    pub fn occupied_buckets(&self) -> usize {
        self.partition.buckets.len()
    }

    /// The `i`-th input point, normalized if it was not a unit vector.
    pub fn point(&self, i: usize) -> &[f64] {
        &self.data[i]
    }

    /// The candidate filters `B_1, ..., B_t` selected by `query` in each factor.
    ///
    /// # Panics
    ///
    /// On the same malformed queries as [`LsfIndex::query`].
    pub fn candidate_filters(&self, query: &[f64]) -> Vec<Vec<u32>> {
        self.partition
            .candidate_filters(&prepare_query(query, self.params().d))
    }

    /// Measures the structure's actual heap footprint from allocated capacities
    /// (not lengths, not an asymptotic estimate). See [`MemoryFootprint`].
    pub fn memory_footprint(&self) -> MemoryFootprint {
        let vec_f64 = std::mem::size_of::<Vec<f64>>();
        let data_bytes = self.data.capacity() * vec_f64
            + self
                .data
                .iter()
                .map(|point| point.capacity() * std::mem::size_of::<f64>())
                .sum::<usize>();

        let filters = &self.partition.filters;
        let filters_bytes = filters.capacity() * std::mem::size_of::<FilterSet>()
            + filters
                .iter()
                .map(|set| {
                    set.gaussian_vectors.capacity() * vec_f64
                        + set
                            .gaussian_vectors
                            .iter()
                            .map(|filter| filter.capacity() * std::mem::size_of::<f64>())
                            .sum::<usize>()
                })
                .sum::<usize>();

        let bucket_index_bytes = self
            .partition
            .buckets
            .memory_bytes(|bucket| bucket.capacity() * std::mem::size_of::<u32>());

        MemoryFootprint {
            data_bytes,
            filters_bytes,
            bucket_index_bytes,
        }
    }

    /// Consumes the index, dropping the input points.
    pub(crate) fn into_partition(self) -> Partition {
        self.partition
    }
}

/// Heap memory a built index actually holds, measured from allocated capacities
/// (not lengths), broken down by what it is spent on.
///
/// `data_bytes` is what any exact baseline needs at minimum — a linear scan does
/// not save anything there, it just skips the rest of this breakdown. Everything
/// else is the price of sub-linear queries.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryFootprint {
    /// The input points, `Vec<Vec<f64>>`.
    pub data_bytes: usize,
    /// The `t * m_sub` Gaussian filters.
    pub filters_bytes: usize,
    /// The sorted bucket index: one key (`t` `u32`s) per non-empty bucket, plus
    /// the point ids stored in it.
    pub bucket_index_bytes: usize,
}

impl MemoryFootprint {
    /// Total heap memory held by the index.
    pub fn total(&self) -> usize {
        self.data_bytes + self.filters_bytes + self.bucket_index_bytes
    }

    /// Memory beyond what storing the raw points alone costs, i.e. what an exact
    /// linear-scan baseline over the same data would not need.
    pub fn overhead_bytes(&self) -> usize {
        self.filters_bytes + self.bucket_index_bytes
    }

    /// Overhead as a fraction of the raw data size.
    pub fn overhead_ratio(&self) -> f64 {
        self.overhead_bytes() as f64 / self.data_bytes.max(1) as f64
    }

    /// Human readable breakdown.
    pub fn summary(&self) -> String {
        format!(
            "{} total = {} raw data + {} overhead ({} filters, {} bucket index), \
             {:.1}% over the raw data a linear scan would need",
            format_bytes(self.total()),
            format_bytes(self.data_bytes),
            format_bytes(self.overhead_bytes()),
            format_bytes(self.filters_bytes),
            format_bytes(self.bucket_index_bytes),
            100. * self.overhead_ratio(),
        )
    }
}

/// Formats a byte count with a binary unit (KiB/MiB/GiB), two decimals.
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024. && unit + 1 < UNITS.len() {
        value /= 1024.;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Properties every ANNS algorithm must have, checked by each algorithm's tests.
#[cfg(test)]
pub(crate) fn assert_ann_contract<A: Algorithm>() {
    use crate::lsf::probe::ProbeIter;
    use crate::test_support::{config, planted_dataset, scaled_by_4};
    use crate::utils::normalize_vector;

    let (data, query) = planted_dataset(400, 20, 0.0, 13);
    let n = data.len();
    let index = LsfIndex::<A>::build(data.clone(), &config::<A>(14)).unwrap();
    assert_eq!(index.params().algorithm, A::NAME);

    // Partition property: no point in two buckets, so the counting variant has
    // sensitivity 1.
    let mut seen = vec![0usize; n];
    for (_, bucket) in index.partition.buckets.iter() {
        for &point_id in bucket {
            seen[point_id as usize] += 1;
        }
    }
    assert!(seen.iter().all(|&times| times <= 1));
    assert_eq!(
        seen.iter().filter(|&&times| times == 1).count(),
        index.stored_points()
    );

    // The query returns the first point at inner product >= beta found by
    // enumerating B_1 x ... x B_t literally, and never a point below beta.
    let outcome = index.query(&query);
    let levels = index.candidate_filters(&query);
    let expected = ProbeIter::new(&levels)
        .filter_map(|key| index.partition.buckets.get(&key))
        .flatten()
        .map(|&point_id| point_id as usize)
        .find(|&point_id| dot_product(&query, index.point(point_id)) >= index.params().beta);
    assert_eq!(outcome.point, expected);
    assert!(outcome.point.is_some(), "the planted cluster was not found");
    assert_eq!(outcome.probed_buckets, product_size(&levels));
    assert!(outcome.matched_buckets <= outcome.probed_buckets);

    // The construction is reproducible from the seed.
    let again = LsfIndex::<A>::build(data.clone(), &config::<A>(14)).unwrap();
    assert_eq!(again.query(&query).point, outcome.point);
    assert_eq!(again.occupied_buckets(), index.occupied_buckets());

    // Points and queries off the sphere are normalized: scaling them by 4 (exact
    // in floating point) gives the index of the normalized data.
    let mut normalized = data.clone();
    normalized
        .iter_mut()
        .for_each(|point| normalize_vector(point));
    let reference = LsfIndex::<A>::build(normalized, &config::<A>(14)).unwrap();
    let scaled = LsfIndex::<A>::build(scaled_by_4(&data), &config::<A>(14)).unwrap();
    let scaled_query: Vec<f64> = query.iter().map(|x| 4.0 * x).collect();
    assert_eq!(
        scaled.query(&scaled_query).point,
        reference.query(&query).point
    );
    assert_eq!(scaled.occupied_buckets(), reference.occupied_buckets());
    for i in 0..n {
        assert_eq!(scaled.point(i), reference.point(i));
    }

    // Malformed data sets are errors naming the offending point.
    let mut ragged = data.clone();
    ragged[5].pop();
    let error = LsfIndex::<A>::build(ragged, &config::<A>(14))
        .err()
        .unwrap();
    assert!(error.contains("point 5 has dimension"), "{error}");

    let memory = index.memory_footprint();
    assert!(memory.data_bytes >= n * index.params().d * std::mem::size_of::<f64>());
    assert!(memory.filters_bytes > 0 && memory.bucket_index_bytes > 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1536), "1.50 KiB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.00 MiB");
    }

    #[test]
    fn test_mismatched_dimensions_are_rejected() {
        let data = vec![vec![1.0, 0.0], vec![1.0]];
        let config = Config::default();
        assert!(LsfIndex::<crate::lsf::algorithms::Top1>::build(data, &config).is_err());
    }
}
