//! TensorTop-1: tensorization with the Top-1 rule.

use super::LsfIndex;
use crate::lsf::algorithms;

/// **TensorTop-1**: [`TensorCloseTop1`](super::TensorCloseTop1) with the Top-1
/// assignment rule in every factor. Not in the paper.
///
/// # Construction
///
/// 1. `t = ceil(log^{1/8}(n) / (1 - alpha^2))` independent factors, each with
///    `m_sub = ceil(n^{(1/t) theta / (1 - alpha^2)})` Gaussian filters — the same
///    parameters as TensorCloseTop-1.
/// 2. In each factor, a point is assigned to the filter maximizing `<a, x>`.
/// 3. The `t` argmax indices `(i_1, ..., i_t)` form the bucket key of the point.
/// 4. `eta = alpha sqrt(2 log m_sub) - sqrt(2 (1 - alpha^2) log log m_sub)`.
///
/// # Query
///
/// Same as TensorCloseTop-1: scan the buckets of `B_1 x ... x B_t`, with
/// `B_i = {j : <a_{i,j}, q> >= eta}`, and return the first point at inner product
/// `>= beta`.
///
/// # Guarantees
///
/// The paper states none. Each factor is a Top-1 structure, whose analysis
/// (Lemmas 7 and 8) relies on the limiting distribution of the extreme concomitant
/// (Theorem 5); CloseTop-1 exists to avoid that assumption, which is why
/// Algorithm 5 tensorizes CloseTop-1. Space (`O(d n)`, one bucket per point) and
/// pre-processing (`t * m_sub` inner products per point) are the same as
/// TensorCloseTop-1.
///
/// # In practice
///
/// Every point is stored: there is no collision band to miss, so this is the
/// natural baseline for TensorCloseTop-1 with
/// [`Config::fallback_to_argmax`](crate::Config::fallback_to_argmax), which uses
/// the argmax only for points that miss the band.
///
/// # Example: your own data
///
/// ```
/// use ann_rust::anns::TensorTop1;
/// use ann_rust::Config;
///
/// // Your data set: one `Vec<f64>` per point, all of the same dimension (here 8
/// // points in dimension 4). Load it from a CSV file, a NumPy export or an
/// // embedding model; `f32` embeddings convert with `x as f64`.
/// let points: Vec<Vec<f64>> = vec![
///     vec![0.90, 0.10, 0.00, 0.40],
///     vec![0.85, 0.15, 0.05, 0.45],
///     vec![0.10, 0.80, 0.50, 0.00],
///     vec![0.00, 0.20, 0.90, 0.30],
///     vec![0.50, 0.50, 0.50, 0.50],
///     vec![0.30, 0.00, 0.10, 0.90],
///     vec![-0.70, 0.20, 0.10, 0.60],
///     vec![0.20, -0.90, 0.30, 0.10],
/// ];
/// // Most rows are not unit vectors: `build` normalizes them and prints a warning
/// // on stderr. Normalize them first (`ann_rust::utils::normalize_vector`) to
/// // silence it.
/// let config = Config { alpha: 0.9, beta: 0.5, seed: 1, ..Config::default() };
/// let index = TensorTop1::build(points, &config)?;
///
/// // A query has the dimension of the points; it is normalized too.
/// let query = [1.0, 0.1, 0.0, 0.4];
/// match index.query(&query).point {
///     Some(i) => println!("near neighbour: point {i} = {:?}", index.point(i)),
///     None => println!("no point at inner product >= beta in the selected buckets"),
/// }
///
/// // A malformed data set is an error that names the offending point.
/// let ragged = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0]];
/// let error = TensorTop1::build(ragged, &config).err().unwrap();
/// assert_eq!(
///     error,
///     "point 1 has dimension 2, but point 0 has dimension 3: all points must have the \
///      same dimension"
/// );
/// # Ok::<(), String>(())
/// ```
///
/// # Example: synthetic benchmark data
///
/// ```
/// use ann_rust::anns::TensorTop1;
/// use ann_rust::data::{generate, GeneratorConfig, PlantConfig};
/// use ann_rust::utils::dot_product;
/// use ann_rust::Config;
///
/// let dataset = generate(&GeneratorConfig {
///     n: 10_000,
///     d: 64,
///     queries: 1,
///     plant: PlantConfig { count: 5, similarity: 0.7, tightness: 0.0 },
///     seed: 1,
/// })?;
/// let query = &dataset.queries[0];
///
/// let config = Config { alpha: 0.7, beta: 0.4, ..Config::default() };
/// let index = TensorTop1::build(dataset.points, &config)?;
/// assert_eq!(index.stored_points(), 10_000);
///
/// if let Some(i) = index.query(query).point {
///     assert!(dot_product(query, index.point(i)) >= 0.4);
/// }
/// # Ok::<(), String>(())
/// ```
pub type TensorTop1 = LsfIndex<algorithms::TensorTop1>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsf::filters::assign_top1;
    use crate::test_support::{config, planted_dataset};

    #[test]
    fn test_ann_contract() {
        super::super::index::assert_ann_contract::<algorithms::TensorTop1>();
    }

    /// Every point is stored, and component `i` of its key is its argmax filter in
    /// factor `i`.
    #[test]
    fn test_keys_are_the_argmax_filters_of_each_factor() {
        let (data, _) = planted_dataset(300, 0, 0.0, 5);
        let index = TensorTop1::build(data.clone(), &config::<algorithms::TensorTop1>(6)).unwrap();
        assert_eq!(index.params().t, 3);
        assert_eq!(index.stored_points(), data.len());
        for (key, bucket) in index.partition.buckets.iter() {
            for &point_id in bucket {
                for (factor, &filter) in index.partition.filters.iter().zip(key) {
                    let argmax = assign_top1(&data[point_id as usize], &factor.gaussian_vectors);
                    assert_eq!(filter, argmax);
                }
            }
        }
    }

    /// With the same seed, TensorTop-1 and TensorCloseTop-1 draw the same filters:
    /// they differ only in the assignment rule.
    #[test]
    fn test_same_filters_as_tensor_close_top1() {
        let (data, _) = planted_dataset(200, 0, 0.0, 7);
        let top1 = TensorTop1::build(data.clone(), &config::<algorithms::TensorTop1>(8)).unwrap();
        let close =
            super::super::TensorCloseTop1::build(data, &config::<algorithms::TensorCloseTop1>(8))
                .unwrap();
        for (a, b) in top1.partition.filters.iter().zip(&close.partition.filters) {
            assert_eq!(a.gaussian_vectors, b.gaussian_vectors);
            assert_eq!(a.eta, b.eta);
        }
    }
}
