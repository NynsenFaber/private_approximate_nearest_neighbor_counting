//! Top-1 (Algorithm 2).

use super::LsfIndex;
use crate::lsf::algorithms;

/// **Top-1** (Algorithm 2): the simplest filter based `(alpha, beta)`-ANN index.
///
/// # Construction
///
/// 1. Sample `m = ceil(n^{theta / (1 - alpha^2)})` Gaussian filters
///    `a_1, ..., a_m ~ N(0, I_d)`.
/// 2. Store every point `x` in the bucket of the filter that maximizes `<a_i, x>`.
/// 3. Set the query threshold
///    `eta = alpha sqrt(2 log m) - sqrt(2 (1 - alpha^2) log log m)`.
///
/// # Query
///
/// `search(q)` selects every filter with `<a_i, q> >= eta`; `query(q)` scans the
/// selected buckets and returns the first point at inner product `>= beta`.
///
/// Why `eta`: by the theory of concomitant order statistics (Theorem 5), a point
/// at inner product `alpha` from `q` sits at a filter whose inner product with `q`
/// is asymptotically `N(alpha sqrt(2 log m), 1 - alpha^2)`. `eta` lies
/// `sqrt(2 log log m)` standard deviations below that mean.
///
/// # Guarantees (Theorem 9, `n -> infinity`)
///
/// With probability `1 - o(1)` the query returns a point at inner product
/// `>= beta` whenever one at `>= alpha` exists (Lemma 7). With the balanced
/// `theta = rho`, a query opens `n^{rho + o(1)}` buckets holding `n^{rho + o(1)}`
/// far points in expectation (Lemma 8, Corollary 10).
///
/// | Cost | Bound |
/// | --- | --- |
/// | Pre-processing | `O(d n^{1 + theta / (1 - alpha^2)})` |
/// | Space | `O(d max(n, n^{theta / (1 - alpha^2)}))` |
/// | Expected query | `O(d max(n^{theta / (1 - alpha^2)}, n^{1 - theta (alpha - beta)^2 / ((1 - alpha^2)(1 - beta^2)) + o(1)}))` |
///
/// The analysis uses the *limiting* distribution of the extreme concomitant, so it
/// holds only asymptotically. [`CloseTop1`](super::CloseTop1) removes that
/// assumption.
///
/// # In practice
///
/// Every point is stored, but `m` grows faster than `n`: at `alpha = 0.7`,
/// `beta = 0.4`, the default `m` is `n^{1.62}`, and every point is compared with
/// every filter. Set [`Config::m_sub`](crate::Config::m_sub) (here: `m`) to use
/// Top-1 on more than a few hundred points.
/// [`TensorTop1`](super::TensorTop1) keeps the Top-1 rule with `n^{o(1)}` filters.
///
/// # Example: your own data
///
/// ```
/// use ann_rust::anns::Top1;
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
/// let index = Top1::build(points, &config)?;
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
/// let error = Top1::build(ragged, &config).err().unwrap();
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
/// use ann_rust::anns::Top1;
/// use ann_rust::data::{generate, GeneratorConfig, PlantConfig};
/// use ann_rust::utils::dot_product;
/// use ann_rust::Config;
///
/// // 2 000 random unit vectors, 5 of them planted at inner product 0.9 from the query.
/// let dataset = generate(&GeneratorConfig {
///     n: 2_000,
///     d: 32,
///     queries: 1,
///     plant: PlantConfig { count: 5, similarity: 0.9, tightness: 0.0 },
///     seed: 1,
/// })?;
/// let query = &dataset.queries[0];
///
/// // The default m = n^{theta / (1 - alpha^2)} is far above n: choose m explicitly.
/// let config = Config { alpha: 0.9, beta: 0.5, m_sub: Some(512), ..Config::default() };
/// let index = Top1::build(dataset.points, &config)?;
/// assert_eq!(index.params().t, 1);
/// assert_eq!(index.stored_points(), 2_000);
///
/// if let Some(i) = index.query(query).point {
///     assert!(dot_product(query, index.point(i)) >= 0.5);
/// }
/// # Ok::<(), String>(())
/// ```
pub type Top1 = LsfIndex<algorithms::Top1>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsf::filters::assign_top1;
    use crate::test_support::{config, planted_dataset};

    #[test]
    fn test_ann_contract() {
        super::super::index::assert_ann_contract::<algorithms::Top1>();
    }

    /// Every point is stored, in the bucket of its argmax filter.
    #[test]
    fn test_every_point_is_stored_at_its_argmax_filter() {
        let (data, _) = planted_dataset(300, 0, 0.0, 3);
        let index = Top1::build(data.clone(), &config::<algorithms::Top1>(4)).unwrap();
        assert_eq!(index.params().t, 1);
        assert_eq!(index.stored_points(), data.len());
        let filters = &index.partition.filters[0].gaussian_vectors;
        for (key, bucket) in index.partition.buckets.iter() {
            for &point_id in bucket {
                assert_eq!(key[0], assign_top1(&data[point_id as usize], filters));
            }
        }
    }
}
