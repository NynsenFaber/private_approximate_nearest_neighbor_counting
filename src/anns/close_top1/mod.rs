//! CloseTop-1 (Algorithm 4).

use super::LsfIndex;
use crate::lsf::algorithms;

/// **CloseTop-1** (Algorithm 4): Top-1 without the asymptotic assumption.
///
/// # Construction
///
/// 1. Sample `m = ceil(n^{theta / (1 - alpha^2)})` Gaussian filters
///    `a_1, ..., a_m ~ N(0, I_d)`.
/// 2. Store every point `x` in the bucket of the **first** filter whose inner
///    product falls in the collision band
///    `[sqrt(2 log m) - (3/2) log log m / sqrt(2 log m), sqrt(2 log m)]`.
///    A point no filter accepts is not stored.
/// 3. Set the query threshold
///    `eta = alpha sqrt(2 log m) - sqrt(2 (1 - alpha^2) log log m)`.
///
/// # Query
///
/// Same as [`Top1`](super::Top1): open every bucket with `<a_i, q> >= eta`, return
/// the first point at inner product `>= beta`.
///
/// # Guarantees (Lemma 15)
///
/// Top-1 assigns a point to the *maximum* `<a_i, x>`, whose distribution is only
/// known in the limit (Theorem 5). The band bounds `<a, x>` from both sides by
/// construction, so Lemmas 7 and 8 — and with them Theorem 9 and Corollary 10 —
/// hold for CloseTop-1 at every `n`, with the same costs as Top-1. A point misses
/// every filter with probability `m^{-Omega(1)}` (Lemma 23).
///
/// # In practice
///
/// * `m` is as large as Top-1's: set [`Config::m_sub`](crate::Config::m_sub) (here:
///   `m`) on more than a few hundred points.
/// * The constant in Lemma 23 is small. At `m = 502` a single filter accepts a
///   point with probability `0.0028`, so a quarter of the points collide with
///   nothing (see [`crate::lsf::filters`] for the erratum behind this).
///   [`Config::fallback_to_argmax`](crate::Config::fallback_to_argmax), on by
///   default, stores such a point at its argmax filter instead — the Top-1 rule.
///   The point still occupies one bucket, so privacy is unaffected. Set it to
///   `false` for the literal Algorithm 4.
///
/// # Example: your own data
///
/// ```
/// use ann_rust::anns::CloseTop1;
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
/// let index = CloseTop1::build(points, &config)?;
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
/// let error = CloseTop1::build(ragged, &config).err().unwrap();
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
/// use ann_rust::anns::CloseTop1;
/// use ann_rust::data::{generate, GeneratorConfig, PlantConfig};
/// use ann_rust::utils::dot_product;
/// use ann_rust::Config;
///
/// let dataset = generate(&GeneratorConfig {
///     n: 2_000,
///     d: 32,
///     queries: 1,
///     plant: PlantConfig { count: 5, similarity: 0.9, tightness: 0.0 },
///     seed: 1,
/// })?;
/// let query = &dataset.queries[0];
///
/// // The literal Algorithm 4: points that no filter accepts are dropped.
/// let config = Config {
///     alpha: 0.9,
///     beta: 0.5,
///     m_sub: Some(512),
///     fallback_to_argmax: false,
///     ..Config::default()
/// };
/// let index = CloseTop1::build(dataset.points, &config)?;
/// assert!(index.stored_points() <= 2_000);
///
/// if let Some(i) = index.query(query).point {
///     assert!(dot_product(query, index.point(i)) >= 0.5);
/// }
/// # Ok::<(), String>(())
/// ```
pub type CloseTop1 = LsfIndex<algorithms::CloseTop1>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{config, planted_dataset};
    use crate::utils::dot_product;
    use crate::Config;

    #[test]
    fn test_ann_contract() {
        super::super::index::assert_ann_contract::<algorithms::CloseTop1>();
    }

    /// Without the fallback, a stored point lies in the collision band of its
    /// filter, and some points are not stored at all.
    #[test]
    fn test_strict_assignment_follows_algorithm_4() {
        let (data, _) = planted_dataset(500, 0, 0.0, 1);
        let strict = Config {
            fallback_to_argmax: false,
            ..config::<algorithms::CloseTop1>(2)
        };
        let index = CloseTop1::build(data.clone(), &strict).unwrap();
        assert_eq!(index.params().t, 1);
        assert!(index.stored_points() < data.len());

        let (lower, upper) = index.params().band;
        let filters = &index.partition.filters[0].gaussian_vectors;
        for (key, bucket) in index.partition.buckets.iter() {
            for &point_id in bucket {
                let inner_product =
                    dot_product(&data[point_id as usize], &filters[key[0] as usize]);
                assert!(inner_product >= lower && inner_product <= upper);
            }
        }
    }

    /// With the fallback (the default) every point is stored.
    #[test]
    fn test_fallback_stores_every_point() {
        let (data, _) = planted_dataset(500, 0, 0.0, 1);
        let index = CloseTop1::build(data.clone(), &config::<algorithms::CloseTop1>(2)).unwrap();
        assert_eq!(index.stored_points(), data.len());
    }
}
