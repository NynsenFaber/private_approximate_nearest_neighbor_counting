//! TensorCloseTop-1 (Algorithm 5).

use super::LsfIndex;
use crate::lsf::algorithms;

/// **TensorCloseTop-1** (Algorithm 5): the linear space `(alpha, beta)`-ANN index.
///
/// # Construction
///
/// 1. `t = ceil(log^{1/8}(n) / (1 - alpha^2))` independent CloseTop-1 factors, each
///    with `m_sub = ceil(n^{(1/t) theta / (1 - alpha^2)})` Gaussian filters.
/// 2. In each factor, a point is assigned to its first filter in the collision band
///    `[sqrt(2 log m_sub) - (3/2) log log m_sub / sqrt(2 log m_sub), sqrt(2 log m_sub)]`.
/// 3. The `t` filter indices `(i_1, ..., i_t)` form the bucket key of the point. A
///    point missing in any factor is not stored.
/// 4. `eta = alpha sqrt(2 log m_sub) - sqrt(2 (1 - alpha^2) log log m_sub)`, from the
///    *per factor* `m_sub` (Algorithm 5 line 9), not from `m_sub^t`.
///
/// `t * m_sub = n^{o(1)}` stored filters thus simulate `m = m_sub^t` buckets
/// (*tensorization*, Proposition 16), and every point is stored at most once.
///
/// # Query
///
/// Each factor selects `B_i = {j : <a_{i,j}, q> >= eta}`; the query scans the
/// buckets of `B_1 x ... x B_t` and returns the first point at inner product
/// `>= beta`.
///
/// The product has `|I(q)|` keys, almost all of them empty (`318 554` keys and `278`
/// occupied buckets at `n = 10^5`). Instead of enumerating it as the pseudocode
/// does, the query walks the occupied keys, sorted once at build time, as a prefix
/// tree ([`crate::lsf::bucket_index`]). Same answers, cost proportional to the
/// buckets actually visited.
///
/// # Guarantees (Theorem 19)
///
/// When `alpha` is not too close to `1` and `beta` not too close to `alpha` (the
/// exact conditions are in Theorem 19), the query finds a close point with
/// probability `1 - o(1)` (Lemma 17), and opens `n^{theta + o(1)}`
/// buckets holding `n^{1 - theta (alpha - beta)^2 / ((1 - alpha^2)(1 - beta^2)) + o(1)}`
/// far points in expectation (Lemma 18).
///
/// | Cost | Bound |
/// | --- | --- |
/// | Pre-processing | `d n^{1 + o(1)}` |
/// | Space | `O(d n)` |
/// | Expected query (balanced `theta = rho`) | `d n^{rho + o(1)}` |
///
/// # In practice
///
/// The per factor success probability is below its asymptotic value at reachable
/// `n` (the band is wide), and the `t` factors multiply it. With the strict rule a
/// point must also collide in all `t` factors: only `42.6%` of the points are stored
/// at `n = 10^5`, `m_sub = 502`, `t = 3`. The default
/// [`Config::fallback_to_argmax`](crate::Config::fallback_to_argmax) stores all of
/// them; see [`CloseTop1`](super::CloseTop1).
///
/// # Example: your own data
///
/// ```
/// use ann_rust::anns::TensorCloseTop1;
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
/// let index = TensorCloseTop1::build(points, &config)?;
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
/// let error = TensorCloseTop1::build(ragged, &config).err().unwrap();
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
/// use ann_rust::anns::TensorCloseTop1;
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
/// // Every other parameter follows Algorithm 5.
/// let config = Config { alpha: 0.7, beta: 0.4, ..Config::default() };
/// let index = TensorCloseTop1::build(dataset.points, &config)?;
/// println!("{}", index.params().summary());
/// println!("{}", index.memory_footprint().summary());
///
/// let outcome = index.query(query);
/// if let Some(i) = outcome.point {
///     assert!(dot_product(query, index.point(i)) >= 0.4);
/// }
/// // The query opened only the non-empty buckets of B_1 x ... x B_t.
/// assert!(outcome.matched_buckets <= outcome.probed_buckets);
/// # Ok::<(), String>(())
/// ```
pub type TensorCloseTop1 = LsfIndex<algorithms::TensorCloseTop1>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{config, planted_dataset};
    use crate::utils::{collision_band, get_threshold};
    use crate::Config;

    #[test]
    fn test_ann_contract() {
        super::super::index::assert_ann_contract::<algorithms::TensorCloseTop1>();
    }

    /// Algorithm 5 line 9 computes `eta` from the *per factor* filter count `m_sub`,
    /// not from the `m_sub^t` simulated total. Getting this wrong would silently
    /// mis-tune every query: a threshold derived from `m_sub^t` is far too high, so
    /// `search` would return almost no candidate filters.
    #[test]
    fn test_eta_is_derived_from_m_sub_not_from_the_simulated_total() {
        let (data, _) = planted_dataset(200, 0, 0.0, 1);
        let index =
            TensorCloseTop1::build(data, &config::<algorithms::TensorCloseTop1>(0)).unwrap();
        let params = index.params();
        assert_eq!(params.t, 3);
        assert_eq!(params.eta, get_threshold(params.alpha, params.m_sub));
        assert!(params.eta < get_threshold(params.alpha, params.total_buckets() as usize));
        // Each factor's collision band likewise uses m_sub (Algorithm 5 line 4
        // constructs every factor as CloseTop-1(S, m_sub)).
        assert_eq!(params.band, collision_band(params.m_sub));
        // A query must be able to reach the filters points were assigned to.
        assert!(params.eta < params.band.0);
        for factor in &index.partition.filters {
            assert_eq!(factor.eta, params.eta);
            assert_eq!(factor.gaussian_vectors.len(), params.m_sub);
        }
    }

    /// A point is stored only if it collides in *every* factor: its key is the
    /// concatenation of the `t` filter indices, so one miss loses the point. This is
    /// the union bound over `t` factors in the proof of Lemma 17.
    #[test]
    fn test_a_point_missing_in_one_factor_is_not_stored() {
        let (data, _) = planted_dataset(500, 0, 0.0, 41);
        let strict = Config {
            fallback_to_argmax: false,
            ..config::<algorithms::TensorCloseTop1>(42)
        };
        let structure = TensorCloseTop1::build(data.clone(), &strict).unwrap();
        assert!(structure.stored_points() < data.len());
        let lenient =
            TensorCloseTop1::build(data.clone(), &config::<algorithms::TensorCloseTop1>(42))
                .unwrap();
        assert_eq!(lenient.stored_points(), data.len());
        for (key, _) in lenient.partition.buckets.iter() {
            assert_eq!(key.len(), 3);
        }
    }
}
