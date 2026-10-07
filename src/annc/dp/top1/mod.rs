//! DPTop-1 (Algorithm 1).

use super::DpLsfCounter;
use crate::lsf::algorithms;

/// **DPTop-1** (Algorithm 1): the [`Top1Counter`](crate::annc::Top1Counter)
/// counters released with the truncated Laplace mechanism.
///
/// # Construction
///
/// 1. Sample `m` Gaussian filters and count, for each filter, the points whose
///    argmax filter it is (Algorithm 1 lines 2-7).
/// 2. `make_private`: add truncated Laplace noise to every non-empty counter and
///    suppress those at or below `1 + A` (see [`crate::annc::dp`]).
/// 3. Publish the filters, `eta` and the noisy counters.
///
/// # Query
///
/// Sum the noisy counters of the filters with `<a_i, q> >= eta`.
///
/// # Guarantees (Theorem 1, `n -> infinity`)
///
/// The release is `(epsilon, delta)`-DP, and with probability at least `2/3` the
/// answer lies in
/// `[(1 - o(1)) |S ∩ B(q, alpha)| - E, |S ∩ B(q, beta)| + E]` with
/// `E = O(log(1/delta) / epsilon * n^{rho + o(1)})` for the balanced `theta = rho`.
/// Costs are those of [`anns::Top1`](crate::anns::Top1).
///
/// # Example: your own data
///
/// ```
/// use ann_rust::annc::dp::{DpTop1, TruncatedLaplace};
/// use ann_rust::annc::Top1Counter;
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
/// let counter = Top1Counter::build(&points, &config)?;
///
/// // (1, 1e-6)-DP for add/remove neighbours (sensitivity 1). The seed makes the
/// // noise reproducible; a real release must draw it from OS entropy.
/// let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
/// let private: DpTop1 = counter.into_private(mechanism, 42);
///
/// // A query has the dimension of the points; it is normalized too.
/// let query = [1.0, 0.1, 0.0, 0.4];
/// println!("private count: {:.1}", private.query(&query).estimate);
///
/// // A malformed data set is an error that names the offending point.
/// let mut broken = points.clone();
/// broken[3] = vec![0.0; 4];
/// let error = Top1Counter::build(&broken, &config).err().unwrap();
/// assert!(error.starts_with("point 3 is the zero vector"));
/// # Ok::<(), String>(())
/// ```
///
/// # Example: synthetic benchmark data
///
/// ```
/// use ann_rust::annc::dp::{DpTop1, TruncatedLaplace};
/// use ann_rust::annc::Top1Counter;
/// use ann_rust::data::{generate, GeneratorConfig, PlantConfig};
/// use ann_rust::Config;
///
/// let dataset = generate(&GeneratorConfig {
///     n: 2_000,
///     d: 32,
///     queries: 1,
///     plant: PlantConfig { count: 300, similarity: 0.9, tightness: 0.999 },
///     seed: 1,
/// })?;
/// let config = Config { alpha: 0.9, beta: 0.5, m_sub: Some(512), ..Config::default() };
/// let counter = Top1Counter::build(&dataset.points, &config)?;
///
/// // (1, 1e-6)-DP with sensitivity 1 (add/remove one point); 42 seeds the noise.
/// let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
/// let private: DpTop1 = counter.into_private(mechanism, 42);
/// let outcome = private.query(&dataset.queries[0]);
/// println!(
///     "estimate {:.1}, noise at most {:.1}",
///     outcome.estimate,
///     private.error_bound(outcome.matched_buckets),
/// );
/// # Ok::<(), String>(())
/// ```
pub type DpTop1 = DpLsfCounter<algorithms::Top1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_contract() {
        super::super::private_counter::assert_dp_contract::<algorithms::Top1>();
    }
}
