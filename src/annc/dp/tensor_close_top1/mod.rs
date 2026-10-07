//! DP-ANNC with TensorCloseTop-1 (Theorem 13 on Algorithm 5).

use super::DpLsfCounter;
use crate::lsf::algorithms;

/// **DP-TensorCloseTop-1**: the
/// [`TensorCloseTop1Counter`](crate::annc::TensorCloseTop1Counter) counters
/// released with the truncated Laplace mechanism.
///
/// # Construction
///
/// Build the TensorCloseTop-1 counters, add truncated Laplace noise to every
/// non-empty counter, suppress those at or below `1 + A` (see [`crate::annc::dp`]),
/// and publish the `t * m_sub` filters, `eta` and the noisy counters. Only occupied
/// buckets are noised: the `m_sub^t` simulated buckets are never materialized.
///
/// # Query
///
/// Sum the noisy counters of the buckets in `B_1 x ... x B_t`.
///
/// # Guarantees (Theorem 13 with Theorem 19, Table 1)
///
/// `(epsilon, delta)`-DP, and with probability at least `2/3` an additive error
/// `O(log(1/delta) / epsilon * n^{rho + o(1)})` for the balanced `theta = rho`, with
/// pre-processing `n^{1 + o(1)}`, query time `n^{rho + o(1)}` and space `O(n)` —
/// the guarantees of Andoni et al. (NeurIPS 2023) with a simpler structure.
///
/// # Example: your own data
///
/// ```
/// use ann_rust::annc::dp::{DpTensorCloseTop1, TruncatedLaplace};
/// use ann_rust::annc::TensorCloseTop1Counter;
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
/// let counter = TensorCloseTop1Counter::build(&points, &config)?;
///
/// // (1, 1e-6)-DP for add/remove neighbours (sensitivity 1). The seed makes the
/// // noise reproducible; a real release must draw it from OS entropy.
/// let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
/// let private: DpTensorCloseTop1 = counter.into_private(mechanism, 42);
///
/// // A query has the dimension of the points; it is normalized too.
/// let query = [1.0, 0.1, 0.0, 0.4];
/// println!("private count: {:.1}", private.query(&query).estimate);
///
/// // A malformed data set is an error that names the offending point.
/// let mut broken = points.clone();
/// broken[3] = vec![0.0; 4];
/// let error = TensorCloseTop1Counter::build(&broken, &config).err().unwrap();
/// assert!(error.starts_with("point 3 is the zero vector"));
/// # Ok::<(), String>(())
/// ```
///
/// # Example: synthetic benchmark data
///
/// ```
/// use ann_rust::annc::dp::{DpTensorCloseTop1, TruncatedLaplace};
/// use ann_rust::annc::TensorCloseTop1Counter;
/// use ann_rust::data::{exact_count, generate, GeneratorConfig, PlantConfig};
/// use ann_rust::Config;
///
/// let dataset = generate(&GeneratorConfig {
///     n: 10_000,
///     d: 64,
///     queries: 1,
///     plant: PlantConfig { count: 500, similarity: 0.7, tightness: 0.95 },
///     seed: 1,
/// })?;
/// let query = &dataset.queries[0];
/// let config = Config { alpha: 0.7, beta: 0.4, ..Config::default() };
/// let counter = TensorCloseTop1Counter::build(&dataset.points, &config)?;
///
/// // Several budgets on the same partition: only the noise is redrawn.
/// for epsilon in [0.5, 1.0, 4.0] {
///     let mechanism = TruncatedLaplace::new(epsilon, 1e-6, 1.0)?;
///     let private: DpTensorCloseTop1 = counter.release(mechanism, 42);
///     println!("epsilon {epsilon}: estimate {:.1}", private.query(query).estimate);
/// }
/// println!("truth {}", exact_count(&dataset.points, query, 0.7));
/// # Ok::<(), String>(())
/// ```
pub type DpTensorCloseTop1 = DpLsfCounter<algorithms::TensorCloseTop1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_contract() {
        super::super::private_counter::assert_dp_contract::<algorithms::TensorCloseTop1>();
    }
}
