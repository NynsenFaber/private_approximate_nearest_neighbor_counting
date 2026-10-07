//! DP-ANNC with CloseTop-1.

use super::DpLsfCounter;
use crate::lsf::algorithms;

/// **DP-CloseTop-1**: the [`CloseTop1Counter`](crate::annc::CloseTop1Counter)
/// counters released with the truncated Laplace mechanism.
///
/// # Construction
///
/// Build the CloseTop-1 counters, add truncated Laplace noise to every non-empty
/// counter, suppress those at or below `1 + A` (see [`crate::annc::dp`]), and
/// publish the filters, `eta` and the noisy counters.
///
/// # Query
///
/// Sum the noisy counters of the filters with `<a_i, q> >= eta`.
///
/// # Guarantees (Theorem 13 with Lemma 15)
///
/// The same as [`DpTop1`](super::DpTop1), without the `n -> infinity` assumption:
/// `(epsilon, delta)`-DP, and with probability at least `2/3` an additive error
/// `O(log(1/delta) / epsilon * n^{rho + o(1)})` for the balanced `theta = rho`.
/// Each point is in at most one bucket with or without
/// [`Config::fallback_to_argmax`](crate::Config::fallback_to_argmax), so the
/// sensitivity is `1` either way.
///
/// # Example: your own data
///
/// ```
/// use ann_rust::annc::dp::{DpCloseTop1, TruncatedLaplace};
/// use ann_rust::annc::CloseTop1Counter;
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
/// let counter = CloseTop1Counter::build(&points, &config)?;
///
/// // (1, 1e-6)-DP for add/remove neighbours (sensitivity 1). The seed makes the
/// // noise reproducible; a real release must draw it from OS entropy.
/// let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
/// let private: DpCloseTop1 = counter.into_private(mechanism, 42);
///
/// // A query has the dimension of the points; it is normalized too.
/// let query = [1.0, 0.1, 0.0, 0.4];
/// println!("private count: {:.1}", private.query(&query).estimate);
///
/// // A malformed data set is an error that names the offending point.
/// let mut broken = points.clone();
/// broken[3] = vec![0.0; 4];
/// let error = CloseTop1Counter::build(&broken, &config).err().unwrap();
/// assert!(error.starts_with("point 3 is the zero vector"));
/// # Ok::<(), String>(())
/// ```
///
/// # Example: synthetic benchmark data
///
/// ```
/// use ann_rust::annc::dp::{DpCloseTop1, TruncatedLaplace};
/// use ann_rust::annc::CloseTop1Counter;
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
/// let counter = CloseTop1Counter::build(&dataset.points, &config)?;
///
/// let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
/// let private: DpCloseTop1 = counter.release(mechanism, 42);
/// println!("{} counters released", private.released_buckets());
/// println!("estimate {:.1}", private.query(&dataset.queries[0]).estimate);
/// # Ok::<(), String>(())
/// ```
pub type DpCloseTop1 = DpLsfCounter<algorithms::CloseTop1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dp_contract() {
        super::super::private_counter::assert_dp_contract::<algorithms::CloseTop1>();
    }
}
