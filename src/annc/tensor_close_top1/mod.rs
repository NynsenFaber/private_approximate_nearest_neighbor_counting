//! TensorCloseTop-1 counting structure (Algorithm 3 on Algorithm 5).

use super::LsfCounter;
use crate::lsf::algorithms;

/// `(alpha, beta)`-ANNC with **TensorCloseTop-1**: Algorithm 3 applied to
/// [`anns::TensorCloseTop1`](crate::anns::TensorCloseTop1).
///
/// # Construction
///
/// Build the TensorCloseTop-1 partition (`t` CloseTop-1 factors of `m_sub`
/// filters; a point's bucket is its `t` filter indices), then replace every bucket
/// by its size. Only the `t * m_sub` filters and the non-empty counters are kept —
/// at most one counter per point, although `m_sub^t` buckets are simulated.
///
/// # Query
///
/// Sum the counters of the buckets in `B_1 x ... x B_t`, visiting only the non-empty
/// ones.
///
/// # Guarantees (Lemma 12 with Lemmas 17-18)
///
/// With probability at least `2/3`,
/// `(1 - o(1)) |S ∩ B(q, alpha)| <= ans <= |S ∩ B(q, beta)| + K`, with
/// `K = n^{rho + o(1)}` far points for the balanced `theta = rho`, in space
/// `O(n)` counters plus `n^{o(1)}` filters.
///
/// # Example: your own data
///
/// ```
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
/// // A query has the dimension of the points; it is normalized too.
/// let query = [1.0, 0.1, 0.0, 0.4];
/// println!("count: {}", counter.count(&query).count);
///
/// // A malformed data set is an error that names the offending point.
/// let mut broken = points.clone();
/// broken[2][1] = f64::NAN;
/// let error = TensorCloseTop1Counter::build(&broken, &config).err().unwrap();
/// assert_eq!(error, "point 2 has the non-finite coordinate NaN at index 1");
/// # Ok::<(), String>(())
/// ```
///
/// # Example: synthetic benchmark data
///
/// ```
/// use ann_rust::anns::TensorCloseTop1;
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
///
/// // Either build the counter directly...
/// let counter = TensorCloseTop1Counter::build(&dataset.points, &config)?;
/// // ...or turn a search index into one, dropping its points.
/// let index = TensorCloseTop1::build(dataset.points.clone(), &config)?;
/// let same = TensorCloseTop1Counter::from(index);
///
/// let answer = counter.count(query);
/// assert_eq!(answer.count, same.count(query).count);
/// println!(
///     "estimate {}, truth {}, {} counters summed",
///     answer.count,
///     exact_count(&dataset.points, query, 0.7),
///     answer.matched_buckets,
/// );
/// # Ok::<(), String>(())
/// ```
pub type TensorCloseTop1Counter = LsfCounter<algorithms::TensorCloseTop1>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_contract() {
        super::super::counter::assert_count_contract::<algorithms::TensorCloseTop1>();
    }
}
