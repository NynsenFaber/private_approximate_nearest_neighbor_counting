//! Locality sensitive filtering data structures for approximate near neighbour
//! search and differentially private approximate near neighbour counting, following
//! *Aumüller, Boninsegna, Silvestri, "Differentially Private High-Dimensional
//! Approximate Range Counting, Revisited" (FORC 2025)*,
//! [doi:10.4230/LIPIcs.FORC.2025.15](https://doi.org/10.4230/LIPIcs.FORC.2025.15).
//!
//! # The four algorithms, three ways
//!
//! | Algorithm | Search ([`anns`]) | Counting ([`annc`]) | Private counting ([`annc::dp`]) |
//! | --- | --- | --- | --- |
//! | Top-1 (Algorithm 2) | [`anns::Top1`] | [`annc::Top1Counter`] | [`annc::dp::DpTop1`] (Algorithm 1) |
//! | CloseTop-1 (Algorithm 4) | [`anns::CloseTop1`] | [`annc::CloseTop1Counter`] | [`annc::dp::DpCloseTop1`] |
//! | TensorCloseTop-1 (Algorithm 5) | [`anns::TensorCloseTop1`] | [`annc::TensorCloseTop1Counter`] | [`annc::dp::DpTensorCloseTop1`] |
//! | TensorTop-1 | [`anns::TensorTop1`] | [`annc::TensorTop1Counter`] | [`annc::dp::DpTensorTop1`] |
//!
//! The counting structures are Algorithm 3 applied to the search structures, and the
//! private ones are Theorem 13 (truncated Laplace noise) applied to the counting
//! ones. Every structure is built from a [`Config`].
//!
//! # Input data
//!
//! A data set is a `Vec<Vec<f64>>` (search structures take ownership, counting
//! structures borrow `&[Vec<f64>]`): one vector per point, as many as you have, all
//! of the same dimension. Similarity is the **inner product of unit vectors**
//! (cosine similarity), so `alpha` and `beta` are cosine thresholds. `build` checks
//! the data set first ([`lsf::input`]):
//!
//! | Data set | Result |
//! | --- | --- |
//! | empty, or zero-dimensional points | `Err` |
//! | points of different dimensions | `Err("point i has dimension ..., but point 0 has dimension ...")` |
//! | a `NaN` or infinite coordinate | `Err("point i has the non-finite coordinate ...")` |
//! | a zero vector | `Err("point i is the zero vector ...")` |
//! | points that are not unit vectors | normalized, with one warning on stderr |
//!
//! A query is a `&[f64]` of the same dimension, normalized if needed. A query of
//! another dimension, with a non-finite coordinate, or equal to zero panics.
//!
//! # Typical use
//!
//! ```
//! use ann_rust::anns::TensorCloseTop1;
//! use ann_rust::annc::dp::TruncatedLaplace;
//! use ann_rust::annc::TensorCloseTop1Counter;
//! use ann_rust::utils::generate_unit_sphere_vectors;
//! use ann_rust::Config;
//!
//! let points = generate_unit_sphere_vectors(1_000, 32, 1);
//! let query = points[0].clone();
//! let config = Config { alpha: 0.7, beta: 0.4, ..Config::default() };
//!
//! // (alpha, beta)-ANN, Algorithm 5.
//! let index = TensorCloseTop1::build(points, &config)?;
//! let answer = index.query(&query);
//!
//! // (alpha, beta)-ANNC, Algorithm 3: the same partition, counters instead of points.
//! let counter = TensorCloseTop1Counter::from(index);
//! let exact = counter.count(&query).count;
//!
//! // (alpha, beta)-ANNC under (epsilon, delta)-DP, Theorem 13.
//! let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
//! let released = counter.into_private(mechanism, 42);
//! let estimate = released.query(&query).estimate;
//! # let _ = (answer, exact, estimate);
//! # Ok::<(), String>(())
//! ```
//!
//! # Crate layout
//!
//! * [`anns`] — the four search structures.
//! * [`annc`] — their counting versions, and [`annc::dp`] the private ones.
//! * [`lsf`] — the filters, partition and bucket index the twelve share.
//! * [`data`] — synthetic data sets with planted neighbours, brute force ground truth.
//! * [`utils`] — inner products, thresholds, seeded sampling.
//!
//! # Conventions
//!
//! * All points and queries are unit vectors; similarity is the inner product, so
//!   `alpha`/`beta` are cosine thresholds with `0 <= beta < alpha < 1`.
//! * `log` always means the natural logarithm, as in the paper's Gaussian tail
//!   analysis.
//! * Every random draw — filters, synthetic data, DP noise — is derived from one
//!   master seed through [`utils::derive_seed`], so a run is reproducible and
//!   independent of the number of threads.

pub mod annc;
pub mod anns;
pub mod data;
pub mod lsf;
pub mod utils;

pub use lsf::{Config, Parameters};

#[cfg(test)]
mod test_support;

/// Compiles and runs the Rust example of the top-level README.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
