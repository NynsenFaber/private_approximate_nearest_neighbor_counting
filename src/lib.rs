//! Locality sensitive filtering data structures for approximate near neighbour
//! search and differentially private approximate near neighbour counting, following
//! *Aumüller, Boninsegna, Silvestri, "Differentially Private High-Dimensional
//! Approximate Range Counting, Revisited" (FORC 2025)*,
//! [doi:10.4230/LIPIcs.FORC.2025.15](https://doi.org/10.4230/LIPIcs.FORC.2025.15).
//!
//! # What is implemented
//!
//! | Paper | Here |
//! | --- | --- |
//! | Algorithm 4, `CloseTop-1` | [`tensor_data_structures::close_top1::CloseTop1`] |
//! | Algorithm 5, `TensorCloseTop-1` | [`tensor_data_structures::tensor_close_top1::TensorCloseTop1`] |
//! | Algorithm 3, ANN to ANNC | [`tensor_data_structures::tensor_close_top1::TensorCloseTop1::count`] |
//! | Theorem 13, DP-ANNC | [`tensor_data_structures::dp_annc::DpAnnc`] |
//! | Truncated Laplace | [`dp::truncated_laplace::TruncatedLaplace`] |
//!
//! # Typical use
//!
//! ```no_run
//! use ann_rust::tensor_data_structures::tensor_close_top1::{Config, TensorCloseTop1};
//! use ann_rust::dp::truncated_laplace::TruncatedLaplace;
//!
//! # fn main() -> Result<(), String> {
//! let points: Vec<Vec<f64>> = vec![/* unit vectors */];
//! let config = Config { alpha: 0.7, beta: 0.4, ..Config::default() };
//!
//! // (alpha, beta)-ANN, Algorithm 5.
//! let structure = TensorCloseTop1::build(points, &config)?;
//! let answer = structure.query(&[/* query */]);
//!
//! // (alpha, beta)-ANNC under (epsilon, delta)-DP, Theorem 13.
//! let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
//! let released = structure.into_private(mechanism, 42);
//! let estimate = released.query(&[/* query */]).estimate;
//! # Ok(())
//! # }
//! ```
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

pub mod cli;
pub mod data;
pub mod utils;

pub mod dp {
    //! Differentially private mechanisms used to release bucket counters.
    pub mod truncated_laplace;
}

pub mod tensor_data_structures {
    //! Tensorized structures: `t` factors of `m_sub` filters simulate `m_sub^t`
    //! buckets, which keeps the space linear and the pre-processing `n^{1+o(1)}`.
    pub mod bucket_index;
    pub mod close_top1;
    pub mod dp_annc;
    pub mod probe;
    pub mod tensor_close_top1;
}
