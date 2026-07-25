//! Locality sensitive filtering data structures for approximate near neighbour
//! search and differentially private approximate near neighbour counting, following
//! *Aumüller, Boninsegna, Silvestri, "Differentially Private High-Dimensional
//! Approximate Range Counting, Revisited" (FORC 2025)*.
//!
//! The entry point is [`tensor_data_structures::tensor_close_top1::TensorCloseTop1`]
//! (Algorithm 5), which solves `(alpha, beta)`-ANN and, once released through
//! [`tensor_data_structures::dp_annc::DpAnnc`], `(alpha, beta)`-ANNC under
//! differential privacy.

pub mod checks;
pub mod cli;
pub mod data;
pub mod utils;

pub mod dp {
    //! Differentially private mechanisms used to release bucket counters.
    pub mod truncated_laplace;
}

pub mod simple_data_structures {
    //! Non tensorized structures: `n^{theta/(1-alpha^2)}` filters are materialized.
    pub mod close_top1;
    pub mod query;
    pub mod top1;
}

pub mod tensor_data_structures {
    //! Tensorized structures: `t` factors of `m_sub` filters simulate `m_sub^t`
    //! buckets, which keeps the space linear and the pre-processing `n^{1+o(1)}`.
    pub mod bucket_index;
    pub mod close_top1;
    pub mod dp_annc;
    pub mod probe;
    pub mod query;
    pub mod tensor_close_top1;
    pub mod tensor_top1;
    pub mod top1;
}
