//! Approximate near neighbour **counting** (ANNC), exact and differentially private.
//!
//! `(alpha, beta)`-ANNC (Definition 3): given a query `q`, return a number between
//! `|S ∩ B(q, alpha)|` and `|S ∩ B(q, beta)|`, where `B(q, r)` is the set of unit
//! vectors at inner product `>= r` from `q`.
//!
//! Each ANNS algorithm of [`crate::anns`] partitions the data set into buckets and
//! stores every point at most once. Algorithm 3 turns such a partition into a
//! counting structure: replace each bucket by its size, and answer a query with the
//! sum of the counters of the buckets it selects. No point is counted twice, and no
//! point needs to be kept.
//!
//! | ANNS | ANNC (this module) | DP-ANNC ([`dp`]) |
//! | --- | --- | --- |
//! | [`anns::Top1`](crate::anns::Top1) | [`Top1Counter`] | [`dp::DpTop1`] |
//! | [`anns::CloseTop1`](crate::anns::CloseTop1) | [`CloseTop1Counter`] | [`dp::DpCloseTop1`] |
//! | [`anns::TensorCloseTop1`](crate::anns::TensorCloseTop1) | [`TensorCloseTop1Counter`] | [`dp::DpTensorCloseTop1`] |
//! | [`anns::TensorTop1`](crate::anns::TensorTop1) | [`TensorTop1Counter`] | [`dp::DpTensorTop1`] |
//!
//! The answer may overshoot by the far points (inner product `< beta`) that share a
//! bucket with near ones, and undershoot by the near points stored in buckets the
//! query does not select (Lemma 12).
//!
//! All four counters are the generic [`LsfCounter`] with the algorithm as type
//! parameter, so they share one API:
//!
//! ```
//! use ann_rust::annc::dp::TruncatedLaplace;
//! use ann_rust::annc::TensorCloseTop1Counter; // or Top1Counter, ...
//! use ann_rust::utils::generate_unit_sphere_vectors;
//! use ann_rust::Config;
//!
//! let points = generate_unit_sphere_vectors(1_000, 32, 1);
//! let config = Config { alpha: 0.9, beta: 0.5, seed: 7, ..Config::default() };
//!
//! let counter = TensorCloseTop1Counter::build(&points, &config)?;
//! let exact = counter.count(&points[0]).count;
//!
//! // (1, 1e-6)-DP release of the counters; 42 seeds the noise.
//! let private = counter.release(TruncatedLaplace::new(1.0, 1e-6, 1.0)?, 42);
//! let noisy = private.query(&points[0]).estimate;
//! println!("exact {exact}, private {noisy:.1}");
//! # Ok::<(), String>(())
//! ```

mod close_top1;
mod counter;
pub mod dp;
mod tensor_close_top1;
mod tensor_top1;
mod top1;

pub use close_top1::CloseTop1Counter;
pub use counter::{CountOutcome, LsfCounter};
pub use tensor_close_top1::TensorCloseTop1Counter;
pub use tensor_top1::TensorTop1Counter;
pub use top1::Top1Counter;
