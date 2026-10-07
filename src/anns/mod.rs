//! Approximate near neighbour **search** (ANNS).
//!
//! `(alpha, beta)`-ANN (Definition 2): given a query `q` such that the data set has
//! a point at inner product `>= alpha` from `q`, return a point at inner product
//! `>= beta`. All points and queries are unit vectors.
//!
//! | Algorithm | Paper | Filters | Space | Analysis |
//! | --- | --- | --- | --- | --- |
//! | [`Top1`] | Algorithm 2 | `n^{theta / (1 - alpha^2)}` | `O(d max(n, m))` | asymptotic (Theorem 9) |
//! | [`CloseTop1`] | Algorithm 4 | `n^{theta / (1 - alpha^2)}` | `O(d max(n, m))` | every `n` (Lemma 15) |
//! | [`TensorCloseTop1`] | Algorithm 5 | `n^{o(1)}` | `O(d n)` | every `n` (Theorem 19) |
//! | [`TensorTop1`] | — | `n^{o(1)}` | `O(d n)` | none stated |
//!
//! [`TensorCloseTop1`] is the one to use on real data: the single factor
//! algorithms compare every point with `m > n` filters at the default `theta`.
//!
//! All four are the same generic type, [`LsfIndex`], with the algorithm as type
//! parameter, so they share one API:
//!
//! ```
//! use ann_rust::anns::TensorCloseTop1; // or Top1, CloseTop1, TensorTop1
//! use ann_rust::utils::generate_unit_sphere_vectors;
//! use ann_rust::Config;
//!
//! let points = generate_unit_sphere_vectors(1_000, 32, 1);
//! let query = points[0].clone();
//! let config = Config { alpha: 0.9, beta: 0.5, seed: 7, ..Config::default() };
//!
//! let index = TensorCloseTop1::build(points, &config)?;
//! // The query point itself is in the data set, at inner product 1.
//! let outcome = index.query(&query);
//! println!("found {:?} after {} points", outcome.point, outcome.inspected_points);
//! # Ok::<(), String>(())
//! ```
//!
//! To count instead of search, turn an index into a counter with
//! [`LsfCounter::from`](crate::annc::LsfCounter), see [`crate::annc`].

mod close_top1;
mod index;
mod tensor_close_top1;
mod tensor_top1;
mod top1;

pub use close_top1::CloseTop1;
pub use index::{AnnOutcome, LsfIndex, MemoryFootprint};
pub use tensor_close_top1::TensorCloseTop1;
pub use tensor_top1::TensorTop1;
pub use top1::Top1;
