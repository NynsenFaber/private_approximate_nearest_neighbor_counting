//! The four algorithms, as marker types.
//!
//! Top-1, CloseTop-1, TensorCloseTop-1 and TensorTop-1 share one implementation.
//! They differ in two choices only:
//!
//! | Marker | Paper | Assignment rule | Factors |
//! | --- | --- | --- | --- |
//! | [`Top1`] | Algorithm 2 | argmax filter | 1 |
//! | [`CloseTop1`] | Algorithm 4 | first filter in the collision band | 1 |
//! | [`TensorCloseTop1`] | Algorithm 5 | first filter in the collision band | `t` |
//! | [`TensorTop1`] | not in the paper | argmax filter | `t` |
//!
//! A marker is the type parameter of the generic structures
//! [`LsfIndex`](crate::anns::LsfIndex), [`LsfCounter`](crate::annc::LsfCounter) and
//! [`DpLsfCounter`](crate::annc::dp::DpLsfCounter). Most code never names a marker:
//! it uses the aliases such as [`crate::anns::TensorCloseTop1`]. Generic code, like
//! the experiments, takes `A: Algorithm` to run any of the four.

/// How a point is assigned to one filter of a factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assignment {
    /// The filter maximizing `<a_i, x>` (Algorithm 2 line 6). Every point is stored.
    Top1,
    /// The first filter whose `<a_i, x>` lies in the collision band (Algorithm 4
    /// lines 6-9). A point that no filter accepts is dropped, unless
    /// [`Config::fallback_to_argmax`](crate::Config::fallback_to_argmax) is set.
    CloseTop1,
}

/// Compile-time description of one of the four algorithms.
pub trait Algorithm {
    /// Name printed by the experiments, e.g. `"TensorCloseTop1"`.
    const NAME: &'static str;
    /// Rule used by each factor to assign a point to a filter.
    const ASSIGNMENT: Assignment;
    /// `true` for the tensorized algorithms (`t` factors), `false` for one factor.
    const TENSORIZED: bool;
}

/// Marker for Top-1 (Algorithm 2). See [`crate::anns::Top1`].
#[derive(Debug, Clone, Copy)]
pub struct Top1;

/// Marker for CloseTop-1 (Algorithm 4). See [`crate::anns::CloseTop1`].
#[derive(Debug, Clone, Copy)]
pub struct CloseTop1;

/// Marker for TensorCloseTop-1 (Algorithm 5). See [`crate::anns::TensorCloseTop1`].
#[derive(Debug, Clone, Copy)]
pub struct TensorCloseTop1;

/// Marker for TensorTop-1. See [`crate::anns::TensorTop1`].
#[derive(Debug, Clone, Copy)]
pub struct TensorTop1;

impl Algorithm for Top1 {
    const NAME: &'static str = "Top1";
    const ASSIGNMENT: Assignment = Assignment::Top1;
    const TENSORIZED: bool = false;
}

impl Algorithm for CloseTop1 {
    const NAME: &'static str = "CloseTop1";
    const ASSIGNMENT: Assignment = Assignment::CloseTop1;
    const TENSORIZED: bool = false;
}

impl Algorithm for TensorCloseTop1 {
    const NAME: &'static str = "TensorCloseTop1";
    const ASSIGNMENT: Assignment = Assignment::CloseTop1;
    const TENSORIZED: bool = true;
}

impl Algorithm for TensorTop1 {
    const NAME: &'static str = "TensorTop1";
    const ASSIGNMENT: Assignment = Assignment::Top1;
    const TENSORIZED: bool = true;
}
