//! Locality sensitive filtering core shared by every algorithm in the crate.
//!
//! All four algorithms build a *space partitioning data structure* (Definition 11):
//! a data independent function `Q` — Gaussian filters and a query threshold — and a
//! partition of the data set into lists. The search structures in [`crate::anns`]
//! keep the lists, the counting structures in [`crate::annc`] keep only their
//! sizes, and the private ones in [`crate::annc::dp`] keep noisy sizes.
//!
//! * [`algorithms`] — the four algorithms as marker types, and what tells them apart.
//! * [`Config`], [`Parameters`] — user settings and the values derived from them.
//! * [`input`] — the checks every data set and query goes through.
//! * [`filters`] — Gaussian filters, `search`, and the Top-1 / CloseTop-1 rules.
//! * [`partition`] — the construction shared by all four algorithms.
//! * [`bucket_index`], [`probe`] — how a query visits the buckets it selects.

pub mod algorithms;
pub mod bucket_index;
mod config;
pub mod filters;
pub mod input;
pub mod partition;
pub mod probe;

pub use algorithms::Algorithm;
pub use config::{Config, Parameters};
