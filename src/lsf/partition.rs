//! The construction shared by the four algorithms: a space partitioning data
//! structure in the sense of Definition 11.
//!
//! [`Partition::build`] samples `t` independent factors of `m_sub` filters, assigns
//! every point to one filter per factor with the algorithm's rule, and stores the
//! point in the bucket `(i_1, ..., i_t)` — its *bucket key*. With `t = 1` this is
//! Top-1 / CloseTop-1, with `t > 1` the tensorized versions, where `t * m_sub`
//! stored filters simulate `m_sub^t` buckets.
//!
//! Every point lands in **at most one** bucket. That is what gives `O(d n)` space
//! and, for the counting structures, sensitivity `1`.

use super::algorithms::{Algorithm, Assignment};
use super::bucket_index::BucketIndex;
use super::filters::{assign_close_top1, assign_top1, FilterSet};
use super::probe::{candidate_filters, BucketKey};
use super::{Config, Parameters};
use crate::utils::derive_seed;
use rayon::prelude::*;
use std::collections::HashMap;

/// Filters (the function `Q`) plus the lists `L` of point ids, one per non-empty bucket.
pub struct Partition {
    /// Resolved parameters.
    pub params: Parameters,
    /// The `t` factors; `filters[i]` is factor `i`.
    pub filters: Vec<FilterSet>,
    /// Ids of the points stored in each non-empty bucket, ascending.
    pub buckets: BucketIndex<Vec<u32>>,
    /// Number of points stored in some bucket.
    pub stored_points: usize,
}

impl Partition {
    /// Builds the partition of `data` for algorithm `A`.
    ///
    /// Expects unit vectors of one dimension, as produced by
    /// [`super::input::check_points`]; the `build` functions of the public
    /// structures run that check first.
    pub fn build<A: Algorithm>(data: &[Vec<f64>], config: &Config) -> Result<Self, String> {
        let n = data.len();
        let d = data.first().map_or(0, |point| point.len());
        let params = Parameters::resolve::<A>(config, n, d)?;
        if data.iter().any(|point| point.len() != d) {
            return Err("all points must have the same dimension".to_string());
        }

        // The t factors are independent: each one gets its own filter seed.
        let mut filters = Vec::with_capacity(params.t);
        let mut assignments: Vec<Vec<Option<u32>>> = Vec::with_capacity(params.t);
        for factor in 0..params.t {
            let set = FilterSet::sample(
                params.m_sub,
                d,
                params.eta,
                derive_seed(config.seed, factor as u64 + 1),
            );
            let assigned = data
                .par_iter()
                .map(|point| match params.assignment {
                    Assignment::Top1 => Some(assign_top1(point, &set.gaussian_vectors)),
                    Assignment::CloseTop1 => assign_close_top1(
                        point,
                        &set.gaussian_vectors,
                        params.band,
                        params.fallback_to_argmax,
                    ),
                })
                .collect();
            filters.push(set);
            assignments.push(assigned);
        }

        // Bucket key of every point: its t filter indices. A point missing in any
        // factor is not stored at all (Definition 11 allows the partition to leave
        // points out).
        let keys: Vec<Option<BucketKey>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut key = BucketKey::with_capacity(params.t);
                for factor in &assignments {
                    key.push(factor[i]?);
                }
                Some(key)
            })
            .collect();

        let mut lists: HashMap<BucketKey, Vec<u32>> = HashMap::new();
        let mut stored_points = 0usize;
        for (i, key) in keys.into_iter().enumerate() {
            if let Some(key) = key {
                lists.entry(key).or_default().push(i as u32);
                stored_points += 1;
            }
        }

        Ok(Partition {
            params,
            filters,
            buckets: BucketIndex::from_map(lists),
            stored_points,
        })
    }

    /// The candidate filters `B_1, ..., B_t` selected by `query` in each factor.
    pub fn candidate_filters(&self, query: &[f64]) -> Vec<Vec<u32>> {
        candidate_filters(self.filters.iter(), query)
    }
}
