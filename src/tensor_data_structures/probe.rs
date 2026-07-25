//! Bucket keys and the Cartesian product enumerated by a tensorized query.
//!
//! A query first runs `search` on each of the `t` factors, obtaining the candidate
//! filter sets `B_1, ..., B_t`, and then inspects every bucket of
//! `B_1 x ... x B_t` (Algorithm 5, procedure `search`). The product is enumerated
//! lazily by [`ProbeIter`] so that an ANN query can stop at the first close point
//! and so that a pathological query never materializes millions of keys.

use super::close_top1::FilterSet;

/// Identifier of a bucket: one filter index per factor.
pub type BucketKey = Vec<u32>;

/// Runs `search` on every factor, returning the candidate filters `B_1, ..., B_t`.
pub fn candidate_filters<'a, I>(filter_sets: I, query: &[f64]) -> Vec<Vec<u32>>
where
    I: IntoIterator<Item = &'a FilterSet>,
{
    filter_sets
        .into_iter()
        .map(|filters| filters.search(query))
        .collect()
}

/// Number of buckets in the Cartesian product of `levels` (saturating).
pub fn product_size(levels: &[Vec<u32>]) -> usize {
    levels
        .iter()
        .try_fold(1usize, |acc, level| acc.checked_mul(level.len()))
        .unwrap_or(usize::MAX)
}

/// Lazy enumeration of `levels[0] x levels[1] x ... x levels[t-1]`.
///
/// The iterator yields at most `limit` keys; [`ProbeIter::truncated`] reports
/// whether the enumeration was cut short.
pub struct ProbeIter<'a> {
    levels: &'a [Vec<u32>],
    counter: Vec<usize>,
    done: bool,
    emitted: usize,
    limit: usize,
}

impl<'a> ProbeIter<'a> {
    /// Creates an iterator over the product of `levels`, capped at `limit` keys.
    pub fn new(levels: &'a [Vec<u32>], limit: usize) -> Self {
        // An empty factor (or no factor at all) makes the product empty.
        let done = levels.is_empty() || levels.iter().any(|level| level.is_empty());
        ProbeIter {
            levels,
            counter: vec![0; levels.len()],
            done,
            emitted: 0,
            limit,
        }
    }

    /// `true` if the enumeration stopped because it reached the cap.
    pub fn truncated(&self) -> bool {
        !self.done && self.emitted >= self.limit
    }
}

impl Iterator for ProbeIter<'_> {
    type Item = BucketKey;

    fn next(&mut self) -> Option<BucketKey> {
        if self.done || self.emitted >= self.limit {
            return None;
        }
        let key: BucketKey = self
            .levels
            .iter()
            .zip(self.counter.iter())
            .map(|(level, &index)| level[index])
            .collect();

        // Odometer increment over the mixed radix given by the level sizes.
        let mut position = self.levels.len();
        loop {
            if position == 0 {
                self.done = true;
                break;
            }
            position -= 1;
            self.counter[position] += 1;
            if self.counter[position] < self.levels[position].len() {
                break;
            }
            self.counter[position] = 0;
        }

        self.emitted += 1;
        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_enumeration() {
        let levels = vec![vec![0u32, 1], vec![7u32], vec![3u32, 4]];
        let keys: Vec<BucketKey> = ProbeIter::new(&levels, usize::MAX).collect();
        assert_eq!(
            keys,
            vec![
                vec![0, 7, 3],
                vec![0, 7, 4],
                vec![1, 7, 3],
                vec![1, 7, 4],
            ]
        );
        assert_eq!(product_size(&levels), 4);
    }

    #[test]
    fn test_empty_factor_yields_no_bucket() {
        let levels = vec![vec![0u32, 1], Vec::new(), vec![3u32]];
        let keys: Vec<BucketKey> = ProbeIter::new(&levels, usize::MAX).collect();
        assert!(keys.is_empty());
        assert_eq!(product_size(&levels), 0);

        let no_levels: Vec<Vec<u32>> = Vec::new();
        assert_eq!(ProbeIter::new(&no_levels, usize::MAX).count(), 0);
    }

    #[test]
    fn test_limit_is_respected() {
        let levels = vec![vec![0u32, 1, 2], vec![0u32, 1, 2]];
        let mut iterator = ProbeIter::new(&levels, 4);
        let keys: Vec<BucketKey> = iterator.by_ref().collect();
        assert_eq!(keys.len(), 4);
        assert!(iterator.truncated());

        let mut full = ProbeIter::new(&levels, 9);
        assert_eq!(full.by_ref().count(), 9);
        assert!(!full.truncated());
    }

    #[test]
    fn test_single_factor() {
        let levels = vec![vec![5u32, 9]];
        let keys: Vec<BucketKey> = ProbeIter::new(&levels, usize::MAX).collect();
        assert_eq!(keys, vec![vec![5], vec![9]]);
    }
}
