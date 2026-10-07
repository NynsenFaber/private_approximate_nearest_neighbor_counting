//! Gaussian filters, the query side `search`, and the two assignment rules.
//!
//! A *factor* is a set of `m` Gaussian filters. At construction, an assignment
//! rule sends every point to (at most) one filter of each factor; at query time,
//! [`FilterSet::search`] selects every filter with `<a_i, q> >= eta`.
//!
//! * [`assign_top1`] — the argmax filter (Algorithm 2 line 6).
//! * [`assign_close_top1`] — the first filter in the collision band (Algorithm 4
//!   lines 6-9).
//!
//! # The collision probability at finite `m`
//!
//! Lemma 23 states that a point fails to collide with probability `m^{-Omega(1)}`,
//! via Lemma 24's bound `Pr[<a, x> in band] >= (2 sqrt(pi)/3) log m / m`.
//!
//! That constant is optimistic. Lemma 24 is derived from Proposition 22, whose
//! printed form carries `sqrt(2 pi)` in the numerator where the Gaussian tail bound
//! it cites has it in the denominator — a factor `2 pi` too large, which makes the
//! printed *lower* bound larger than the quantity it bounds. At `m = 502` (the
//! default `m_sub` for `n = 10^5`) the printed Lemma 24 bound is `0.0146` while the
//! true band probability is `0.00278`. The asymptotic form `Omega(log m / m)`, and
//! therefore every downstream result, is unaffected — only the constant moves.
//!
//! The practical consequence is real, though: at `m = 502` a single filter accepts a
//! point with probability `0.00278`, so a factor stores only `1 - (1 - p)^m = 75.2%`
//! of the points and `t = 3` factors store `42.6%`. That is what
//! [`Config::fallback_to_argmax`](crate::Config::fallback_to_argmax) exists to repair.

use crate::utils::{dot_product, generate_normal_gaussian_vectors_seeded};

/// One factor's filters and its query threshold.
///
/// Together with the other factors this is the function `Q` of Definition 11. It is
/// drawn without looking at the data, so it can be published as is — which is
/// what lets the private counting structures spend their whole privacy budget on
/// the counters.
#[derive(Debug, Clone)]
pub struct FilterSet {
    /// The `m` filters, drawn from `N(0, 1)^d`.
    pub gaussian_vectors: Vec<Vec<f64>>,
    /// Query threshold `eta`.
    pub eta: f64,
}

impl FilterSet {
    /// Samples `m` filters in dimension `d`, reproducibly from `seed`.
    pub fn sample(m: usize, d: usize, eta: f64, seed: u64) -> Self {
        FilterSet {
            gaussian_vectors: generate_normal_gaussian_vectors_seeded(m, d, seed),
            eta,
        }
    }

    /// Indices of the filters a query must open: all `i` with `<a_i, q> >= eta`
    /// (Algorithm 2 `search`, Algorithm 5 `search` line 3).
    ///
    /// The result is ascending, which [`super::bucket_index::BucketIndex`] relies
    /// on to intersect it against the sorted bucket keys by merging.
    pub fn search(&self, query: &[f64]) -> Vec<u32> {
        self.gaussian_vectors
            .iter()
            .enumerate()
            .filter(|(_, filter)| dot_product(query, filter) >= self.eta)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// Top-1 rule: the filter maximizing `<a_i, x>` (Algorithm 2 line 6). Ties go to the
/// lowest index.
pub fn assign_top1(point: &[f64], gaussian_vectors: &[Vec<f64>]) -> u32 {
    let mut best: Option<(f64, u32)> = None;
    for (i, filter) in gaussian_vectors.iter().enumerate() {
        let inner_product = dot_product(point, filter);
        if best.is_none_or(|(value, _)| inner_product > value) {
            best = Some((inner_product, i as u32));
        }
    }
    best.expect("a factor has at least one filter").1
}

/// CloseTop-1 rule: the **first** filter whose inner product falls inside the
/// collision band (Algorithm 4 lines 5-9 — the `break` is the early `return`).
///
/// "First", not "best": that is the whole difference from Top-1. Because the band
/// bounds `<a, x>` from *both* sides, the analysis needs no assumption about the
/// limiting distribution of the maximum (Lemma 15).
///
/// With `fallback_to_argmax` a point that matched no filter is kept at its Top-1
/// (argmax) filter instead of being dropped. This is *not* in Algorithm 4. It
/// cannot break anything downstream: the point still lands in exactly one bucket,
/// which is all the sensitivity-1 argument of Theorem 13 needs.
pub fn assign_close_top1(
    point: &[f64],
    gaussian_vectors: &[Vec<f64>],
    band: (f64, f64),
    fallback_to_argmax: bool,
) -> Option<u32> {
    let (lower, upper) = band;
    let mut best: Option<(f64, u32)> = None;
    for (i, filter) in gaussian_vectors.iter().enumerate() {
        let inner_product = dot_product(point, filter);
        if inner_product >= lower && inner_product <= upper {
            return Some(i as u32);
        }
        // Only reached when no filter has matched yet, so on exit `best` is the
        // argmax over all `m` filters — the same filter `assign_top1` picks.
        if fallback_to_argmax && best.is_none_or(|(value, _)| inner_product > value) {
            best = Some((inner_product, i as u32));
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{collision_band, generate_unit_sphere_vectors, get_threshold};

    fn filters(m: usize, d: usize, seed: u64) -> FilterSet {
        FilterSet::sample(m, d, get_threshold(0.9, m), seed)
    }

    /// The Top-1 rule returns the argmax filter.
    #[test]
    fn test_top1_returns_the_argmax() {
        let set = filters(64, 16, 1);
        for point in generate_unit_sphere_vectors(100, 16, 2) {
            let chosen = assign_top1(&point, &set.gaussian_vectors);
            let best = set
                .gaussian_vectors
                .iter()
                .map(|filter| dot_product(&point, filter))
                .fold(f64::NEG_INFINITY, f64::max);
            assert_eq!(
                dot_product(&point, &set.gaussian_vectors[chosen as usize]),
                best
            );
        }
    }

    /// Every point the CloseTop-1 rule stores sits inside the band of its filter.
    #[test]
    fn test_close_top1_assignment_is_inside_the_band() {
        let m = 256;
        let set = filters(m, 32, 2);
        let band = collision_band(m);
        let mut stored = 0;
        for point in generate_unit_sphere_vectors(500, 32, 1) {
            if let Some(filter) = assign_close_top1(&point, &set.gaussian_vectors, band, false) {
                let inner_product = dot_product(&point, &set.gaussian_vectors[filter as usize]);
                assert!(inner_product >= band.0 && inner_product <= band.1);
                stored += 1;
            }
        }
        assert!(stored > 0, "no point collided at all");
    }

    /// With the fallback, a point no filter accepts goes to its argmax filter, so
    /// every point is stored.
    #[test]
    fn test_close_top1_fallback_is_the_top1_filter() {
        let m = 64;
        let set = filters(m, 16, 4);
        let band = collision_band(m);
        let mut fell_back = 0;
        for point in generate_unit_sphere_vectors(300, 16, 3) {
            let strict = assign_close_top1(&point, &set.gaussian_vectors, band, false);
            let lenient = assign_close_top1(&point, &set.gaussian_vectors, band, true);
            match strict {
                Some(filter) => assert_eq!(lenient, Some(filter)),
                None => {
                    assert_eq!(lenient, Some(assign_top1(&point, &set.gaussian_vectors)));
                    fell_back += 1;
                }
            }
        }
        assert!(fell_back > 0, "the test never exercised the fallback");
    }

    /// `search` returns exactly the filters above the threshold, in ascending order.
    #[test]
    fn test_search_matches_brute_force() {
        let set = filters(128, 16, 6);
        let query = &generate_unit_sphere_vectors(1, 16, 5)[0];
        let expected: Vec<u32> = set
            .gaussian_vectors
            .iter()
            .enumerate()
            .filter(|(_, filter)| dot_product(query, filter) >= set.eta)
            .map(|(i, _)| i as u32)
            .collect();
        let found = set.search(query);
        assert_eq!(found, expected);
        assert!(found.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
