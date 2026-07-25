//! One CloseTop-1 factor (Algorithm 4) of the tensorized data structure.
//!
//! The structure owns `m` Gaussian filters. During construction every point is
//! associated to the **first** filter whose inner product falls inside the collision
//! band `[sqrt(2 log m) - (3/2) log log m / sqrt(2 log m), sqrt(2 log m)]`; a query
//! selects every filter whose inner product with the query is at least `eta`.
//!
//! Unlike Top-1 (which associates a point to the filter maximizing the inner
//! product), CloseTop-1 bounds the inner product from *both* sides by construction,
//! so no assumption on the limiting distribution of the extreme concomitant is
//! needed (Lemma 15 of the paper).

use crate::utils::{collision_band, dot_product, generate_normal_gaussian_vectors_seeded};
use rayon::prelude::*;

/// The data independent part of a factor: its filters and its query threshold.
///
/// This is the function `Q` of Definition 11. It is drawn without looking at the
/// data, so it can be published as is — which is precisely what allows the private
/// counting structure to spend its whole privacy budget on the counters.
#[derive(Debug, Clone)]
pub struct FilterSet {
    /// The `m` filters, drawn from `N(0, 1)^d`.
    pub gaussian_vectors: Vec<Vec<f64>>,
    /// Query threshold `eta`.
    pub eta: f64,
}

impl FilterSet {
    /// Returns the indices of all filters with `<a_i, q> >= eta` (procedure `search`).
    pub fn search(&self, query: &[f64]) -> Vec<u32> {
        self.gaussian_vectors
            .iter()
            .enumerate()
            .filter(|(_, filter)| dot_product(query, filter) >= self.eta)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Number of filters `m`.
    pub fn m(&self) -> usize {
        self.gaussian_vectors.len()
    }
}

/// A single CloseTop-1 factor: `m` Gaussian filters plus the filter assigned to
/// each input point.
pub struct CloseTop1 {
    /// Filters and query threshold; the publishable part of the factor.
    pub filters: FilterSet,
    /// `match_list[i]` is the filter assigned to the `i`-th input point, or `None`
    /// if the point did not collide with any filter and is therefore not stored.
    pub match_list: Vec<Option<u32>>,
    /// Collision band `(lower, upper)` used at construction time.
    pub band: (f64, f64),
}

impl CloseTop1 {
    /// Builds the factor over `data` with `m` filters.
    ///
    /// `fallback_to_argmax` keeps a point that collided with no filter by assigning
    /// it to the filter maximizing the inner product (i.e. the Top-1 rule) instead
    /// of dropping it; see the README for why this matters at finite `n`.
    pub fn build(
        data: &[Vec<f64>],
        m: usize,
        eta: f64,
        fallback_to_argmax: bool,
        seed: u64,
    ) -> Self {
        let d = data.first().map(|point| point.len()).unwrap_or(0);
        let gaussian_vectors = generate_normal_gaussian_vectors_seeded(m, d, seed);
        let band = collision_band(m);
        let match_list = data
            .par_iter()
            .map(|point| assign(point, &gaussian_vectors, band, fallback_to_argmax))
            .collect();
        CloseTop1 {
            filters: FilterSet {
                gaussian_vectors,
                eta,
            },
            match_list,
            band,
        }
    }

    /// Returns the indices of all filters with `<a_i, q> >= eta` (procedure `search`).
    pub fn search(&self, query: &[f64]) -> Vec<u32> {
        self.filters.search(query)
    }

    /// Filter assigned to the `i`-th input point, if the point was stored.
    pub fn bucket_of(&self, i: usize) -> Option<u32> {
        self.match_list[i]
    }

    /// Number of filters `m` held by this factor.
    pub fn m(&self) -> usize {
        self.filters.m()
    }

    /// Number of input points this factor was able to store.
    pub fn stored_points(&self) -> usize {
        self.match_list.iter().filter(|slot| slot.is_some()).count()
    }
}

/// Associates `point` to the first filter inside the collision band.
fn assign(
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
        if fallback_to_argmax && best.map_or(true, |(value, _)| inner_product > value) {
            best = Some((inner_product, i as u32));
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{generate_unit_sphere_vectors, get_threshold};

    /// Every stored point must really sit inside the collision band of its filter.
    #[test]
    fn test_assignment_is_inside_the_band() {
        let m = 256;
        let data = generate_unit_sphere_vectors(500, 32, 1);
        let factor = CloseTop1::build(&data, m, get_threshold(0.9, m), false, 2);
        let (lower, upper) = factor.band;
        let mut stored = 0;
        for (i, point) in data.iter().enumerate() {
            if let Some(filter) = factor.bucket_of(i) {
                let inner_product =
                    dot_product(point, &factor.filters.gaussian_vectors[filter as usize]);
                assert!(inner_product >= lower && inner_product <= upper);
                stored += 1;
            }
        }
        // With m = 256 filters the vast majority of points collides.
        assert!(stored > 0, "no point collided at all");
        assert_eq!(stored, factor.stored_points());
    }

    /// The fallback rule stores every point, using the Top-1 (argmax) filter.
    #[test]
    fn test_fallback_stores_every_point() {
        let m = 64;
        let data = generate_unit_sphere_vectors(300, 16, 3);
        let factor = CloseTop1::build(&data, m, get_threshold(0.9, m), true, 4);
        assert_eq!(factor.stored_points(), data.len());
    }

    /// `search` returns exactly the filters above the threshold.
    #[test]
    fn test_search_matches_brute_force() {
        let m = 128;
        let data = generate_unit_sphere_vectors(100, 16, 5);
        let eta = get_threshold(0.9, m);
        let factor = CloseTop1::build(&data, m, eta, true, 6);
        let query = &data[0];
        let expected: Vec<u32> = factor
            .filters
            .gaussian_vectors
            .iter()
            .enumerate()
            .filter(|(_, filter)| dot_product(query, filter) >= eta)
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(factor.search(query), expected);
    }
}
