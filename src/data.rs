//! Synthetic data sets on the unit sphere, plus the brute force ground truth used
//! by the experiments.
//!
//! The background of every data set consists of vectors drawn uniformly at random
//! from `S^{d-1}`. Such vectors are almost orthogonal in high dimension, so a random
//! query has no near neighbour at all and both experiments would be vacuous.
//! Each query therefore comes with *planted* neighbours at a prescribed inner
//! product from it (see [`PlantConfig`]).

use crate::utils::{
    dot_product, generate_unit_sphere_vectors, normalize_vector, random_unit_vector,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use savefile::prelude::*;
use savefile_derive::Savefile;
use std::io::{Error, ErrorKind};

/// Points and queries of a synthetic benchmark.
#[derive(Savefile)]
pub struct SyntheticDataset {
    /// The data set `S`, including the planted neighbours.
    pub points: Vec<Vec<f64>>,
    /// The query vectors; `queries[i]` owns the `i`-th planted group.
    pub queries: Vec<Vec<f64>>,
}

/// How to plant near neighbours around a query.
#[derive(Debug, Clone)]
pub struct PlantConfig {
    /// Number of planted points.
    pub count: usize,
    /// Inner product of each planted point with the query (at least `alpha`).
    pub similarity: f64,
    /// Mutual similarity of the planted points. `0.0` plants them independently
    /// around the query, a value close to `1.0` plants a tight cluster, which is
    /// the interesting regime for counting: the near neighbours then share a few
    /// buckets and their counters survive the privacy threshold.
    pub tightness: f64,
}

/// Configuration of [`generate`].
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Total number of points, planted neighbours included.
    pub n: usize,
    /// Dimension.
    pub d: usize,
    /// Number of queries.
    pub queries: usize,
    /// How the neighbours of each query are planted.
    pub plant: PlantConfig,
    /// Master seed.
    pub seed: u64,
}

/// Samples a unit vector at inner product exactly `similarity` from `anchor`.
///
/// Writes `x = s * anchor + sqrt(1 - s^2) * u` with `u` a random unit vector
/// orthogonal to `anchor`.
pub fn point_at_similarity<R: Rng + ?Sized>(
    anchor: &[f64],
    similarity: f64,
    rng: &mut R,
) -> Vec<f64> {
    let d = anchor.len();
    let mut orthogonal = loop {
        let mut candidate = random_unit_vector(d, rng);
        let projection = dot_product(&candidate, anchor);
        for (value, anchor_value) in candidate.iter_mut().zip(anchor.iter()) {
            *value -= projection * anchor_value;
        }
        let norm = candidate.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Degenerate only if the sample was (numerically) parallel to the anchor.
        if norm > 1e-9 {
            normalize_vector(&mut candidate);
            break candidate;
        }
    };
    let orthogonal_weight = (1. - similarity * similarity).max(0.).sqrt();
    for (value, anchor_value) in orthogonal.iter_mut().zip(anchor.iter()) {
        *value = similarity * anchor_value + orthogonal_weight * *value;
    }
    normalize_vector(&mut orthogonal);
    orthogonal
}

/// Margin added to the requested similarity when a point is planted exactly at the
/// threshold, so that round-off never pushes it below `alpha` and out of the ground
/// truth. It is far larger than the `~1e-16` error of a dot product and far smaller
/// than any similarity difference that matters.
const PLANT_MARGIN: f64 = 1e-9;

/// Plants `config.count` near neighbours around `query`.
///
/// All returned points have inner product at least `config.similarity` with the
/// query.
pub fn plant_neighbours<R: Rng + ?Sized>(
    query: &[f64],
    config: &PlantConfig,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let similarity = (config.similarity + PLANT_MARGIN).min(1.0);
    if config.tightness <= 0.0 {
        return (0..config.count)
            .map(|_| point_at_similarity(query, similarity, rng))
            .collect();
    }

    // A cluster centre strictly closer than `similarity`, so that spreading the
    // members around it keeps them above the threshold.
    let centre_similarity = (similarity + 1.) / 2.;
    let centre = point_at_similarity(query, centre_similarity, rng);
    (0..config.count)
        .map(|_| {
            for _ in 0..64 {
                let member = point_at_similarity(&centre, config.tightness, rng);
                if dot_product(query, &member) >= similarity {
                    return member;
                }
            }
            // Fall back to a point right at the threshold rather than returning one
            // that would sit outside the ground truth.
            point_at_similarity(query, similarity, rng)
        })
        .collect()
}

/// Generates a synthetic data set with planted near neighbours.
pub fn generate(config: &GeneratorConfig) -> Result<SyntheticDataset, String> {
    let planted = config.queries * config.plant.count;
    if planted > config.n {
        return Err(format!(
            "cannot plant {planted} neighbours in a data set of {} points",
            config.n
        ));
    }
    if !(config.plant.similarity > 0. && config.plant.similarity < 1.) {
        return Err(format!(
            "the planted similarity must lie in (0, 1), got {}",
            config.plant.similarity
        ));
    }

    let background = config.n - planted;
    let mut points = generate_unit_sphere_vectors(background, config.d, config.seed);
    let mut queries = Vec::with_capacity(config.queries);
    for q in 0..config.queries {
        let mut rng =
            StdRng::seed_from_u64(crate::utils::derive_seed(config.seed, 1 << 40 | q as u64));
        let query = random_unit_vector(config.d, &mut rng);
        points.extend(plant_neighbours(&query, &config.plant, &mut rng));
        queries.push(query);
    }

    // Planted points are appended after the background, so without this shuffle the
    // storage order would encode which points are the answers. Nothing in the LSF
    // structure cares (a point lands in its bucket regardless of its index), but any
    // baseline that scans in storage order would be measured against a worst case
    // layout, so the order is randomized once here rather than at every use site.
    let mut rng = StdRng::seed_from_u64(crate::utils::derive_seed(config.seed, 1 << 41));
    points.shuffle(&mut rng);

    Ok(SyntheticDataset { points, queries })
}

/// Exact number of points at inner product at least `threshold` from `query`.
pub fn exact_count(points: &[Vec<f64>], query: &[f64], threshold: f64) -> u64 {
    points
        .par_iter()
        .filter(|point| dot_product(query, point) >= threshold)
        .count() as u64
}

/// Largest inner product between `query` and the data set.
pub fn best_similarity(points: &[Vec<f64>], query: &[f64]) -> f64 {
    points
        .par_iter()
        .map(|point| dot_product(query, point))
        .reduce(|| f64::NEG_INFINITY, f64::max)
}

/// Naive `(alpha, beta)`-ANN baseline: a single threaded scan of `points` in
/// storage order, returning the first one at inner product at least `beta` from
/// `query`. Unlike the LSF structure this is exact — it finds a match whenever one
/// exists — so it is the accuracy baseline that TensorCloseTop1 trades against
/// query time.
pub fn linear_search_first(points: &[Vec<f64>], query: &[f64], beta: f64) -> Option<usize> {
    points
        .iter()
        .position(|point| dot_product(query, point) >= beta)
}

/// Saves a data set to `path` in savefile's binary format.
pub fn save(path: &str, dataset: &SyntheticDataset) -> std::io::Result<()> {
    save_file(path, 0, dataset).map_err(|e| Error::other(format!("failed to save {path}: {e}")))
}

/// Loads a data set previously written by [`save`].
pub fn load(path: &str) -> std::io::Result<SyntheticDataset> {
    load_file(path, 0)
        .map_err(|e| Error::new(ErrorKind::NotFound, format!("failed to load {path}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::is_normalized;

    /// A planted point must be normalized and at the requested inner product.
    #[test]
    fn test_point_at_similarity() {
        let mut rng = StdRng::seed_from_u64(1);
        let anchor = random_unit_vector(32, &mut rng);
        for similarity in [0.0, 0.3, 0.9, 0.99] {
            let point = point_at_similarity(&anchor, similarity, &mut rng);
            assert!(is_normalized(&point));
            assert!((dot_product(&anchor, &point) - similarity).abs() < 1e-9);
        }
    }

    /// Every planted neighbour respects the requested threshold, in both the
    /// independent and the clustered regime.
    #[test]
    fn test_plant_neighbours() {
        let mut rng = StdRng::seed_from_u64(2);
        let query = random_unit_vector(64, &mut rng);
        for tightness in [0.0, 0.99, 0.9999] {
            let config = PlantConfig {
                count: 50,
                similarity: 0.9,
                tightness,
            };
            let neighbours = plant_neighbours(&query, &config, &mut rng);
            assert_eq!(neighbours.len(), 50);
            for neighbour in &neighbours {
                assert!(is_normalized(neighbour));
                // Strictly above the threshold, so the brute force ground truth
                // (which uses a plain `>=`) counts every planted point.
                assert!(dot_product(&query, neighbour) >= 0.9);
            }
            assert_eq!(exact_count(&neighbours, &query, 0.9), 50);
            if tightness > 0.0 {
                // Members of a tight cluster are close to each other as well.
                let mutual = dot_product(&neighbours[0], &neighbours[1]);
                assert!(mutual > 0.5, "cluster is not tight: {mutual}");
            }
        }
    }

    /// The generator produces the requested sizes and a correct ground truth.
    #[test]
    fn test_generate_and_ground_truth() {
        let config = GeneratorConfig {
            n: 1000,
            d: 32,
            queries: 4,
            plant: PlantConfig {
                count: 25,
                similarity: 0.9,
                tightness: 0.999,
            },
            seed: 3,
        };
        let dataset = generate(&config).unwrap();
        assert_eq!(dataset.points.len(), 1000);
        assert_eq!(dataset.queries.len(), 4);
        for query in &dataset.queries {
            // The 25 planted points are found by the brute force ground truth, and
            // uniform background points essentially never reach 0.9 in dimension 32.
            assert_eq!(exact_count(&dataset.points, query, 0.9), 25);
            assert!(best_similarity(&dataset.points, query) >= 0.9);
        }
        assert!(generate(&GeneratorConfig { n: 10, ..config }).is_err());
    }

    /// A data set survives a save/load round trip unchanged, and a missing file is
    /// an error.
    #[test]
    fn test_save_and_load() {
        let config = GeneratorConfig {
            n: 50,
            d: 8,
            queries: 2,
            plant: PlantConfig {
                count: 3,
                similarity: 0.9,
                tightness: 0.0,
            },
            seed: 4,
        };
        let dataset = generate(&config).unwrap();
        let path = std::env::temp_dir().join(format!("ann_rust_test_{}.bin", std::process::id()));
        let path = path.to_str().unwrap();
        save(path, &dataset).unwrap();
        let loaded = load(path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded.points, dataset.points);
        assert_eq!(loaded.queries, dataset.queries);
        assert!(load(path).is_err());

        let bad = GeneratorConfig {
            plant: PlantConfig {
                similarity: 1.0,
                ..config.plant
            },
            ..config
        };
        assert!(generate(&bad).is_err());
    }

    /// The baseline returns the first point at inner product >= beta, in order.
    #[test]
    fn test_linear_search_first() {
        let points = vec![vec![0.0, 1.0], vec![0.6, 0.8], vec![1.0, 0.0]];
        assert_eq!(linear_search_first(&points, &[1.0, 0.0], 0.5), Some(1));
        assert_eq!(linear_search_first(&points, &[-1.0, 0.0], 0.5), None);
    }

    /// The planted points must be spread through the data set, not parked at the
    /// end: a baseline that scans in storage order would otherwise be timed against
    /// a worst case layout rather than a representative one.
    #[test]
    fn test_planted_points_are_not_all_at_the_end() {
        let config = GeneratorConfig {
            n: 2000,
            d: 32,
            queries: 1,
            plant: PlantConfig {
                count: 40,
                similarity: 0.9,
                tightness: 0.0,
            },
            seed: 17,
        };
        let dataset = generate(&config).unwrap();
        let query = &dataset.queries[0];
        let positions: Vec<usize> = dataset
            .points
            .iter()
            .enumerate()
            .filter(|(_, point)| dot_product(query, point) >= 0.9)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions.len(), 40);
        // Without the shuffle every position would be >= n - 40 = 1960.
        let first = positions[0];
        assert!(
            first < 1960,
            "planted points still start at index {first}, i.e. only at the tail"
        );
        // The mean position should sit near the middle of the data set, not the end.
        let mean = positions.iter().sum::<usize>() as f64 / positions.len() as f64;
        assert!(
            (mean - 1000.).abs() < 400.,
            "planted points are not spread out, mean position {mean}"
        );
    }
}
