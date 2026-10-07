//! Small fixtures shared by the unit tests of the twelve structures.

use crate::data::{plant_neighbours, PlantConfig};
use crate::lsf::Algorithm;
use crate::utils::{generate_unit_sphere_vectors, random_unit_vector};
use crate::Config;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Dimension of every test data set.
pub const DIMENSION: usize = 16;

/// A small configuration for algorithm `A`: 64 filters per factor, and 3 factors
/// for the tensorized algorithms.
pub fn config<A: Algorithm>(seed: u64) -> Config {
    Config {
        alpha: 0.9,
        beta: 0.5,
        t: A::TENSORIZED.then_some(3),
        m_sub: Some(64),
        seed,
        ..Config::default()
    }
}

/// Every coordinate times 4: off the unit sphere, and exactly normalizable back.
pub fn scaled_by_4(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|point| point.iter().map(|x| 4.0 * x).collect())
        .collect()
}

/// `background` random unit vectors plus `planted` points at inner product
/// `>= 0.95` from the returned query, with mutual similarity `tightness`.
pub fn planted_dataset(
    background: usize,
    planted: usize,
    tightness: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let query = random_unit_vector(DIMENSION, &mut rng);
    let mut data = generate_unit_sphere_vectors(background, DIMENSION, seed);
    data.extend(plant_neighbours(
        &query,
        &PlantConfig {
            count: planted,
            similarity: 0.95,
            tightness,
        },
        &mut rng,
    ));
    (data, query)
}
