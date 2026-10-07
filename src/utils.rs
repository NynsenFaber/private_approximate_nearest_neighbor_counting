//! Shared numeric helpers: inner products, sphere/Gaussian sampling, the
//! thresholds of Algorithms 4 and 5, and the normal tail used for reporting.
//!
//! Everything random here is *seeded*: a value derived from the caller's master
//! seed through [`derive_seed`] drives each independent draw, so results do not
//! depend on how rayon happens to schedule the work.

use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rayon::prelude::*;

/// Inner product of two equal-length vectors.
///
/// For unit vectors this is the cosine similarity, which is the only notion of
/// "close" used in this crate. Extra trailing entries of the longer slice are
/// ignored, so callers are responsible for passing matching dimensions —
/// [`crate::lsf::partition::Partition::build`]
/// checks this once, up front.
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

/// Derives an independent stream seed from a master seed and a stream index.
///
/// Used to make every parallel/independent random draw of the library
/// reproducible from a single user supplied seed.
pub fn derive_seed(seed: u64, stream: u64) -> u64 {
    // Splitmix64 finalizer, a cheap way to decorrelate nearby seeds.
    let mut z = seed
        .wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `true` if `vector` lies on the unit sphere up to a `1e-6` tolerance.
pub fn is_normalized(vector: &[f64]) -> bool {
    let norm = vector.iter().map(|x| x * x).sum::<f64>();
    (norm - 1.0).abs() <= 1e-6
}

/// Scales `vector` in place to unit length.
///
/// A zero vector would divide by zero; every caller here samples from a
/// continuous distribution, where that has probability zero.
pub fn normalize_vector(vector: &mut [f64]) {
    let norm: f64 = vector.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    for value in vector.iter_mut() {
        *value /= norm;
    }
}

/// Query threshold `eta = alpha * sqrt(2 log m) - sqrt(2 (1 - alpha^2) log log m)`
/// (Algorithm 4 line 10, Algorithm 5 line 9).
///
/// A filter (Gaussian vector) `a` is inspected by a query `q` if `<a, q> >= eta`.
/// The value follows from the theory of concomitant order statistics: a point at
/// inner product `alpha` from `q` is associated to a filter whose inner product with
/// `q` is distributed as `N(alpha * sqrt(2 log m), 1 - alpha^2)`, and `eta` sits
/// `sqrt(2 log log m)` standard deviations below that mean.
///
/// In the tensorized structure `m` is the *per factor* count `m_sub`, not the
/// `m_sub^t` simulated total: Algorithm 5 line 9 computes `eta` from `m̃`.
pub fn get_threshold(alpha: f64, m: usize) -> f64 {
    let ln_m = (m as f64).ln();
    let ln_ln_m = ln_m.ln();
    alpha * (2. * ln_m).sqrt() - (2. * (1. - alpha.powi(2)) * ln_ln_m).sqrt()
}

/// Collision band of CloseTop-1 (Algorithm 4, line 7).
///
/// A point `x` is associated to the *first* Gaussian vector `a` such that
/// `<a, x>` lies in `[sqrt(2 log m) - (3/2) log log m / sqrt(2 log m), sqrt(2 log m)]`.
/// Bounding `<a, x>` from both sides replaces the asymptotic argument of Top-1
/// (where the maximum concentrates around `sqrt(2 log m)`) by a guarantee that
/// holds for every finite `m`.
pub fn collision_band(m: usize) -> (f64, f64) {
    let ln_m = (m as f64).ln();
    let upper = (2. * ln_m).sqrt();
    let lower = upper - 1.5 * ln_m.ln() / upper;
    (lower, upper)
}

/// Upper tail `Pr[Z >= x]` of a standard normal random variable.
///
/// Only used to report the expected number of inspected filters/buckets, never on
/// the critical path of the data structure.
pub fn normal_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical Recipes `erfcc`, relative error < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1. / (1. + 0.5 * z);
    let poly = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let ans = t * poly.exp();
    if x >= 0. {
        ans
    } else {
        2. - ans
    }
}

/// Samples a vector uniformly at random from the unit sphere `S^{d-1}`.
pub fn random_unit_vector<R: Rng + ?Sized>(d: usize, rng: &mut R) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).expect("N(0, 1) is a valid distribution");
    let mut vector: Vec<f64> = (0..d).map(|_| normal.sample(rng)).collect();
    normalize_vector(&mut vector);
    vector
}

/// Generates `n` vectors distributed uniformly on the unit sphere `S^{d-1}`.
///
/// Reproducible for a given `seed` and independent of the number of threads.
pub fn generate_unit_sphere_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(derive_seed(seed, i as u64));
            random_unit_vector(d, &mut rng)
        })
        .collect()
}

/// Generates `m` vectors with i.i.d. `N(0, 1)` entries, reproducibly from `seed`.
pub fn generate_normal_gaussian_vectors_seeded(m: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    (0..m)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(derive_seed(seed, i as u64));
            let normal = Normal::new(0.0, 1.0).expect("N(0, 1) is a valid distribution");
            (0..d).map(|_| normal.sample(&mut rng)).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inner product must agree with the textbook definition.
    #[test]
    fn test_dot_product() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = dot_product(&vec1, &vec2);
        assert_eq!(result, 32.0);

        let vec1 = vec![0.5, 0.5, 0.];
        let vec2 = vec![0.5, 0.5, 0.];
        let result = dot_product(&vec1, &vec2);
        assert_eq!(result, 0.5);
    }

    /// Normalizing must produce a unit vector.
    #[test]
    fn test_normalize_vector() {
        let mut vector = vec![1.0, 2.0, 3.0];
        normalize_vector(&mut vector);
        let norm: f64 = vector.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= 1e-6);

        let mut vector = vec![0.5, 0.5, 0.];
        normalize_vector(&mut vector);
        let norm: f64 = vector.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= 1e-6);
    }

    /// The tail probability is checked against a few textbook values.
    #[test]
    fn test_normal_sf() {
        assert!((normal_sf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_sf(1.0) - 0.158655).abs() < 1e-5);
        assert!((normal_sf(1.96) - 0.025).abs() < 1e-4);
        assert!((normal_sf(-1.0) - 0.841345).abs() < 1e-5);
        assert!((normal_sf(3.0) - 0.001350).abs() < 1e-5);
    }

    /// The collision band must be exactly Algorithm 4 line 7:
    /// `[sqrt(2 log m) - (3/2) log log m / sqrt(2 log m), sqrt(2 log m)]`.
    #[test]
    fn test_collision_band() {
        for m in [16usize, 100, 10_000, 1_000_000] {
            let (lower, upper) = collision_band(m);
            let ln_m = (m as f64).ln();
            let expected_upper = (2. * ln_m).sqrt();
            let expected_lower = expected_upper - 1.5 * ln_m.ln() / expected_upper;
            assert!((upper - expected_upper).abs() < 1e-12, "m = {m}");
            assert!((lower - expected_lower).abs() < 1e-12, "m = {m}");
            assert!(lower < upper, "empty band for m = {m}");
            assert!(lower > 0.0);
        }
    }

    /// The query threshold must be exactly Algorithm 5 line 9:
    /// `eta = alpha sqrt(2 log m) - sqrt(2 (1 - alpha^2) log log m)`.
    #[test]
    fn test_query_threshold_matches_the_paper() {
        for m in [16usize, 502, 10_000] {
            for alpha in [0.5, 0.7, 0.9] {
                let ln_m = (m as f64).ln();
                let expected =
                    alpha * (2. * ln_m).sqrt() - (2. * (1. - alpha * alpha) * ln_m.ln()).sqrt();
                assert!((get_threshold(alpha, m) - expected).abs() < 1e-12);
            }
        }
        // eta must sit below the collision band: a query has to be able to reach the
        // filters that points were assigned to, otherwise nothing is ever found.
        let (lower, _) = collision_band(502);
        assert!(get_threshold(0.7, 502) < lower);
    }

    /// Unit sphere sampling must produce normalized vectors and be reproducible.
    #[test]
    fn test_generate_unit_sphere_vectors() {
        let vectors = generate_unit_sphere_vectors(32, 8, 7);
        assert_eq!(vectors.len(), 32);
        for vector in &vectors {
            assert_eq!(vector.len(), 8);
            assert!(is_normalized(vector));
        }
        assert_eq!(vectors, generate_unit_sphere_vectors(32, 8, 7));
        assert_ne!(vectors, generate_unit_sphere_vectors(32, 8, 8));
    }

    /// Gaussian filters must be reproducible and have the right shape.
    #[test]
    fn test_generate_normal_gaussian_vectors_seeded() {
        let vectors = generate_normal_gaussian_vectors_seeded(64, 4, 3);
        assert_eq!(vectors.len(), 64);
        assert_eq!(vectors[0].len(), 4);
        assert_eq!(vectors, generate_normal_gaussian_vectors_seeded(64, 4, 3));
        // Sample mean of 256 standard normal entries is within 4 standard errors of 0.
        let mean = vectors.iter().flatten().sum::<f64>() / 256.;
        assert!(mean.abs() < 4. / 16.);
    }
}
