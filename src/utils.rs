use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rayon::prelude::*;
use std::io;

/// Computes the dot product of two vectors.
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

/// Generates n random Normal Gaussian vectors of dimension d.
pub fn generate_normal_gaussian_vectors(n: usize, d: usize) -> Result<Vec<Vec<f64>>, io::Error> {
    // Step 1: Define the normal distribution with mean 0 and standard deviation sigma
    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Failed to create normal distribution: {}", e),
        )
    })?;

    // Step 2: Generate N random Gaussian vectors of dimension d
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let vector: Vec<f64> = (0..d)
            .map(|_| normal.sample(&mut rand::thread_rng()))
            .collect();
        vectors.push(vector);
    }

    // Return the generated vectors
    Ok(vectors)
}

/// Generates n random Normal Gaussian vectors of dimension d.
pub fn generate_normal_gaussian_vectors_parallel(n: usize, d: usize) -> Result<Vec<Vec<f64>>, io::Error> {
    // Step 1: Define the normal distribution with mean 0 and standard deviation sigma
    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Failed to create normal distribution: {}", e),
        )
    })?;

    // Step 2: Generate N random Gaussian vectors of dimension d in parallel
    let vectors: Vec<Vec<f64>> = (0..n).into_par_iter()
        .map(|_| {
            (0..d)
                .map(|_| normal.sample(&mut rand::thread_rng()))
                .collect()
        })
        .collect();

    // Return the generated vectors
    Ok(vectors)
}

/// Helper function to check if a vector is normalized.
pub fn is_normalized(vector: &Vec<f64>) -> bool {
    let norm = vector.iter().map(|x| x * x).sum::<f64>();
    (norm - 1.0).abs() <= 1e-6
}

/// Normalizes a vector to have unit length.
pub fn normalize_vector(vector: &mut Vec<f64>) {
    let norm: f64 = vector.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    for i in 0..vector.len() {
        vector[i] /= norm;
    }
}

/// Helper function to find a close vector in a list of vectors.
pub fn find_close_vector(query: &Vec<f64>, vectors: &Vec<Vec<f64>>, beta: f64) -> Option<Vec<f64>> {
    for vector in vectors {
        if dot_product(query, vector) >= beta {
            return Some(vector.clone());
        }
    }
    None
}

/// Query threshold `eta = alpha * sqrt(2 log m) - sqrt(2 (1 - alpha^2) log log m)`.
///
/// A filter (Gaussian vector) `a` is inspected by a query `q` if `<a, q> >= eta`.
/// The value follows from the theory of concomitant order statistics: a point at
/// inner product `alpha` from `q` is associated to a filter whose inner product with
/// `q` is distributed as `N(alpha * sqrt(2 log m), 1 - alpha^2)`, and `eta` sits
/// `sqrt(2 log log m)` standard deviations below that mean.
pub fn get_threshold(alpha: f64, m: usize) -> f64 {
    let ln_m = (m as f64).ln();
    let ln_ln_m = ln_m.ln();
    let first_term = alpha * (2. * ln_m).sqrt();
    let second_term = -(2. * (1. - alpha.powi(2)) * ln_ln_m).sqrt();
    let threshold = first_term + second_term;
    threshold
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

    #[allow(unused_imports)]
    use super::*;

    /// Test function to check if the dot product function works.
    /// The test checks if the dot product of two vectors is computed correctly.
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

    /// Test function to check if the generate_gaussian_vectors function works.
    /// The test checks if the generated vectors have the correct length and dimension.
    #[test]
    fn test_generate_gaussian_vectors() {
        let n = 10;
        let d = 5;
        let vectors = generate_normal_gaussian_vectors(n, d).unwrap();
        assert_eq!(vectors.len(), n);
        assert_eq!(vectors[0].len(), d);
    }

    /// Test function to check if the normalize_vector function works.
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

    /// The collision band must sit just below `sqrt(2 log m)` and be non-empty.
    #[test]
    fn test_collision_band() {
        for m in [16usize, 100, 10_000, 1_000_000] {
            let (lower, upper) = collision_band(m);
            assert!(lower < upper, "empty band for m = {}", m);
            assert!((upper - (2. * (m as f64).ln()).sqrt()).abs() < 1e-12);
            // The band shrinks (relatively) as m grows, but stays a constant factor away.
            assert!(lower > 0.0);
        }
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
