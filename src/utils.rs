use rand::distributions::Distribution;
use rand_distr::Normal;
use std::io;
use rayon::prelude::*;

/// Computes the dot product of two vectors.
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
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

pub fn get_threshold(alpha: f64, m: usize) -> f64 {
    let ln_m = (m as f64).ln();
    let ln_ln_m = ln_m.ln();
    let first_term = alpha * (2. * ln_m).sqrt();
    let second_term = -(2. * (1. - alpha.powi(2)) * ln_ln_m).sqrt();
    let threshold = first_term + second_term;
    threshold
}

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
}
