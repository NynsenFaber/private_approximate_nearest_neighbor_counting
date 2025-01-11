use rand::distributions::Distribution;
use rand_distr::Normal;
use std::io::{Error, ErrorKind};

/// Computes the cosine similarity between two vectors.
pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let magnitude2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot_product / (magnitude1 * magnitude2)
}

/// Generates N random Gaussian vectors with zero mean and variance sigma^2.
pub fn generate_gaussian_vectors(n: usize, d: usize) -> std::io::Result<Vec<Vec<f64>>> {
    // Step 1: Define the normal distribution with mean 0 and standard deviation sigma
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "Invalid standard deviation"))?;

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
