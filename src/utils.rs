use rand::distributions::Distribution;
use rand_distr::Normal;
use std::io::{Error, ErrorKind};

/// COmputes the dot product of two vectors.
pub fn dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

/// Computes the cosine similarity between two vectors.
pub fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = dot_product(vec1, vec2);
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



#[cfg(test)]
mod tests {
    use super::*;

    /// Test function to check if the dot product function works.
    /// The test checks if the dot product of two vectors is computed correctly.
    #[test]
    fn test_dot_product() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = dot_product(&vec1, &vec2);
        assert_eq!(result, 32.0);
    }

    /// Test function to check if the cosine similarity function works.
    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = cosine_similarity(&vec1, &vec2);
        assert_eq!(result, 0.9746318461970762);
    }

    /// Test function to check if the generate_gaussian_vectors function works.
    /// The test checks if the generated vectors have the correct length and dimension.
    #[test]
    fn test_generate_gaussian_vectors() {
        let n = 10;
        let d = 5;
        let vectors = generate_gaussian_vectors(n, d).unwrap();
        assert_eq!(vectors.len(), n);
        assert_eq!(vectors[0].len(), d);
    }
}

