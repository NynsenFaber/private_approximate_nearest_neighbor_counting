use rand::distributions::Distribution;
use rand_distr::Normal;
use std::io::{Error, ErrorKind};

/// Computes the dot product of two vectors.
pub fn get_dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
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
        let result = get_dot_product(&vec1, &vec2);
        assert_eq!(result, 32.0);
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
