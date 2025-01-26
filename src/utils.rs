use rand::distributions::Distribution;
use rand_distr::Normal;
use std::collections::HashMap;
use std::io;

/// Computes the dot product of two vectors.
pub fn get_dot_product(vec1: &[f64], vec2: &[f64]) -> f64 {
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

pub fn get_threshold(alpha: f64, m: usize) -> f64 {
    let first_term = alpha * (2. * (m as f64).ln()).sqrt();
    let second_term = -(2. * (1. - alpha.powi(2)) * ((m as f64).ln()).ln()).sqrt();
    let threshold = first_term + second_term;
    threshold
}

/// Given a query `q`, return a close point according to dot product.
pub fn query(
    gaussian_vectors: &Vec<Vec<f64>>,
    query: &Vec<f64>,
    threshold: f64,
    hash_table: &HashMap<usize, Vec<Vec<f64>>>,
    beta: f64,
) -> Result<Option<Vec<f64>>, io::Error> {
    // Check if the query vector is normalized
    if !is_normalized(query) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Query vector is not normalized",
        ));
    }
    // Get indices of Gaussian vectors that meet the threshold
    let indices = match search(gaussian_vectors, query, threshold) {
        None => return Ok(None), // No matching Gaussian vectors
        Some(indices) => indices,
    };

    // Search for a close vector in the hash table
    for i in indices {
        if let Some(vectors) = hash_table.get(&i) {
            if let Some(close_vector) = find_close_vector(query, vectors, beta) {
                if cfg!(test) {println!("Found a close vector! .");}
                return Ok(Some(close_vector));
            }
        }
    }
    if cfg!(test) {println!("No close vector found.");}
    // If no vector meets the `beta` threshold, return None
    Ok(None)
}

/// Helper function to check if a vector is normalized.
fn is_normalized(vector: &Vec<f64>) -> bool {
    let norm = vector.iter().map(|x| x * x).sum::<f64>();
    (norm - 1.0).abs() <= 1e-6
}

/// Helper function to find a close vector in a list of vectors.
fn find_close_vector(query: &Vec<f64>, vectors: &Vec<Vec<f64>>, beta: f64) -> Option<Vec<f64>> {
    for vector in vectors {
        if get_dot_product(query, vector) >= beta {
            return Some(vector.clone());
        }
    }
    None
}

/// Given a `query`, return all the indices of the Gaussian vectors with dot product
/// greater than or equal to the `threshold`.
fn search(
    gaussian_vectors: &Vec<Vec<f64>>,
    query: &Vec<f64>,
    threshold: f64,
) -> Option<Vec<usize>> {
    let mut result = Vec::new();
    for (i, gaussian_vector) in gaussian_vectors.iter().enumerate() {
        if get_dot_product(query, gaussian_vector) >= threshold {
            result.push(i);
        }
    }
    // If vector is empty, return None
    if result.is_empty() {
        if cfg!(test) {println!("Search returned a empty vector .");}
        None
    } else {
        if cfg!(test) {println!("Search returned a NOT empty vector.");}
        Some(result)
    }
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
        let vectors = generate_normal_gaussian_vectors(n, d).unwrap();
        assert_eq!(vectors.len(), n);
        assert_eq!(vectors[0].len(), d);
    }

    /// Test function to check if search function works.
    #[test]
    fn test_search() {
        let alpha = 0.9;
        let n = 100;
        let gaussian_vectors = generate_normal_gaussian_vectors(n, 3).unwrap();
        let threshold = get_threshold(alpha, n);

        let query = vec![1.0, 2.0, 3.0];
        let indices = search(&gaussian_vectors, &query, threshold);

        // Get all Gaussian vector indices that meet the threshold
        let matched_gaussian_indices: Vec<usize> = gaussian_vectors
            .iter()
            .enumerate()
            .filter(|(_, gaussian_vector)| get_dot_product(&query, gaussian_vector) >= threshold)
            .map(|(i, _)| i)
            .collect();

        // Ensure that the indices returned by `search` match the expected indices
        assert_eq!(indices, Some(matched_gaussian_indices));
    }
}
