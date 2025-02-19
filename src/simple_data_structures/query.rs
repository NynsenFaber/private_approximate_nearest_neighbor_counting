use crate::utils::{dot_product, is_normalized, find_close_vector};
use std::collections::HashMap;
use std::io;

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

/// Given a `query`, return all the indices of the Gaussian vectors with dot product
/// greater than or equal to the `threshold`.
fn search(
    gaussian_vectors: &Vec<Vec<f64>>,
    query: &Vec<f64>,
    threshold: f64,
) -> Option<Vec<usize>> {
    let mut result = Vec::new();
    for (i, gaussian_vector) in gaussian_vectors.iter().enumerate() {
        if dot_product(query, gaussian_vector) >= threshold {
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
    use crate::utils::get_threshold;
    use crate::utils::generate_normal_gaussian_vectors;
    use super::*;

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
            .filter(|(_, gaussian_vector)| dot_product(&query, gaussian_vector) >= threshold)
            .map(|(i, _)| i)
            .collect();

        // Ensure that the indices returned by `search` match the expected indices
        assert_eq!(indices, Some(matched_gaussian_indices));
    }
}
