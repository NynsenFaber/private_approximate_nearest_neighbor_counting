use crate::utils::{get_dot_product, generate_gaussian_vectors};
use rand_distr::num_traits::Pow;
use std::collections::HashMap;

pub struct Top1 {
    pub gaussian_vectors: Vec<Vec<f64>>,
    pub hash_table: HashMap<usize, Vec<Vec<f64>>>,
    pub alpha: f64,
    pub beta: f64,
    pub threshold: f64,
    pub m: usize,
}

impl Top1 {
    /// Constructor for the Top1 struct.
    pub fn new(data: Vec<Vec<f64>>, alpha: f64, beta: f64, theta: f64) -> Self {
        // Check inputs
        match check_input(&data, alpha, beta, theta) {
            Ok(_) => {},
            Err(err) => eprintln!("Input validation failed: {}", err),
        }

        let d = data[0].len(); // Dimension of the vectors
        let n = data.len(); // Number of vectors in the data
        let m = (n as f64).pow(theta / (1. - alpha.powf(2.))).ceil() as usize; // Number of Gaussian vectors

        // Generate Gaussian vectors
        println!("Generating {} Gaussian vectors...", m);
        let gaussian_vectors = generate_gaussian_vectors(m, d).unwrap();

        // Create hash table
        println!("Creating hash table...");
        let hash_table = get_hash_table(&data, &gaussian_vectors);

        // Create Top1 struct
        Top1 {
            gaussian_vectors,
            hash_table,
            alpha,
            beta,
            m,
            threshold: get_threshold(alpha, m),
        }
    }

    /// Given a query `q`, return all the indices of the Gaussian vectors with dot product
    /// greater than or equal to the threshold.
    fn search(&self, q: &Vec<f64>) -> Vec<usize> {
        let mut result = Vec::new();
        for (i, gaussian_vector) in self.gaussian_vectors.iter().enumerate() {
            if get_dot_product(&q, gaussian_vector) >= self.threshold {
                result.push(i);
            }
        }
        result
    }

    /// Given a query `q`, return a close point according to dot product.
    pub fn query(&self, q: &Vec<f64>) -> Option<Vec<f64>> {
        // Check if query is normalized
        let norm = q.iter().map(|x| x * x).sum::<f64>();
        if (norm - 1.0).abs() > 1e-6 {
            eprintln!("Query vector is not normalized (norm = {}).", norm);
            return None;
        }

        // Get indices of Gaussian vectors that meet the threshold
        let indices = self.search(q);

        // If no vectors meet the threshold, return None
        if indices.is_empty() {
            return None;
        }

        // Search in the hash table for a close vector with dot product greater than `beta`
        for i in indices {
            if let Some(vectors) = self.hash_table.get(&i) {
                for vector in vectors {
                    if get_dot_product(q, vector) >= self.beta {
                        return Some(vector.clone());
                    }
                }
            }
        }

        // If no vector meets the `beta` threshold, return None
        None
    }
}

fn get_threshold(alpha: f64, m: usize) -> f64 {
    let first_term = alpha * (2. * (m as f64).ln()).sqrt();
    let second_term = -(2. * (1. - alpha.powi(2)) * ((m as f64).ln()).ln()).sqrt();
    let threshold = first_term + second_term;
    threshold
}

/// For each vector in `data`, find the Gaussian vector with the highest dot product.
/// Store the result in a `HashMap` where the key is the index of the Gaussian vector and
/// the value is the list of data vectors that are closest to it.
fn get_hash_table(
    data: &Vec<Vec<f64>>,
    gaussian_vectors: &Vec<Vec<f64>>,
) -> HashMap<usize, Vec<Vec<f64>>> {
    let mut closest_gaussian_vectors: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();

    // Iterate over each data vector
    for data_vector in data.iter() {
        let mut max_dot_product = f64::MIN;
        let mut max_dot_product_index = 0;

        // Iterate over each Gaussian vector
        for (j, gaussian_vector) in gaussian_vectors.iter().enumerate() {
            // Compute dot product between the data vector and this Gaussian vector
            let dot_product_value = get_dot_product(data_vector, gaussian_vector);

            if dot_product_value > max_dot_product {
                max_dot_product = dot_product_value;
                max_dot_product_index = j;
            }
        }

        // Insert or update the list of data vectors for the closest Gaussian vector
        closest_gaussian_vectors
            .entry(max_dot_product_index)
            .or_insert_with(Vec::new)
            .push(data_vector.clone());
    }

    closest_gaussian_vectors
}

/// Check if the input data is valid.
fn check_input(data: &Vec<Vec<f64>>, alpha: f64, beta: f64, theta: f64) -> Result<(), String> {
    // Validate alpha
    if !(0.0 < alpha && alpha < 1.0) {
        return Err("Invalid value for alpha. Alpha must be in the range (0, 1).".to_string());
    }

    // Validate beta
    if !(0.0 < beta && beta < alpha) {
        return Err("Invalid value for beta. Beta must be in the range (0, alpha).".to_string());
    }

    // Validate theta
    if !(theta > 0.0) {
        return Err("Invalid value for theta. Theta must be positive.".to_string());
    }

    // Validate data is non-empty
    if data.is_empty() {
        return Err("Data cannot be empty.".to_string());
    }

    // Check if all vectors have the same dimension and are normalized
    let d = data[0].len(); // Dimension of the first vector
    if d == 0 {
        return Err("Vectors cannot have zero dimensions.".to_string());
    }

    for (i, vector) in data.iter().enumerate() {
        // Check if all vectors have the same dimension
        if vector.len() != d {
            return Err(format!(
                "Vector at index {} has a different dimension (expected {}, got {}).",
                i,
                d,
                vector.len()
            ));
        }

        // Check if the vector is normalized (sum of squares equals 1)
        let norm = vector.iter().map(|x| x * x).sum::<f64>();
        if (norm - 1.0).abs() > 1e-6 {
            return Err(format!(
                "Vector at index {} is not normalized (norm = {}).",
                i,
                norm
            ));
        }
    }

    Ok(())
}


/// Test function for Top1 struct.
#[cfg(test)]
mod tests {
    use super::*;

    pub fn check_result(result: Option<&Vec<f64>>, query: &Vec<f64>, beta: f64) -> bool {
        match result {
            Some(close_point) => {
                let dot_product = get_dot_product(&close_point, &query);
                if dot_product >= beta {
                    true
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// Test function to check if the Top1 struct works.
    #[test]
    fn test_top1_query() {
        // Create a sample data
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let alpha = 0.9;
        let beta = 0.8;
        let theta = 0.5;
        let top1 = Top1::new(data, alpha, beta, theta);

        // Good query
        let query = vec![1.0, 0.0, 0.0];
        let result = top1.query(&query);
        assert!(check_result(result.as_ref(), &query, beta));

        // Bad query
        let query = vec![2.0, 0.0, 0.0];
        let result = top1.query(&query);
        assert!(!check_result(result.as_ref(), &query, beta));
    }

    /// Test function to check if the get_threshold function works.
    #[test]
    fn test_get_threshold() {
        let alpha = 0.9;
        let m = 100;
        let threshold = get_threshold(alpha, m);
        assert!(threshold > 0.0);
    }

    /// Test function to check if the get_hash_table function works.
    #[test]
    fn test_get_hash_table() {
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let gaussian_vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let hash_table = get_hash_table(&data, &gaussian_vectors);

        // Check if the hash table is correct
        assert_eq!(hash_table.len(), 3);
        assert_eq!(hash_table[&0].len(), 2);
        assert_eq!(hash_table[&1].len(), 1);
        assert_eq!(hash_table[&2].len(), 1);
        // Check if the hash table contains the correct data
        assert_eq!(hash_table[&0][0], vec![1.0, 0.0, 0.0]);
        assert_eq!(hash_table[&0][1], vec![1.0, 0.0, 0.0]);
        assert_eq!(hash_table[&1][0], vec![0.0, 1.0, 0.0]);
        assert_eq!(hash_table[&2][0], vec![0.0, 0.0, 1.0]);
    }

    /// Test function to check if search function works.
    #[test]
    fn test_search() {
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let alpha = 0.9;
        let beta = 0.8;
        let theta = 0.5;
        let top1 = Top1::new(data, alpha, beta, theta);

        let query = vec![1.0, 2.0, 3.0];
        let indices = top1.search(&query);

        // Get all Gaussian vector indices that meet the threshold
        let matched_gaussian_indices: Vec<usize> = top1.gaussian_vectors
            .iter()
            .enumerate()
            .filter(|(_, gaussian_vector)| get_dot_product(&query, gaussian_vector) >= top1.threshold)
            .map(|(i, _)| i)
            .collect();

        // Ensure that the indices returned by `search` match the expected indices
        assert_eq!(indices, matched_gaussian_indices);
    }
}