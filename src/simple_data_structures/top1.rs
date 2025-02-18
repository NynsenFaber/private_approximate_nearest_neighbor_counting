use crate::utils::{generate_normal_gaussian_vectors, get_dot_product, get_threshold};
use crate::checks::check_input;
use super::query::query;
use rand_distr::num_traits::Pow;
use std::collections::HashMap;
use std::io;

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
            Ok(_) => {}
            Err(err) => eprintln!("Input validation failed: {}", err),
        }

        // Dimension of the vectors
        let d = data[0].len();
        // Number of vectors in the data
        let n = data.len();
        // Number of Gaussian vectors
        let m = (n as f64).pow(theta / (1. - alpha.powf(2.))).ceil() as usize;

        // Generate Gaussian vectors
        println!("Generating {} Gaussian vectors...", m);
        let gaussian_vectors = generate_normal_gaussian_vectors(m, d).unwrap();

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

    /// Given a query `q`, return a close point according to dot product.
    pub fn query(&self, q: &Vec<f64>) -> Result<Option<Vec<f64>>, io::Error> {
        query(
            &self.gaussian_vectors,
            q,
            self.threshold,
            &self.hash_table,
            self.beta,
        )
    }
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

/// Test function for Top1 struct.
#[cfg(test)]
mod tests {
    use super::*;

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
        // if threshold is lower than all the dot products, the result should be None
        let mut flag: bool = true;
        for vector in top1.gaussian_vectors.iter() {
            let dot_product = get_dot_product(&query, vector);
            // A vector has a dot product greater than the threshold, so the result should not be None
            if dot_product >= top1.threshold {
                println!("Dot product: {}", dot_product);
                flag = false;
                break;
            }
        }
        if flag {
            // Result should be None
            assert_eq!(result.unwrap(), None);
        } else {
            // Result should be close to the query
            let dot_product = get_dot_product(&query, &result.unwrap().unwrap());
            assert!(dot_product >= beta);
        }

        // Bad query
        let query = vec![2.0, 0.0, 0.0];
        let result = top1.query(&query);
        // Result should be an Error
        assert!(result.is_err());
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
}
