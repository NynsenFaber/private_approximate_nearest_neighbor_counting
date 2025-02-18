use crate::utils::{generate_normal_gaussian_vectors, dot_product, get_threshold};
use crate::checks::check_input;
use super::query::query;
use rand_distr::num_traits::Pow;
use std::collections::HashMap;
use std::io;


pub struct CloseTop1 {
    pub gaussian_vectors: Vec<Vec<f64>>,
    pub hash_table: HashMap<usize, Vec<Vec<f64>>>,
    pub alpha: f64,
    pub beta: f64,
    pub threshold: f64,
    pub m: usize,
}

impl CloseTop1 {
    /// Constructor for the Top1 struct.
    pub fn new(data: Vec<Vec<f64>>, alpha: f64, beta: f64, theta: f64) -> Self {
        // Check inputs
        match check_input(&data, alpha, beta, theta) {
            Ok(_) => {}
            Err(err) => eprintln!("Input validation failed: {}", err),
        }

        let d = data[0].len(); // Dimension of the vectors
        let n = data.len(); // Number of vectors in the data
        let m = (n as f64).pow(theta / (1. - alpha.powf(2.))).ceil() as usize; // Number of Gaussian vectors

        // Generate Gaussian vectors
        println!("Generating {} Gaussian vectors...", m);
        let gaussian_vectors = generate_normal_gaussian_vectors(m, d).unwrap();

        // Create hash table
        println!("Creating hash table...");
        let hash_table = get_hash_table(&data, &gaussian_vectors);

        // Create Top1 struct
        CloseTop1 {
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

    let m = gaussian_vectors.len() as f64;
    let ln_m = m.ln();
    let left_bound = (2. * ln_m).sqrt() - (3./2.) * (ln_m.ln()/(2. * ln_m).sqrt());
    let right_bound = (2. * ln_m).sqrt();

    // Iterate over each data vector
    for data_vector in data.iter() {

        // Iterate over each Gaussian vector
        for (i, gaussian_vector) in gaussian_vectors.iter().enumerate() {
            // Compute dot product between the data vector and this Gaussian vector
            let dot_product_value = dot_product(data_vector, gaussian_vector);

            if (dot_product_value >= left_bound) && (dot_product_value <= right_bound) {
                // Insert or update the list of data vectors for the closest Gaussian vector
                closest_gaussian_vectors
                    .entry(i)
                    .or_insert_with(Vec::new)
                    .push(data_vector.clone());
                break;
            }
        }

    }

    closest_gaussian_vectors
}


/// Test function for Top1 struct.
#[cfg(test)]
mod tests {
    use super::*;

    /// Test function to check if the Top1 struct works.
    #[test]
    fn test_close_top1_query() {
        // Create a sample data
        let data = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let alpha = 0.9;
        let beta = 0.8;
        let theta = 0.5;
        let top1 = CloseTop1::new(data, alpha, beta, theta);

        // Good query
        let query = vec![1.0, 0.0, 0.0];
        let result = top1.query(&query);
        // if threshold is lower than all the dot products, the result should be None
        let mut flag: bool = true;
        for vector in top1.gaussian_vectors.iter() {
            let dot_product = dot_product(&query, vector);
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
            let dot_product = dot_product(&query, &result.unwrap().unwrap());
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
    fn test_close_top_1_get_hash_table() {
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

        // Count how many vectors are in the hash table
        let mut count_hash = 0;
        for (_, vectors) in hash_table.iter() {
            count_hash += vectors.len();
        }

        let m = gaussian_vectors.len() as f64;
        let ln_m = m.ln();
        let left_bound = (2. * ln_m).sqrt() - (3./2.) * (ln_m.ln()/(2. * ln_m).sqrt());
        let right_bound = (2. * ln_m).sqrt();

        // Compute how many data passes the filter
        let mut count_data = 0;
        let mut data_index: Vec<usize> = Vec::new();
        for (i, data_vector) in data.iter().enumerate() {
            for gaussian_vector in gaussian_vectors.iter() {
                let dot_product_value = dot_product(data_vector, gaussian_vector);
                if (dot_product_value >= left_bound) && (dot_product_value <= right_bound) {
                    count_data += 1;
                    data_index.push(i);
                    break;
                }
            }
        }
        assert_eq!(count_hash, count_data);

        // Check if all the data vectors who passed the filter are in the hash table
        for i in data_index.iter() {
            let mut flag = false;
            for (_, vectors) in hash_table.iter() {
                for vector in vectors.iter() {
                    if data[*i] == *vector {
                        flag = true;
                        break;
                    }
                }
            }
            assert!(flag);
        }
    }
}