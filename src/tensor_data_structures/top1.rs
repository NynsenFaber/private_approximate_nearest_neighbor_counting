use crate::utils::{generate_normal_gaussian_vectors, get_dot_product, get_threshold};
use rand_distr::num_traits::Pow;
use std::collections::HashMap;
use std::io;

pub struct Top1 {
    pub gaussian_vectors: Vec<Vec<f64>>,
    pub hash_table: HashMap<usize, Vec<Vec<f64>>>,
    pub threshold: f64,
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

    /// Given an index (a number from 0 to n-1) indicating a point of the dataset, return
    /// the hash table entry for that point.
    pub fn get_hash(&self, index: usize) -> Option<&Vec<Vec<f64>> {
        self.hash_table.get(&index)
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
}
