use crate::utils::cosine_similarity;
use crate::utils::generate_gaussian_vectors;
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
        let n = data.len(); // Number of vectors in the data
        let d = data[0].len(); // Dimension of each vector
        let m = (n as f64).pow(theta / (1. - alpha.powf(2.))).ceil() as usize; // Number of Gaussian vectors

        // Generate Gaussian vectors
        let gaussian_vectors = generate_gaussian_vectors(m, d).unwrap();

        // Create hash table
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

    /// Given a query `q`, return all the indices of the Gaussian vectors with cosine similarity
    /// greater than or equal to the threshold.
    fn search(&self, q: &Vec<f64>) -> Vec<usize> {
        let mut result = Vec::new();
        for (i, gaussian_vector) in self.gaussian_vectors.iter().enumerate() {
            if cosine_similarity(&q, gaussian_vector) >= self.threshold {
                result.push(i);
            }
        }
        result
    }

    /// Given a query `q`, return a close point according to cosine similarity.
    pub fn query(&self, q: Vec<f64>) -> Option<Vec<f64>> {
        // Get indices of Gaussian vectors that meet the threshold
        let indices = self.search(&q);

        // If no vectors meet the threshold, return None
        if indices.is_empty() {
            return None;
        }

        // Search in the hash table for a close vector with cosine similarity greater than `beta`
        for i in indices {
            if let Some(vectors) = self.hash_table.get(&i) {
                for vector in vectors {
                    if cosine_similarity(&q, vector) >= self.beta {
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
    let second_term = -(2. * (1. - alpha.powf(2.) * ((m as f64).ln()).ln())).sqrt();
    first_term + second_term
}

/// For each vector in `data`, find the Gaussian vector with the highest cosine similarity.
/// Store the result in a `HashMap` where the key is the index of the Gaussian vector and
/// the value is the list of data vectors that are closest to it.
fn get_hash_table(
    data: &Vec<Vec<f64>>,
    gaussian_vectors: &Vec<Vec<f64>>,
) -> HashMap<usize, Vec<Vec<f64>>> {
    let mut closest_gaussian_vectors: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();

    // Iterate over each data vector
    for data_vector in data.iter() {
        let mut max_cosine_similarity = f64::MIN;
        let mut max_cosine_similarity_index = 0;

        // Iterate over each Gaussian vector
        for (j, gaussian_vector) in gaussian_vectors.iter().enumerate() {
            // Compute cosine similarity between the data vector and this Gaussian vector
            let cosine_similarity_value = cosine_similarity(data_vector, gaussian_vector);

            if cosine_similarity_value > max_cosine_similarity {
                max_cosine_similarity = cosine_similarity_value;
                max_cosine_similarity_index = j;
            }
        }

        // Insert or update the list of data vectors for the closest Gaussian vector
        closest_gaussian_vectors
            .entry(max_cosine_similarity_index)
            .or_insert_with(Vec::new)
            .push(data_vector.clone());
    }

    closest_gaussian_vectors
}
