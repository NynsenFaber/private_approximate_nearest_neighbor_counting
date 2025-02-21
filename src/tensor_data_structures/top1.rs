use crate::checks::check_input;
use crate::utils::{dot_product, generate_normal_gaussian_vectors, get_threshold};
use rand_distr::num_traits::Pow;
use rayon::prelude::*;

pub struct Top1 {
    // Random Gaussian vectors
    pub gaussian_vectors: Vec<Vec<f64>>,
    // Vector of length n with the indices of the closest Gaussian vector
    pub match_list: Vec<usize>,
    // threshold
    pub threshold: f64,
}

impl Top1 {
    /// Constructor for the Top1 struct.
    pub fn new(data: &Vec<Vec<f64>>, alpha: f64, beta: f64, theta: f64) -> Self {
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
        let gaussian_vectors = generate_normal_gaussian_vectors(m, d).unwrap();

        // Create match_list using parallel computation
        let match_list = get_match_list_parallel(data, &gaussian_vectors);

        // Create Top1 struct
        Top1 {
            gaussian_vectors,
            match_list,
            threshold: get_threshold(alpha, m),
        }
    }

    /// Given a `query`, return all the indices of the Gaussian vectors with dot product
    /// greater than or equal to the `threshold`. The output is encoded as Vec<String>.
    pub fn search(&self, query: &Vec<f64>) -> Vec<String> {
        let result = search(&self.gaussian_vectors, query, self.threshold);
        println!("Result: {:?}", result);
        result
    }

    /// Given a number from 0 to n-1, return a hash, which is the index of the closest Gaussian vector.
    pub fn hash(&self, i: usize) -> String {
        // format returns a new String
        format!("{}#", self.match_list[i])
    }
}

/// Given a `query`, return all the indices of the Gaussian vectors with dot product
/// greater than or equal to the `threshold`.
fn search(gaussian_vectors: &[Vec<f64>], query: &Vec<f64>, threshold: f64) -> Vec<String> {
    gaussian_vectors
        .iter()
        .enumerate()
        .filter_map(|(i, gaussian_vector)| {
            if dot_product(query, gaussian_vector) >= threshold {
                Some(format!("{}#", i))
            } else {
                None
            }
        })
        .collect()
}

/// For each vector in `data`, find the Gaussian vector with the highest dot product.
/// Store the indices of the closest Gaussian vector in a Vec<usize>.
#[allow(dead_code)]
fn get_match_list(
    data: &Vec<Vec<f64>>,             // Input data vectors
    gaussian_vectors: &Vec<Vec<f64>>, // Gaussian vectors
) -> Vec<usize> {
    data.iter()
        .map(|point| {
            gaussian_vectors
                .iter()
                .enumerate()
                .map(|(j, gaussian_vector)| (j, dot_product(point, gaussian_vector)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

/// For each vector in `data`, find the Gaussian vector with the highest dot product.
/// Store the indices of the closest Gaussian vector in a Vec<usize>.
/// This function uses Rayon to parallelize the computation.
#[allow(dead_code)]
fn get_match_list_parallel(
    data: &Vec<Vec<f64>>,             // Input data vectors
    gaussian_vectors: &Vec<Vec<f64>>, // Gaussian vectors
) -> Vec<usize> {
    // Use par_iter() to convert into a parallel iterator
    data.par_iter()
        .map(|point| {
            gaussian_vectors
                .iter() // Not many Gaussian vectors, so no need to parallelize
                .enumerate()
                .map(|(j, gaussian_vector)| (j, dot_product(point, gaussian_vector)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0
        })
        .collect()
}

/// Test function for Top1 struct.
#[cfg(test)]
mod tests {
    use super::*;

    // test match_list
    #[test]
    fn test_match_list() {
        let data = vec![vec![1.0, 0., 0.], vec![0., 1.0, 0.]];
        let gaussian_vectors = vec![vec![1.0, 0., 0.], vec![0.5, 0.5, 0.]];
        let match_list = get_match_list(&data, &gaussian_vectors);
        assert_eq!(match_list, vec![0, 1]);
    }

    // test search
    #[test]
    fn test_search() {
        let gaussian_vectors = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let query = vec![1.0, 2.0, 3.0];
        let threshold = 20.0;
        let result = search(&gaussian_vectors, &query, threshold);
        assert_eq!(result, vec![String::from("1#")]);

        let gaussian_vectors = vec![vec![1.0, 0., 0.], vec![0., 1.0, 0.]];
        let query = vec![1.0, 0.5, 0.];
        let threshold = 0.5;
        let result = search(&gaussian_vectors, &query, threshold);
        assert_eq!(result, vec![String::from("0#"), String::from("1#")]);

        let gaussian_vectors = vec![vec![1.0, 0., 0.], vec![0., 1.0, 0.]];
        let query = vec![1.0, 0.5, 0.];
        let threshold = 2.0;
        let result = search(&gaussian_vectors, &query, threshold);
        assert_eq!(result, Vec::<String>::new());
    }
}
