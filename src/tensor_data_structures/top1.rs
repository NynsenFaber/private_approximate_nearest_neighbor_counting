use crate::utils::{generate_normal_gaussian_vectors, dot_product, get_threshold};
use crate::checks::check_input;
use rand_distr::num_traits::Pow;

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
        let match_list = get_match_list(&data, &gaussian_vectors);

        // Create Top1 struct
        Top1 {
            gaussian_vectors,
            match_list,
            threshold: get_threshold(alpha, m),
        }
    }

    /// Given a `query`, return all the indices of the Gaussian vectors with dot product
    /// greater than or equal to the `threshold`.
    pub fn search(
        gaussian_vectors: &[Vec<f64>],
        query: &[f64],
        threshold: f64,
    ) -> Option<Vec<usize>> {
        let result: Vec<usize> = gaussian_vectors
            .iter()
            .enumerate()
            .filter_map(|(i, gaussian_vector)| {
                if dot_product(query, gaussian_vector) >= threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }
}

/// For each vector in `data`, find the Gaussian vector with the highest dot product.
/// Store the indices of the closest Gaussian vector in a Vec<usize>.
fn get_match_list(
    data: &Vec<Vec<f64>>,
    gaussian_vectors: &Vec<Vec<f64>>,
) -> Vec<usize> {
    let mut match_list: Vec<usize> = Vec::new();

    // Iterate over each data vector
    for point in data.iter() {
        let mut max_dot_product = f64::MIN;
        let mut max_dot_product_index = 0;
        // Iterate over each Gaussian vector
        for (j, gaussian_vector) in gaussian_vectors.iter().enumerate() {
            // Compute dot product between the data vector and this Gaussian vector
            let dot_product_value = dot_product(point, gaussian_vector);
            // Update the maximum dot product and the index of the closest Gaussian vector
            if dot_product_value > max_dot_product {
                max_dot_product = dot_product_value;
                max_dot_product_index = j;
            }
        }
        // Store the index of the closest Gaussian vector
        match_list.push(max_dot_product_index);
    }
    match_list
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
        let result = Top1::search(&gaussian_vectors, &query, threshold);
        assert_eq!(result, Some(vec![1]));

        let gaussian_vectors = vec![vec![1.0, 0., 0.], vec![0., 1.0, 0.]];
        let query = vec![1.0, 0.5, 0.];
        let threshold = 0.5;
        let result = Top1::search(&gaussian_vectors, &query, threshold);
        assert_eq!(result, Some(vec![0, 1]));

        let gaussian_vectors = vec![vec![1.0, 0., 0.], vec![0., 1.0, 0.]];
        let query = vec![1.0, 0.5, 0.];
        let threshold = 2.0;
        let result = Top1::search(&gaussian_vectors, &query, threshold);
        assert_eq!(result, None);
    }
}
