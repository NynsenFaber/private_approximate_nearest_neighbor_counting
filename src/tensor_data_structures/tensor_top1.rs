use super::query::query;
use super::top1::Top1;
use crate::utils::get_threshold;
use std::collections::HashMap;
use std::io;
use rand_distr::num_traits::Pow;

pub struct TensorTop1 {
    pub top1_list: Vec<Top1>,
    pub hash_table: HashMap<String, Vec<Vec<f64>>>,
    pub alpha: f64,
    pub beta: f64,
}

impl TensorTop1 {
    pub fn new(data: Vec<Vec<f64>>,
               alpha: f64,
               beta: f64,
               theta: f64,
               fast_preprocessing: bool,
    ) -> Self {
        // Number of data points
        let n = data.len() as f64;
        // Number of Top1 structures
        let t = if fast_preprocessing{
            // Fast preprocessing n^{1+o(1)}
            (n.ln().powf(1. / 8.) / (1. - alpha.powi(2))).ceil() as usize
        }
        else{
            // Pre-processing as query time n^{1+o(1)}
            (1. / (1. - alpha.powi(2))).ceil() as usize
        };
        // Update theta
        let theta = theta / (t as f64);

        //// Print parameters
        {
            println!("\nParameters:");
            println!("Number of data points: {}", n);
            println!("Dimension of the data points: {}", data[0].len());
            println!("Alpha: {}", alpha);
            println!("Beta: {}", beta);
            println!("Fast P  reprocessing: {}", fast_preprocessing);
            println!("Number of Top1 structures: {}", t);
            let m = (n as f64).pow(theta / (1. - alpha.powi(2))).ceil() as usize;
            println!("Number of Gaussian vectors for each Top1 structure: {}", m);
            let threshold = get_threshold(alpha, m);
            println!("Threshold: {}", threshold);
            println!("\n");
        }

        //// Store t Top1 structures
        let mut top1_list = Vec::new();
        for i in 0..t {
            println!("Creating Top1 structure {}/{}", i, t);
            let top1 = Top1::new(&data, alpha, beta, theta);
            top1_list.push(top1);
        }

        //// Create the Hash Table (move data into the hash table)
        println!("Creating the Hash Table");
        let hash_table = get_hash_table(data, &top1_list);

        TensorTop1 {
            top1_list,
            hash_table,
            alpha,
            beta,
        }
    }

    pub fn query(&self, q: &Vec<f64>) -> Result<Option<Vec<f64>>, io::Error> {
        println!("Querying the TensorTop1 structure");
        query(q, &self.top1_list, &self.hash_table, self.beta)
    }
}

/// Create the Hash Table (HashMap of Vec<Vec<f64>> indexed by String)
/// The string is the concatenation of the indices of the closest Gaussian vectors
/// of each Top1 structure. Example, the string "0#1#2#" means that the closest Gaussian
/// vector of the first Top1 structure is the first, the second is the second, and the third
/// is the third.
///
/// Parameters:
/// data: Vec<Vec<f64>> - The data points as reference
/// top1_list: &Vec<Top1> - The list of Top1 structures as reference
///
/// Returns:
/// HashMap<String, Vec<Vec<f64>>> - The Hash Table indexed by the string of indices
fn get_hash_table(data: Vec<Vec<f64>>, top1_list: &Vec<Top1>) -> HashMap<String, Vec<Vec<f64>>> {

    // Initialize the Hash Table
    let mut hash_table: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

    // Iterate over each data vector using a consuming iterator
    for (i, point) in data.into_iter().enumerate() {

        // Initialize the hash
        let mut hash: String = String::new();

        // Get the hashes of each data structure and concatenate them
        // Example: "0#1#2#"
        for top1 in top1_list.iter() {
            // Concatenate the hash of the i-th data point
            hash += &top1.hash(i);
        }

        // Insert the point in the Hash Table
        hash_table
            .entry(hash)
            .or_insert_with(Vec::new)
            .push(point)
    }

    hash_table
}
