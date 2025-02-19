use super::top1::Top1;
use super::query::query;
use std::collections::HashMap;
use std::io;

pub struct TensorTop1 {
    pub top1_list: Vec<Top1>,
    pub hash_table: HashMap<String, Vec<Vec<f64>>>,
    pub alpha: f64,
    pub beta: f64,
    pub threshold: f64,
}

impl TensorTop1 {
    pub fn new(data: Vec<Vec<f64>>, alpha: f64, beta: f64, theta: f64) -> Self {
        // Number of data points
        let n = data.len() as f64;
        // Number of Top1 structures
        let t = ( n.ln().powf(1. / 8.) / (1. - alpha.powf(2.))).ceil() as usize;
        // Update theta
        let theta = theta / (t as f64);
        //// Store t Top1 structures
        let mut top1_list = Vec::new();
        for i in 0..t {
            println!("Creating Top1 structure {}/{}", i, t);
            let top1 = Top1::new(data.clone(), alpha, beta, theta);
            top1_list.push(top1);
        }
        println!("Creating the Hash Table");
        let hash_table = get_hash_table(&data, &top1_list);
        // All the thresholds are the same, so we can just use the first one
        let threshold = top1_list[0].threshold;
        TensorTop1 {
            top1_list,
            hash_table,
            alpha,
            beta,
            threshold,
        }
    }

    pub fn query(&self, q: &Vec<f64>) -> Result<Option<Vec<f64>>, io::Error> {
        println!("Querying the TensorTop1 structure");
        query(q,
              &self.top1_list,
              &self.hash_table,
              self.beta)
    }
}

fn get_hash_table(
    data: &Vec<Vec<f64>>,
    top1_list: &Vec<Top1>,
) -> HashMap<String, Vec<Vec<f64>>> {
    // Initialize the Hash Table
    let mut hash_table: HashMap<String, Vec<Vec<f64>>> = HashMap::new();

    // Iterate over each data vector
    for (i, point) in data.iter().enumerate() {
        let mut hash: String = String::new();
        // Get the hashes of each data structure and concatenate them
        for top1 in top1_list.iter() {
            hash += &top1.hash(i);
        }
        // Insert the point in the Hash Table
        hash_table
            .entry(hash)
            .or_insert_with(Vec::new)
            .push(point.clone())
    }

    hash_table
}
