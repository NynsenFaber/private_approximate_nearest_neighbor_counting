mod top_1;
mod utils;

use rand::distributions::Distribution;
use rand_distr::num_traits::Pow;
use savefile::prelude::*;
use savefile_derive::Savefile;
use std::io::{Error, ErrorKind};

// Load cosine_similarity function from utils.rs
use crate::top_1::Top1;
use utils::generate_gaussian_vectors;

#[derive(Savefile)]
struct GaussianVectors {
    vectors: Vec<Vec<f64>>,
}

fn main() {
    let n = 100; // Number of vectors
    let d = 10; // Dimension of each vector
    let alpha: f64 = 0.9; // close point according to cosine similarity
    let beta: f64 = 0.5; // far point according to cosine similarity

    // Load file
    let file_name = format!("data/dimension_{}/sample_{}.bin", d, n);
    // Load or generate data
    let data = match load_vectors(&file_name) {
        Ok(data) => {
            println!(
                "Successfully loaded {} vectors from '{}'.",
                data.vectors.len(),
                file_name
            );
            data.vectors
        }
        Err(e) => {
            eprintln!("Failed to load vectors: {}. Generating new vectors...", e);
            let vectors = generate_gaussian_vectors(n, d).unwrap();
            vectors
        }
    };

    // Create Top1 struct
    let theta = (1. - alpha.powi(2)) * (1. - beta.powi(2)) / (1. - alpha * beta).powi(2);
    let query = data[0].clone();
    let top1 = Top1::new(data, alpha, beta, theta);

    // Query the Top1 struct
    let result = top1.query(query);
    match result {
        Some(close_point) => {
            println!("Close point found: {:?}", close_point);
        }
        None => {
            println!("No close point found.");
        }
    }
}

fn load_vectors(file_name: &str) -> std::io::Result<GaussianVectors> {
    load_file(file_name, 0)
        .map_err(|e| Error::new(ErrorKind::NotFound, format!("Failed to load file: {}", e)))
}
