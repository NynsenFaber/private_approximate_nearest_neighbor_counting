use savefile::prelude::*;
use savefile_derive::Savefile;
use std::io;

// Load cosine_similarity function from utils.rs
use ann_rust::tensor_data_structures::tensor_top1::TensorTop1;
use ann_rust::utils::{generate_normal_gaussian_vectors, dot_product};

#[derive(Savefile)]
struct GaussianVectors {
    vectors: Vec<Vec<f64>>,
}

fn main() {
    let n = 1_000_000; // Number of vectors
    let d = 100; // Dimension of each vector
    let alpha: f64 = 0.9; // close point according to cosine similarity
    let beta: f64 = 0.55; // far point according to cosine similarity
    let fast_preprocessing = true;

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
            let vectors = generate_normal_gaussian_vectors(n, d).unwrap();
            vectors
        }
    };

    // Create Top1 struct
    let theta = (1. - alpha.powi(2)) * (1. - beta.powi(2)) / (1. - alpha * beta).powi(2);
    // Get first vector to query
    let query = data[0].clone();
    // Create TensorTop1 struct
    let tensor_top1 = TensorTop1::new(data, alpha, beta, theta, fast_preprocessing);

    // Query the Top1 struct
    let result = tensor_top1.query(&query);
    match result {
        Ok(Some(close_point)) => {
            let dot_product = dot_product(&query, &close_point);
            println!("Close point found with dot_product: {:?}", dot_product);
        }
        Ok(None) => {
            println!("No close point found.");
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
        }
    }
}

fn load_vectors(file_name: &str) -> io::Result<GaussianVectors> {
    load_file(file_name, 0)
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("Failed to load file: {}", e)))
}
