mod top_1;
mod utils;

use rand::distributions::Distribution;
use rand_distr::num_traits::Pow;
use savefile::prelude::*;
use savefile_derive::Savefile;
use std::io::{Error, ErrorKind};

// Load cosine_similarity function from utils.rs
use utils::generate_gaussian_vectors;

#[derive(Savefile)]
struct GaussianVectors {
    vectors: Vec<Vec<f64>>,
}

fn main() {
    let n = 100; // Number of vectors
    let d = 10; // Dimension of each vector
    // let alpha = 0.9;  // close point according to cosine similarity
    // let beta = 0.5;  // far point according to cosine similarity

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
    // print the first vector
    println!("First vector: {:?}", data[0]);
}


fn load_vectors(file_name: &str) -> std::io::Result<GaussianVectors> {
    load_file(file_name, 0)
        .map_err(|e| Error::new(ErrorKind::NotFound, format!("Failed to load file: {}", e)))
}
