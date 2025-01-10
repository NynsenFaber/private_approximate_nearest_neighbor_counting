use rand::distributions::Distribution;
use rand_distr::num_traits::Pow;
use rand_distr::Normal;
use savefile::prelude::*;
use savefile_derive::Savefile;
use std::collections::HashMap;
use std::io::{Error, ErrorKind};

#[derive(Savefile)]
struct GaussianVectors {
    vectors: Vec<Vec<f64>>,
}

fn main() {
    let n = 100; // Number of vectors
    let d = 10; // Dimension of each vector
    let alpha = 0.9;  // close point according to cosine similarity
    let beta = 0.5;  // far point according to cosine similarity
    let theta = (1. - alpha.pow(2)) * (1. - beta.pow(2)) / (1. - alpha * beta).pow(2);

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

    // Generate the Gaussian vectors
    let number_of_vectors: usize = (n as f64).powf(theta / (1. - alpha.pow(2))).ceil() as usize;
    println!("Number of vectors: {}", number_of_vectors);
    let gaussian_vectors = generate_gaussian_vectors(number_of_vectors, d).unwrap();
    println!(
        "Generated {} vectors, of dimension {}.",
        gaussian_vectors.len(),
        gaussian_vectors[0].len()
    );

    // For each vector in data, get the gaussian vector with highest cosine similarity. Store the result in a
    // HashMap with the index of the data vector as the key and the index of the gaussian vector as the value.
    let mut closest_gaussian_vectors: HashMap<usize, usize> = HashMap::new();
    for (i, data_vector) in data.iter().enumerate() {
        let mut max_cosine_similarity = f64::MIN;
        let mut max_cosine_similarity_index = 0;
        for (j, gaussian_vector) in gaussian_vectors.iter().enumerate() {
            let cosine_similarity_value = cosine_similarity(data_vector, gaussian_vector);
            if cosine_similarity_value > max_cosine_similarity {
                max_cosine_similarity = cosine_similarity_value;
                max_cosine_similarity_index = j;
            }
        }
        closest_gaussian_vectors.insert(i, max_cosine_similarity_index);
    }

    println!("Closest Gaussian Vectors: {:?}", closest_gaussian_vectors);
}

/// Generates N random Gaussian vectors with zero mean and variance sigma^2.
fn generate_gaussian_vectors(n: usize, d: usize) -> std::io::Result<Vec<Vec<f64>>> {
    // Step 1: Define the normal distribution with mean 0 and standard deviation sigma
    let normal = Normal::new(0.0, 1.0)
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "Invalid standard deviation"))?;

    // Step 2: Generate N random Gaussian vectors of dimension d
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let vector: Vec<f64> = (0..d)
            .map(|_| normal.sample(&mut rand::thread_rng()))
            .collect();
        vectors.push(vector);
    }

    // Return the generated vectors
    Ok(vectors)
}

fn load_vectors(file_name: &str) -> std::io::Result<GaussianVectors> {
    load_file(file_name, 0)
        .map_err(|e| Error::new(ErrorKind::NotFound, format!("Failed to load file: {}", e)))
}

/// Computes the cosine similarity between two vectors.
fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let magnitude2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot_product / (magnitude1 * magnitude2)
}