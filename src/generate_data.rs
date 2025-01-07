use rand_distr::{Distribution, Normal};
use std::fs::{create_dir_all};
use savefile::prelude::*;
use savefile_derive::Savefile;

#[derive(Savefile, PartialEq, Debug)] // Derive Save file traits for serialization
pub struct GaussianVectors {
    pub vectors: Vec<Vec<f64>>,
}

/// Generates N random Gaussian vectors with zero mean and variance sigma^2,
/// saves them in a folder called `data` under the name `sample_N.bin`.
pub fn generate_and_save_gaussian_vectors(n: usize, d: usize, sigma: f64) -> Result<(), SavefileError> {
    // Step 1: Define the normal distribution with mean 0 and standard deviation sigma
    let normal = Normal::new(0.0, sigma).expect("Failed to create normal distribution");

    // Step 2: Generate N random Gaussian vectors of dimension d
    let mut vectors = Vec::with_capacity(n);
    for _ in 0..n {
        let vector: Vec<f64> = (0..d).map(|_| normal.sample(&mut rand::thread_rng())).collect();
        vectors.push(vector);
    }

    // Wrap vectors in a serializable structure
    let data = GaussianVectors { vectors };

    // Step 3: Create the `data` folder if it doesn't exist
    let folder_name = format!("data/dimension_{}", d);
    create_dir_all(&folder_name).expect("Failed to create data directory");

    // Step 4: Save the vectors to a binary file using save file
    let file_name = format!("{}/sample_{}.bin", folder_name, n);
    save_file(file_name.as_str(), 0, &data)?;

    println!("Gaussian vectors saved to: {}", file_name);
    Ok(())
}

/// Loads Gaussian vectors from a binary file.
pub fn load_vectors(file_name: &str) -> Result<GaussianVectors, SavefileError> {
    // Use save file's load_file function to deserialize the binary file into GaussianVectors
    load_file(file_name, 0)
}