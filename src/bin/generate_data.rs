use rand_distr::{Distribution, Normal};
use savefile::prelude::*; // For save_file
use savefile_derive::Savefile; // For #[derive(Savefile)]
use std::fs::create_dir_all;
use std::io::{Error, ErrorKind}; // Import only Error and ErrorKind

#[derive(Savefile)] // Derive Savefile for serialization
struct GaussianVectors {
    vectors: Vec<Vec<f64>>,
}

fn main() -> std::io::Result<()> {
    let n = 100; // Number of vectors
    let d = 100; // Dimension of each vector
    let sigma = 1.0; // Standard deviation (sqrt of variance)

    // Define the folder and file name
    let folder_name = format!("data/dimension_{}", d);
    let file_name = format!("{}/sample_{}.bin", folder_name, n);

    // Generate the Gaussian vectors
    let vectors = generate_gaussian_vectors(n, d, sigma)?;

    // Wrap vectors in a struct for serialization
    let data = GaussianVectors { vectors };

    // Create the folder if not present
    create_dir_all(&folder_name)?;

    // Save the file
    save_vectors(&file_name, &data)?;

    println!("Vectors successfully saved to {}", file_name);

    Ok(())
}

/// Generates N random Gaussian vectors with zero mean and variance sigma^2.
fn generate_gaussian_vectors(n: usize, d: usize, sigma: f64) -> std::io::Result<Vec<Vec<f64>>> {
    // Step 1: Define the normal distribution with mean 0 and standard deviation sigma
    let normal = Normal::new(0.0, sigma)
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

/// Save the Gaussian vectors to a binary file.
fn save_vectors(file_name: &str, data: &GaussianVectors) -> std::io::Result<()> {
    save_file(file_name, 0, data)
        .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to save file: {}", e)))
}
