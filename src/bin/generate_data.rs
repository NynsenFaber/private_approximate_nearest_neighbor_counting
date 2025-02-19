use savefile::prelude::*; // For save_file
use savefile_derive::Savefile; // For #[derive(Savefile)]
use std::fs::create_dir_all;
use std::io::{Error, ErrorKind}; // Import only Error and ErrorKind

use ann_rust::utils::{generate_normal_gaussian_vectors, normalize_vector}; // Import generate_gaussian_vectors

#[derive(Savefile)] // Derive Savefile for serialization
struct GaussianVectors {
    vectors: Vec<Vec<f64>>,
}

fn main() -> std::io::Result<()> {
    let n = 1000; // Number of vectors
    let d = 10000; // Dimension of each vector

    // Define the folder and file name
    let folder_name = format!("data/dimension_{}", d);
    let file_name = format!("{}/sample_{}.bin", folder_name, n);

    // Generate the Gaussian vectors
    let mut vectors = generate_normal_gaussian_vectors(n, d).unwrap();

    // Normalize the vectors
    for vector in vectors.iter_mut() {
        normalize_vector(vector);
    }

    // Wrap vectors in a struct for serialization
    let data = GaussianVectors { vectors };

    // Create the folder if not present
    create_dir_all(&folder_name)?;

    // Save the file
    save_vectors(&file_name, &data)?;

    println!("Vectors successfully saved to {}", file_name);

    Ok(())
}

/// Save the Gaussian vectors to a binary file.
fn save_vectors(file_name: &str, data: &GaussianVectors) -> std::io::Result<()> {
    save_file(file_name, 0, data)
        .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to save file: {}", e)))
}
