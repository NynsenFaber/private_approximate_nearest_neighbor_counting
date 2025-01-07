mod generate_data;

fn main() {
    let n = 100; // Number of vectors
    let d = 10;  // Dimension of each vector
    let sigma = 1.0; // Standard deviation (sqrt of variance)

    // Generate and save Gaussian vectors
    if let Err(err) = generate_data::generate_and_save_gaussian_vectors(n, d, sigma) {
        eprintln!("Error generating Gaussian vectors: {}", err);
    }
    // Load vectors from file
    let file_name = format!("data/dimension_{}/sample_{}.bin", d, n);
    match generate_data::load_vectors(file_name.as_str()) {
        Ok(vectors) => {
            println!("Loaded {} vectors of dimension {}: {:?}",
                     vectors.vectors.len(), vectors.vectors[0].len(), vectors.vectors[0]);
        },
        Err(err) => eprintln!("Error loading vectors: {}", err),
    }
}