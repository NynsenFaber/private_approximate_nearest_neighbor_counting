/// Check if the input data is valid.
pub fn check_input(
    data: &Vec<Vec<f64>>,
    alpha: f64,
    beta: f64,
    theta: f64,
) -> Result<(), String> {
    // Validate alpha
    if !(0.0 < alpha && alpha < 1.0) {
        return Err("Invalid value for alpha. Alpha must be in the range (0, 1).".to_string());
    }

    // Validate beta
    if !(0.0 < beta && beta < alpha) {
        return Err("Invalid value for beta. Beta must be in the range (0, alpha).".to_string());
    }

    // Validate theta
    if !(theta > 0.0) {
        return Err("Invalid value for theta. Theta must be positive.".to_string());
    }

    // Validate data is non-empty
    if data.is_empty() {
        return Err("Data cannot be empty.".to_string());
    }

    // Check if all vectors have the same dimension and are normalized
    let d = data[0].len(); // Dimension of the first vector
    if d == 0 {
        return Err("Vectors cannot have zero dimensions.".to_string());
    }

    for (i, vector) in data.iter().enumerate() {
        // Check if all vectors have the same dimension
        if vector.len() != d {
            return Err(format!(
                "Vector at index {} has a different dimension (expected {}, got {}).",
                i,
                d,
                vector.len()
            ));
        }

        // Check if the vector is normalized (sum of squares equals 1)
        let norm = vector.iter().map(|x| x * x).sum::<f64>();
        if (norm - 1.0).abs() > 1e-6 {
            return Err(format!(
                "Vector at index {} is not normalized (norm = {}).",
                i, norm
            ));
        }
    }

    Ok(())
}