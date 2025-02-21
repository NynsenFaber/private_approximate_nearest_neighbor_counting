use super::top1::Top1;
use crate::utils::{find_close_vector, is_normalized};
use std::collections::HashMap;
use std::io;

/// Query the hash table for a close vector to the query vector.
/// If the query vector is not normalized, an error is returned.
/// If no close vector is found, None is returned and a message is printed.
///
/// Parameters:
/// - `q`: Query vector
/// - `top1_list`: List of Top1 structures
/// - `hash_table`: Hash table
/// - `beta`: Threshold value
///
/// Returns:
/// - `Result<Option<Vec<f64>>, io::Error>`: Close vector or None or an error
///
/// # Example
/// If we have two Top1 structures with  ["0#"] and ["0#", "2#"] as the hashes of the
/// Gaussian vectors that meet the threshold, the Cartesian product will be ["0#0#", "0#2#"] and the
/// query will be searched in the hash table with the keys "0#0#" and "0#2#". If a close vector is found,
/// it will be returned. If no close vector is found, None will be returned.
///
/// # Example
/// If one of the Top1 structures has an empty hash, the Cartesian product will be empty and the query
/// will not be searched in the hash table. In this case, None will be returned.
pub fn query(
    q: &Vec<f64>,
    top1_list: &Vec<Top1>,
    hash_table: &HashMap<String, Vec<Vec<f64>>>,
    beta: f64,
) -> Result<Option<Vec<f64>>, io::Error> {
    // Check if the query vector is normalized
    if !is_normalized(q) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Query vector is not normalized",
        ));
    }

    // Get the cartesian product of the hashes of the Gaussian vectors that meet the threshold
    let indices = search(top1_list, q);

    // If the indices are empty, return None
    if indices.is_empty() {
        println!("Some indices are empty. Query is not possible.");
        return Ok(None);
    }

    // Search for a close vector in the hash table
    for i in indices {
        if let Some(vectors) = hash_table.get(&i) {
            if let Some(close_vector) = find_close_vector(q, vectors, beta) {
                println!("Found a close vector! .");
                return Ok(Some(close_vector));
            }
        }
    }

    println!("No close vector found.");
    // If no vector meets the `beta` threshold, return None
    Ok(None)
}

/// Search for the indices of the Gaussian vectors that meet the threshold in each Top1 structure.
/// The output is the Cartesian product of the indices.
///
/// Parameters:
/// - `top1_list`: List of Top1 structures
/// - `q`: Query vector
///
/// Returns:
/// - `Vec<String>`: Cartesian product of the indices
///
/// # Example
/// If we have two Top1 structures with  ["0#"] and ["0#", "2#"] as the hashes of the
/// Gaussian vectors that meet the threshold, the Cartesian product will be ["0#0#", "0#2#"].
fn search(top1_list: &Vec<Top1>, q: &Vec<f64>) -> Vec<String> {
    // Instantiate a collection to store the results
    let mut collection: Vec<Vec<String>> = Vec::new();
    // Iterate over each Top1 structure
    top1_list.iter().enumerate().for_each(|(i, top1)| {
        let hashes = top1.search(q);
        println!("For Top1 structure {}: {:?}", i, hashes);
        collection.push(hashes);
    });
    // Create the Cartesian product of the results
    cartesian_product(collection)
}

/// Compute the Cartesian product of a collection of collections.
fn cartesian_product(collection: Vec<Vec<String>>) -> Vec<String> {
    // If the collection is empty, return an empty vector
    if collection.is_empty() {
        return vec![];
    }

    // Use fold to accumulate the Cartesian product
    collection.iter().fold(vec!["".to_string()], |acc, set| {
        // For each prefix in the accumulator, append each suffix in the current set
        acc.into_iter()
            .flat_map(|prefix| set.iter().map(move |suffix| format!("{}{}", prefix, suffix)))
            .collect() // Collect the results into a vector
    })
}

/// Test function
#[cfg(test)]
mod tests {
    use super::*;

    // Test cartesian product
    #[test]
    fn test_cartesian_product() {
        let vec1 = vec!["a".to_string(), "b".to_string()];
        let vec2 = vec!["c".to_string(), "d".to_string()];
        let collection = vec![vec1, vec2];
        let result = cartesian_product(collection);
        assert_eq!(
            result,
            vec![
                "ac".to_string(),
                "ad".to_string(),
                "bc".to_string(),
                "bd".to_string()
            ]
        );

        let vec1 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let vec2 = vec!["c".to_string()];
        let collection = vec![vec1, vec2];
        let result = cartesian_product(collection);
        assert_eq!(
            result,
            vec!["ac".to_string(), "bc".to_string(), "cc".to_string()]
        );

        let vec1 = vec!["a".to_string()];
        let collection = vec![vec1];
        let result = cartesian_product(collection);
        assert_eq!(result, vec!["a".to_string()]);

        let vec1 = vec!["a".to_string(), "b".to_string()];
        let vec2 = Vec::<String>::new();
        let vec3 = vec!["c".to_string()];
        let collection = vec![vec1, vec2, vec3];
        let result = cartesian_product(collection);
        assert_eq!(result, Vec::<String>::new());

        let vec1 = vec!["a#".to_string(), "b#".to_string()];
        let vec2 = vec!["c#".to_string()];
        let vec3 = vec!["d#".to_string()];
        let collection = vec![vec1, vec2, vec3];
        let result = cartesian_product(collection);
        assert_eq!(
            result,
            vec![
                "a#c#d#".to_string(),
                "b#c#d#".to_string()
            ]
        );
    }
}
