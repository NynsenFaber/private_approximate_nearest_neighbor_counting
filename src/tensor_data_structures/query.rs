use super::top1::Top1;
use crate::utils::{find_close_vector, is_normalized};
use std::collections::HashMap;
use std::io;

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
    // Get indices of Gaussian vectors that meet the threshold
    let indices = search(top1_list, q);

    // If the indices are empty, return None
    if indices.is_empty() {
        println!("Some indices are empty.");
        return Ok(None);
    }
    // Search for a close vector in the hash table
    for i in indices {
        if let Some(vectors) = hash_table.get(&i) {
            if let Some(close_vector) = find_close_vector(q, vectors, beta) {
                if cfg!(test) {
                    println!("Found a close vector! .");
                }
                return Ok(Some(close_vector));
            }
        }
    }

    if cfg!(test) {
        println!("No close vector found.");
    }
    println!("No close vector found.");
    // If no vector meets the `beta` threshold, return None
    Ok(None)
}

fn search(top1_list: &Vec<Top1>, q: &Vec<f64>) -> Vec<String> {
    // Instantiate a collection to store the results
    let mut collection: Vec<Vec<String>> = Vec::new();
    // Iterate over each Top1 structure
    top1_list.iter().for_each(|top1| {
        collection.push(top1.search(q));
    });
    // Create the Cartesian product of the results
    cartesian_product(collection)
}

fn cartesian_product(collection: Vec<Vec<String>>) -> Vec<String> {
    // Instantiate a collection to store the results
    let mut result: Vec<String> = Vec::new();
    // Iterate over each element in the first collection
    for element in collection[0].iter() {
        // If there is only one collection, return the element
        if collection.len() == 1 {
            result.push(element.clone());
        } else {
            // Recursively call the function with the rest of the collections
            for rest in cartesian_product(collection[1..].to_vec()) {
                result.push(format!("{}{}", element, rest));
            }
        }
    }
    result
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
