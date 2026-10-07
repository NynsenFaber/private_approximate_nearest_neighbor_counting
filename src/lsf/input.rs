//! Checks on user supplied points and queries.
//!
//! Every structure in the crate works on **unit vectors of one dimension**, and the
//! similarity is their inner product (cosine similarity). The `build` functions of
//! [`crate::anns`] and [`crate::annc`] run [`check_points`] first:
//!
//! * an empty data set, a zero-dimensional point, points of different dimensions,
//!   a `NaN`/infinite coordinate or a zero vector is an **error**, naming the
//!   offending point;
//! * a point that is not of unit length is **normalized**, and one warning on
//!   stderr says how many were.
//!
//! Queries go through [`prepare_query`]: a query of the wrong dimension, with a
//! non-finite coordinate, or equal to zero is a programming error and **panics**; a
//! query that is not of unit length is normalized silently.

use crate::utils::{is_normalized, normalize_vector};
use std::borrow::Cow;

/// Validates a data set and returns the indices of the points that are not of unit
/// length, after printing one warning about them.
///
/// Errors on an empty data set, zero-dimensional points, points whose dimension
/// differs from the first point's, non-finite coordinates and zero vectors.
pub fn check_points(data: &[Vec<f64>]) -> Result<Vec<usize>, String> {
    let d = data.first().ok_or("the data set is empty")?.len();
    if d == 0 {
        return Err("the points have zero dimensions".to_string());
    }
    let mut not_unit = Vec::new();
    for (i, point) in data.iter().enumerate() {
        if point.len() != d {
            return Err(format!(
                "point {i} has dimension {}, but point 0 has dimension {d}: all points \
                 must have the same dimension",
                point.len()
            ));
        }
        if let Some(j) = point.iter().position(|x| !x.is_finite()) {
            return Err(format!(
                "point {i} has the non-finite coordinate {} at index {j}",
                point[j]
            ));
        }
        if !is_normalized(point) {
            if point.iter().all(|&x| x == 0.0) {
                return Err(format!(
                    "point {i} is the zero vector, which has no direction and cannot be \
                     normalized"
                ));
            }
            not_unit.push(i);
        }
    }

    if let Some(&first) = not_unit.first() {
        let norm = data[first].iter().map(|x| x * x).sum::<f64>().sqrt();
        eprintln!(
            "warning: {} of {} points are not unit vectors (e.g. point {first}, norm {norm:.4}); \
             they were normalized, since similarity is the inner product of unit vectors",
            not_unit.len(),
            data.len()
        );
    }
    Ok(not_unit)
}

/// Normalizes the points [`check_points`] reported, in place.
pub fn normalize_points(data: &mut [Vec<f64>], indices: &[usize]) {
    for &i in indices {
        normalize_vector(&mut data[i]);
    }
}

/// [`check_points`] for borrowed data: borrows it when every point is a unit vector,
/// copies and normalizes it otherwise.
pub fn prepare_points(data: &[Vec<f64>]) -> Result<Cow<'_, [Vec<f64>]>, String> {
    let not_unit = check_points(data)?;
    if not_unit.is_empty() {
        return Ok(Cow::Borrowed(data));
    }
    let mut owned = data.to_vec();
    normalize_points(&mut owned, &not_unit);
    Ok(Cow::Owned(owned))
}

/// Checks a query against the data set dimension `d`, normalizing it if needed.
///
/// # Panics
///
/// If the query does not have dimension `d`, has a non-finite coordinate, or is the
/// zero vector.
pub fn prepare_query(query: &[f64], d: usize) -> Cow<'_, [f64]> {
    assert_eq!(
        query.len(),
        d,
        "the query has dimension {}, but the data set has dimension {d}",
        query.len()
    );
    assert!(
        query.iter().all(|x| x.is_finite()),
        "the query has a non-finite coordinate"
    );
    if is_normalized(query) {
        return Cow::Borrowed(query);
    }
    assert!(
        query.iter().any(|&x| x != 0.0),
        "the query is the zero vector, which has no direction"
    );
    let mut owned = query.to_vec();
    normalize_vector(&mut owned);
    Cow::Owned(owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malformed_data_sets_are_rejected() {
        let error = |data: Vec<Vec<f64>>| check_points(&data).unwrap_err();
        assert!(error(vec![]).contains("empty"));
        assert!(error(vec![vec![]]).contains("zero dimensions"));
        assert!(error(vec![vec![1.0, 0.0], vec![1.0]]).contains("point 1 has dimension 1"));
        assert!(error(vec![vec![1.0, 0.0], vec![f64::NAN, 0.0]]).contains("non-finite"));
        assert!(error(vec![vec![1.0, 0.0], vec![0.0, 0.0]]).contains("zero vector"));
    }

    #[test]
    fn test_only_points_off_the_sphere_are_normalized() {
        let data = vec![vec![1.0, 0.0], vec![3.0, 4.0], vec![0.0, 1.0]];
        assert_eq!(check_points(&data).unwrap(), vec![1]);
        let prepared = prepare_points(&data).unwrap();
        assert_eq!(prepared[1], vec![0.6, 0.8]);
        // Unit vectors are left bit-for-bit untouched.
        assert_eq!(prepared[0], data[0]);

        let unit = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(matches!(prepare_points(&unit).unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn test_query_is_normalized() {
        assert_eq!(prepare_query(&[3.0, 4.0], 2).as_ref(), &[0.6, 0.8]);
        assert!(matches!(prepare_query(&[0.6, 0.8], 2), Cow::Borrowed(_)));
    }

    #[test]
    #[should_panic(expected = "the query has dimension 3, but the data set has dimension 2")]
    fn test_query_of_the_wrong_dimension_panics() {
        prepare_query(&[1.0, 0.0, 0.0], 2);
    }

    #[test]
    #[should_panic(expected = "zero vector")]
    fn test_zero_query_panics() {
        prepare_query(&[0.0, 0.0], 2);
    }
}
