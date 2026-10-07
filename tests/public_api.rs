//! The twelve public structures, used the way a downstream crate would use them:
//! only through public paths, on one shared planted data set.

use ann_rust::annc::dp::{
    DpCloseTop1, DpLsfCounter, DpTensorCloseTop1, DpTensorTop1, DpTop1, TruncatedLaplace,
};
use ann_rust::annc::{
    CloseTop1Counter, LsfCounter, TensorCloseTop1Counter, TensorTop1Counter, Top1Counter,
};
use ann_rust::anns::{CloseTop1, LsfIndex, TensorCloseTop1, TensorTop1, Top1};
use ann_rust::data::{generate, GeneratorConfig, PlantConfig, SyntheticDataset};
use ann_rust::lsf::Algorithm;
use ann_rust::utils::dot_product;
use ann_rust::Config;

const ALPHA: f64 = 0.9;
const BETA: f64 = 0.5;

/// 3 000 points in dimension 32; each of the 3 queries owns a cluster of 200
/// points at inner product >= alpha.
fn dataset() -> SyntheticDataset {
    generate(&GeneratorConfig {
        n: 3_000,
        d: 32,
        queries: 3,
        plant: PlantConfig {
            count: 200,
            similarity: ALPHA,
            tightness: 0.999,
        },
        seed: 5,
    })
    .unwrap()
}

/// Small enough for the single factor algorithms, whose default `m` exceeds `n`.
fn config(tensorized: bool) -> Config {
    Config {
        alpha: ALPHA,
        beta: BETA,
        t: tensorized.then_some(2),
        m_sub: Some(if tensorized { 64 } else { 256 }),
        seed: 9,
        ..Config::default()
    }
}

/// Search, then count, then privately count with the structures of algorithm `A`,
/// checking what each layer promises.
fn exercise<A: Algorithm>(
    index: LsfIndex<A>,
    dataset: &SyntheticDataset,
) -> (LsfCounter<A>, DpLsfCounter<A>) {
    let params = index.params().clone();
    assert_eq!(params.algorithm, A::NAME);
    assert_eq!(params.t, if A::TENSORIZED { 2 } else { 1 });
    assert!(index.stored_points() <= dataset.points.len());

    let mut found = 0;
    for query in &dataset.queries {
        let outcome = index.query(query);
        if let Some(i) = outcome.point {
            assert!(dot_product(query, index.point(i)) >= BETA);
            found += 1;
        }
    }
    assert!(found > 0, "{} found no planted point", A::NAME);

    let counter = LsfCounter::from(index);
    assert_eq!(counter.params().m_sub, params.m_sub);
    let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
    let private = counter.release(mechanism, 3);
    for query in &dataset.queries {
        let exact = counter.count(query);
        assert!(exact.count <= counter.stored_points() as u64);
        let noisy = private.query(query);
        // Every summed counter is off by at most A, every dropped one by at most 1 + A.
        let bound = private.error_bound(exact.matched_buckets)
            + mechanism.suppression_threshold() * exact.matched_buckets as f64;
        assert!((noisy.estimate - exact.count as f64).abs() <= bound);
    }
    assert_eq!(
        private.released_buckets() + private.suppressed_buckets(),
        counter.occupied_buckets()
    );
    (counter, private)
}

#[test]
fn top1() {
    let dataset = dataset();
    let index: Top1 = Top1::build(dataset.points.clone(), &config(false)).unwrap();
    let (counter, _private): (Top1Counter, DpTop1) = exercise(index, &dataset);
    // Every point is stored by the Top-1 rule.
    assert_eq!(counter.stored_points(), dataset.points.len());
}

#[test]
fn close_top1() {
    let dataset = dataset();
    let index: CloseTop1 = CloseTop1::build(dataset.points.clone(), &config(false)).unwrap();
    let _: (CloseTop1Counter, DpCloseTop1) = exercise(index, &dataset);

    // The literal Algorithm 4 drops the points no filter accepts.
    let strict = Config {
        fallback_to_argmax: false,
        ..config(false)
    };
    let counter = CloseTop1Counter::build(&dataset.points, &strict).unwrap();
    assert!(counter.stored_points() < dataset.points.len());
}

#[test]
fn tensor_close_top1() {
    let dataset = dataset();
    let index: TensorCloseTop1 =
        TensorCloseTop1::build(dataset.points.clone(), &config(true)).unwrap();
    let (counter, private): (TensorCloseTop1Counter, DpTensorCloseTop1) = exercise(index, &dataset);

    // Building the counter directly, or consuming it, gives the same answers.
    let direct = TensorCloseTop1Counter::build(&dataset.points, &config(true)).unwrap();
    let query = &dataset.queries[0];
    assert_eq!(direct.count(query).count, counter.count(query).count);
    let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0).unwrap();
    let consumed: DpTensorCloseTop1 = direct.into_private(mechanism, 3);
    assert_eq!(
        consumed.query(query).estimate,
        private.query(query).estimate
    );
}

#[test]
fn tensor_top1() {
    let dataset = dataset();
    let index: TensorTop1 = TensorTop1::build(dataset.points.clone(), &config(true)).unwrap();
    let (counter, _private): (TensorTop1Counter, DpTensorTop1) = exercise(index, &dataset);
    assert_eq!(counter.stored_points(), dataset.points.len());
}

#[test]
fn invalid_configurations_are_rejected() {
    let dataset = dataset();
    // Top-1 and CloseTop-1 have a single factor.
    assert!(Top1::build(dataset.points.clone(), &config(true)).is_err());
    assert!(CloseTop1Counter::build(&dataset.points, &config(true)).is_err());
    // beta must lie below alpha.
    let inverted = Config {
        alpha: 0.4,
        beta: 0.7,
        ..config(true)
    };
    assert!(TensorCloseTop1::build(dataset.points.clone(), &inverted).is_err());
    assert!(TensorTop1Counter::build(&[], &config(true)).is_err());
}

#[test]
fn malformed_data_sets_are_errors_and_off_sphere_points_are_normalized() {
    let ragged = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0]];
    let error = TensorCloseTop1::build(ragged, &config(true)).err().unwrap();
    assert!(error.starts_with("point 1 has dimension 2"), "{error}");

    // Points of any length are accepted and stored normalized.
    let points = vec![vec![3.0, 4.0], vec![0.0, 2.0], vec![1.0, 0.0]];
    let index = Top1::build(points, &config(false)).unwrap();
    assert_eq!(index.point(0), &[0.6, 0.8]);
    assert_eq!(index.point(1), &[0.0, 1.0]);
}

#[test]
#[should_panic(expected = "the query has dimension 3, but the data set has dimension 2")]
fn query_of_the_wrong_dimension_panics() {
    let points = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let counter = TensorTop1Counter::build(&points, &config(true)).unwrap();
    counter.count(&[1.0, 0.0, 0.0]);
}
