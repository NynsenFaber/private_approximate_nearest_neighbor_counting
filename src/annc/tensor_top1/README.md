<!-- Generated from the rustdoc of `src/annc/tensor_top1/mod.rs` by `UPDATE_READMES=1 cargo test --test algorithm_readmes`. Edit the doc comment, not this file. -->

# TensorTop-1 counting

`ann_rust::annc::TensorTop1Counter` · [all algorithms](../../../README.md)

`(alpha, beta)`-ANNC with **TensorTop-1**: Algorithm 3 applied to
[`anns::TensorTop1`](../../anns/tensor_top1/README.md).

## Construction

Build the TensorTop-1 partition (`t` factors of `m_sub` filters; a point's
bucket is its `t` argmax filter indices), then replace every bucket by its size.
Every point is counted in exactly one bucket.

## Query

Sum the counters of the buckets in `B_1 x ... x B_t`, visiting only the non-empty
ones.

## Guarantees

Lemma 12 turns any ANN partition into an ANNC structure, so the answer satisfies
`(1 - o(1)) |S ∩ B(q, alpha)| <= ans <= |S ∩ B(q, beta)| + K` as soon as the
partition finds close points with probability `1 - o(1)` and selects `K` far
points in expectation. The paper does not prove those two properties for
TensorTop-1; see [`anns::TensorTop1`](../../anns/tensor_top1/README.md).

## Example: your own data

```rust
use ann_rust::annc::TensorTop1Counter;
use ann_rust::Config;

// Your data set: one `Vec<f64>` per point, all of the same dimension (here 8
// points in dimension 4). Load it from a CSV file, a NumPy export or an
// embedding model; `f32` embeddings convert with `x as f64`.
let points: Vec<Vec<f64>> = vec![
    vec![0.90, 0.10, 0.00, 0.40],
    vec![0.85, 0.15, 0.05, 0.45],
    vec![0.10, 0.80, 0.50, 0.00],
    vec![0.00, 0.20, 0.90, 0.30],
    vec![0.50, 0.50, 0.50, 0.50],
    vec![0.30, 0.00, 0.10, 0.90],
    vec![-0.70, 0.20, 0.10, 0.60],
    vec![0.20, -0.90, 0.30, 0.10],
];
// Most rows are not unit vectors: `build` normalizes them and prints a warning
// on stderr. Normalize them first (`ann_rust::utils::normalize_vector`) to
// silence it.
let config = Config { alpha: 0.9, beta: 0.5, seed: 1, ..Config::default() };
let counter = TensorTop1Counter::build(&points, &config)?;

// A query has the dimension of the points; it is normalized too.
let query = [1.0, 0.1, 0.0, 0.4];
println!("count: {}", counter.count(&query).count);

// A malformed data set is an error that names the offending point.
let mut broken = points.clone();
broken[2][1] = f64::NAN;
let error = TensorTop1Counter::build(&broken, &config).err().unwrap();
assert_eq!(error, "point 2 has the non-finite coordinate NaN at index 1");
```

## Example: synthetic benchmark data

```rust
use ann_rust::annc::TensorTop1Counter;
use ann_rust::data::{exact_count, generate, GeneratorConfig, PlantConfig};
use ann_rust::Config;

let dataset = generate(&GeneratorConfig {
    n: 10_000,
    d: 64,
    queries: 1,
    plant: PlantConfig { count: 500, similarity: 0.7, tightness: 0.95 },
    seed: 1,
})?;
let query = &dataset.queries[0];

let config = Config { alpha: 0.7, beta: 0.4, ..Config::default() };
let counter = TensorTop1Counter::build(&dataset.points, &config)?;
assert_eq!(counter.stored_points(), 10_000);
let answer = counter.count(query).count;
println!("estimate {answer}, truth {}", exact_count(&dataset.points, query, 0.7));
```
