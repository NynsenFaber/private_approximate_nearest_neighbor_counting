<!-- Generated from the rustdoc of `src/annc/top1/mod.rs` by `UPDATE_READMES=1 cargo test --test algorithm_readmes`. Edit the doc comment, not this file. -->

# Top-1 counting

`ann_rust::annc::Top1Counter` · [all algorithms](../../../README.md)

`(alpha, beta)`-ANNC with **Top-1**: Algorithm 3 applied to
[`anns::Top1`](../../anns/top1/README.md).

## Construction

Build the Top-1 partition (every point in the bucket of its argmax filter among
`m = ceil(n^{theta / (1 - alpha^2)})`), then replace every bucket by its size.
Only the filters and the non-empty counters are kept.

## Query

Sum the counters of the buckets whose filter has `<a_i, q> >= eta`.

## Guarantees (Lemma 12 with Lemmas 7-8, `n -> infinity`)

With probability at least `2/3`,
`(1 - o(1)) |S ∩ B(q, alpha)| <= ans <= |S ∩ B(q, beta)| + K`, where `K` is the
expected number of far points in the selected buckets: `n^{rho + o(1)}` for the
balanced `theta = rho` (Corollary 10).

## Example: your own data

```rust
use ann_rust::annc::Top1Counter;
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
let counter = Top1Counter::build(&points, &config)?;

// A query has the dimension of the points; it is normalized too.
let query = [1.0, 0.1, 0.0, 0.4];
println!("count: {}", counter.count(&query).count);

// A malformed data set is an error that names the offending point.
let mut broken = points.clone();
broken[2][1] = f64::NAN;
let error = Top1Counter::build(&broken, &config).err().unwrap();
assert_eq!(error, "point 2 has the non-finite coordinate NaN at index 1");
```

## Example: synthetic benchmark data

```rust
use ann_rust::annc::Top1Counter;
use ann_rust::data::{exact_count, generate, GeneratorConfig, PlantConfig};
use ann_rust::Config;

// A cluster of 100 points at inner product >= 0.9 from the query.
let dataset = generate(&GeneratorConfig {
    n: 2_000,
    d: 32,
    queries: 1,
    plant: PlantConfig { count: 100, similarity: 0.9, tightness: 0.99 },
    seed: 1,
})?;
let query = &dataset.queries[0];

let config = Config { alpha: 0.9, beta: 0.5, m_sub: Some(512), ..Config::default() };
let counter = Top1Counter::build(&dataset.points, &config)?;
let answer = counter.count(query).count;
println!("estimate {answer}, truth {}", exact_count(&dataset.points, query, 0.9));
```
