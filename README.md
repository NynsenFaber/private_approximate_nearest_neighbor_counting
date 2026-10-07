# Approximate Near Neighbour Counting with Differential Privacy

[![CI](https://github.com/NynsenFaber/private_approximate_nearest_neighbor_counting/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/NynsenFaber/private_approximate_nearest_neighbor_counting/actions/workflows/ci.yml)
[![Clippy](https://github.com/NynsenFaber/private_approximate_nearest_neighbor_counting/actions/workflows/clippy.yml/badge.svg?branch=master)](https://github.com/NynsenFaber/private_approximate_nearest_neighbor_counting/actions/workflows/clippy.yml)
[![codecov](https://codecov.io/gh/NynsenFaber/private_approximate_nearest_neighbor_counting/branch/master/graph/badge.svg)](https://codecov.io/gh/NynsenFaber/private_approximate_nearest_neighbor_counting)

A Rust implementation of the data structures in

> Martin Aumüller, Fabrizio Boninsegna, Francesco Silvestri.
> *Differentially Private High-Dimensional Approximate Range Counting, Revisited.*
> FORC 2025. [doi:10.4230/LIPIcs.FORC.2025.15](https://doi.org/10.4230/LIPIcs.FORC.2025.15)

The paper builds a simple locality sensitive filter for near neighbour search on the
unit sphere, shows how to turn it into a counting structure, and releases the counts
with differential privacy. This crate implements four variants of that filter, each
as a search structure, a counting structure and a private counting structure.

## Three problems

All points and queries are unit vectors in `R^d`, and similarity is the inner
product (cosine similarity). Write `B(q, α)` for the points at inner product at
least `α` from `q`, and fix two thresholds `0 ≤ β < α < 1`.

- **ANNS** (approximate near neighbour search). If the data set `S` has a point in
  `B(q, α)`, return some point of `S ∩ B(q, β)`.
- **ANNC** (approximate near neighbour counting). Return a number between
  `|S ∩ B(q, α)|` and `|S ∩ B(q, β)|`.
- **DP-ANNC**. Solve ANNC with a structure that can be published: it is
  `(ε, δ)`-differentially private, so adding or removing one point of `S` barely
  changes what it reveals. The price is an additive error.

## How the algorithms work

### Top-1

Sample `m` random Gaussian vectors, the *filters*. Store each point in the bucket
of the filter it has the largest inner product with. A query opens every bucket
whose filter has inner product at least `η` with it and scans those points. The
threshold `η = α√(2 log m) − √(2(1 − α²) log log m)` comes from the theory of
concomitant order statistics: a point at inner product `α` from `q` sits at a filter
whose inner product with `q` is close to `α√(2 log m)`.

Top-1 needs `m = n^{θ/(1−α²)}` filters, more than the number of points `n`, and
every point is compared with every filter. CloseTop-1 changes the assignment rule
(a point goes to the first filter whose inner product falls in a narrow band), which
makes the analysis hold at every `n` instead of only in the limit.

### Tensorization: TensorTop-1

Instead of one set of `m` filters, use `t` independent sets of `m_sub` filters. Each
point is assigned to one filter per set, and the tuple of the `t` indices is its
bucket. The `t · m_sub` stored filters address `m_sub^t` buckets, so the structure
gets the many small buckets Top-1 needs while storing `n^{o(1)}` filters and each
point once. A query selects candidate filters `B_i` in every set and visits the
buckets of `B_1 × … × B_t`; only the non-empty ones are opened.

TensorTop-1 uses the Top-1 rule in each set and is not in the paper.
TensorCloseTop-1 (Algorithm 5) uses the CloseTop-1 rule and comes with the paper's
guarantees: linear space, `n^{1+o(1)}` construction and `n^{ρ+o(1)}` query time.

### From search to counting to privacy

Every algorithm partitions the data set: each point is in at most one bucket.
Replacing each bucket by its size gives the counting structure (Algorithm 3), and a
query sums the counters of the buckets it selects.

Two facts make the counters cheap to privatize. The filters are drawn without
looking at the data, so they can be published as they are. Adding or removing one
point changes one counter by one, so the sensitivity is 1. The counters are released
with the truncated Laplace mechanism (Geng et al., AISTATS 2020): each gets Laplace
noise of scale `1/ε` truncated to `[−A, A]`, with `A = (1/ε) ln(1 + (e^ε − 1)/(2δ))`.
The noise is bounded, so every released counter is within `A` of the truth. Only
non-empty buckets are noised, and a counter at or below `1 + A` is dropped, which
keeps the set of released buckets private as well.

## The twelve structures

Each page below is generated from the rustdoc of the type and describes the
construction, the guarantees and an example.

| Algorithm | ANNS | ANNC | DP-ANNC |
| --- | --- | --- | --- |
| Top-1 (Algorithm 2) | [`anns::Top1`](src/anns/top1/README.md) | [`annc::Top1Counter`](src/annc/top1/README.md) | [`annc::dp::DpTop1`](src/annc/dp/top1/README.md) |
| CloseTop-1 (Algorithm 4) | [`anns::CloseTop1`](src/anns/close_top1/README.md) | [`annc::CloseTop1Counter`](src/annc/close_top1/README.md) | [`annc::dp::DpCloseTop1`](src/annc/dp/close_top1/README.md) |
| TensorCloseTop-1 (Algorithm 5) | [`anns::TensorCloseTop1`](src/anns/tensor_close_top1/README.md) | [`annc::TensorCloseTop1Counter`](src/annc/tensor_close_top1/README.md) | [`annc::dp::DpTensorCloseTop1`](src/annc/dp/tensor_close_top1/README.md) |
| TensorTop-1 | [`anns::TensorTop1`](src/anns/tensor_top1/README.md) | [`annc::TensorTop1Counter`](src/annc/tensor_top1/README.md) | [`annc::dp::DpTensorTop1`](src/annc/dp/tensor_top1/README.md) |

Use a tensorized algorithm on real data. Top-1 and CloseTop-1 are there for
comparison and need `m_sub` set by hand beyond a few thousand points.

## Installation

The crate is not on crates.io yet. Add it from GitHub (Rust 1.82 or newer):

```bash
cargo add ann_rust --git https://github.com/NynsenFaber/private_approximate_nearest_neighbor_counting
```

or in `Cargo.toml`:

```toml
[dependencies]
ann_rust = { git = "https://github.com/NynsenFaber/private_approximate_nearest_neighbor_counting" }
```

Add `rev = "<commit>"` to pin a version. Run `cargo doc --open` in your project to
browse the API.

## Example: TensorTop-1 on your own data

A data set is a `Vec<Vec<f64>>` with one vector per point, all of the same
dimension. Load it however you like (CSV, NumPy export, an embedding model) and
convert `f32` values with `x as f64`. `build` returns an error that names the point
if the dimensions differ, a coordinate is `NaN` or infinite, or a point is zero.
Vectors that are not unit length are normalized, with a warning on stderr.

```rust
use ann_rust::annc::dp::{DpTensorTop1, TruncatedLaplace};
use ann_rust::annc::TensorTop1Counter;
use ann_rust::anns::TensorTop1;
use ann_rust::Config;

fn main() -> Result<(), String> {
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
    let query = [1.0, 0.1, 0.0, 0.4];

    // Points at cosine similarity >= 0.9 must be found, points >= 0.5 may be used.
    // t, m_sub and theta default to the paper's formulas.
    let config = Config { alpha: 0.9, beta: 0.5, seed: 1, ..Config::default() };

    // ANNS
    let index = TensorTop1::build(points, &config)?;
    if let Some(i) = index.query(&query).point {
        println!("near neighbour: point {i}");
    }

    // ANNC: the same buckets, with counters instead of points.
    let counter = TensorTop1Counter::from(index);
    println!("count: {}", counter.count(&query).count);

    // DP-ANNC: a (1, 1e-6)-differentially private release of the counters.
    // The seed makes the noise reproducible; a real release must use OS entropy.
    let mechanism = TruncatedLaplace::new(1.0, 1e-6, 1.0)?;
    let private: DpTensorTop1 = counter.release(mechanism, 42);
    println!("private count: {:.1}", private.query(&query).estimate);
    Ok(())
}
```

Swap `TensorTop1` for any other algorithm in the table; the API is the same.

## Repository layout

```text
src/anns/      the four search structures, one folder each
src/annc/      the four counting structures, one folder each
src/annc/dp/   the four private ones, and the truncated Laplace mechanism
src/lsf/       filters, partition and bucket index shared by all twelve
experiments/   the experiment binaries
tests/         API tests, and the check that keeps each algorithm's README in sync
```

`cargo test --release` runs the unit, integration and doc tests. The README in each
algorithm folder is generated from the rustdoc of its type. After editing a doc
comment, run `UPDATE_READMES=1 cargo test --test algorithm_readmes`; CI rejects a
pull request with a stale README and regenerates them on every push to `master`.

## Experiments

Two experiments measure what the theory predicts: how often ANN search succeeds,
and the error of private counting. Both generate their own synthetic data
(uniform unit vectors plus planted near neighbours), take `--help`, and accept
`--algorithm tensor-close-top1|tensor-top1|close-top1|top1` (default
`tensor-close-top1`). Always build with `--release`; debug is about 18 times slower.

```bash
cargo run --release --bin ann_experiment
cargo run --release --bin dp_annc_experiment
```

Everything except timings is deterministic given `--seed`. The numbers below were
measured on an Apple M1 Pro with TensorCloseTop-1, `d = 128`, `α = 0.7`, `β = 0.4`
and the paper's parameters. Timings vary by about 10% between runs.

### ANN success rate

Each of 200 queries has exactly one planted point at inner product `α`. A query
succeeds if it returns a point at inner product at least `β`. In `d = 128` a random
point reaches `0.4` with probability about `3·10⁻⁶`, so the planted point is
essentially the only valid answer.

| `n` | `t` | `m_sub` | build | success rate | `\|I(q)\|` | buckets opened | query | linear scan | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 000 | 3 | 145 | 0.05 s | 0.685 ± 0.033 | 17 455 | 54.6 | 0.104 ms | 0.505 ms | 4.9× |
| 100 000 | 3 | 502 | 0.94 s | 0.745 ± 0.031 | 318 554 | 278.1 | 1.020 ms | 8.843 ms | 8.7× |
| 1 000 000 | 3 | 1 741 | 31.9 s | 0.820 ± 0.027 | 5 552 064 | 1 326.4 | 7.196 ms | 71.034 ms | 9.9× |

`|I(q)|` is the number of buckets in `B_1 × … × B_t`; the query opens only the
non-empty ones. The success rate grows with `n`, as the `1 − o(1)` guarantee
predicts. TensorTop-1 with the same parameters succeeds 0.785 ± 0.029 of the time at
`n = 100 000`, at the same query cost.

The baseline is an exact, single-threaded scan that stops at the first point at
inner product `β` or more. It wins below `n ≈ 200`. The planted points are shuffled
into the data set so the scan does not have to reach the end, and the baseline
always finds a match, so the speedup is paid for with the failures in the success
rate column. A looser `β` lets the scan stop earlier: at `n = 10 000` and `β = 0.1`
the scan is 18.7 times faster than the structure.

### DP-ANNC error

Each of 20 queries owns a planted cluster of 2 000 points at inner product at least
`α`. The table reports the mean absolute error against `|S ∩ B(q, α)|` over 20
queries and 5 noise draws, at `n = 100 000` and `δ = 10⁻⁶`.

| `ε` | noise bound `A` | MAE | mean estimate | counters summed | `A` × counters |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 108.70 | 1 398.7 | 613.5 | 2.9 | 319.6 |
| 0.25 | 47.45 | 1 130.4 | 903.2 | 7.1 | 335.0 |
| 0.50 | 25.38 | 915.4 | 1 125.7 | 13.6 | 345.7 |
| 1.00 | 13.66 | 702.9 | 1 359.6 | 25.7 | 351.4 |
| 2.00 | 7.49 | 547.8 | 1 521.7 | 40.4 | 302.4 |
| 4.00 | 4.28 | 431.9 | 1 645.5 | 59.1 | 252.9 |
| 8.00 | 2.64 | 355.9 | 1 744.7 | 82.1 | 216.7 |
| non-private | | 362.8 | 2 332.6 | 573.8 | |

The error falls with `ε` and reaches the non-private error of the same buckets
(362.8), which comes from far points that share buckets with the cluster. Most of
the privacy cost is suppression: at `ε = 0.1` a counter must exceed `1 + A ≈ 110`
to be released, so only 2.9 of the 573.8 buckets a query hits survive and the
estimate falls short. With a tighter cluster (`--tightness 0.9999`) the points
share one or two large buckets and the error drops to about 300 at every `ε`.

### Behaviour at finite n

The guarantees are asymptotic, and two effects show at these sizes.

The collision band of CloseTop-1 is wide (`[2.75, 3.53]` at `m_sub = 502`), so a
point at its lower edge is found less often, and the `t` factors multiply that loss.
A larger `m_sub` buys accuracy with query time: at `n = 100 000` and `t = 2`, the
success rate goes from 0.820 at `θ = 0.5` to 0.895 at `θ = ρ`, while `|I(q)|` grows
from 1 917 to 535 199.

Some points also fall in no band. At `m_sub = 502` a filter accepts a point with
probability 0.0028, so each factor drops a quarter of the points and three factors
keep 42.6% of them. By default such a point goes to its argmax filter instead,
which keeps it in one bucket and leaves privacy unchanged. `--strict` runs the
literal Algorithm 4:

| mode | points stored | success rate |
| --- | ---: | ---: |
| `--strict` | 42.6% | 0.340 ± 0.034 |
| default | 100% | 0.745 ± 0.031 |

The 0.0028 also exposes a typo in the paper. Lemma 24 states a lower bound of
`(2√π/3) log m / m`, which is 0.0146 at `m = 502`, five times the true value.
Propositions 21 and 22 put `√(2π)` in the numerator where the cited Gaussian tail
bound has it in the denominator. The asymptotic statements are unaffected, and the
code computes the exact tail, not the bound.

### Memory

Both experiments print the heap memory of the structure after the build.

| `n` | raw data | filters | bucket index | overhead |
| ---: | ---: | ---: | ---: | ---: |
| 100 000 | 99.95 MiB | 1.51 MiB | 7.22 MiB | 8.7% |
| 1 000 000 | 999.45 MiB | 5.22 MiB | 72.30 MiB | 7.8% |

The raw data is what a linear scan needs. The filters do not grow with `n`; the
bucket index stores one key and one point id per point.

### Reproducing the tables

```bash
cargo run --release --bin ann_experiment     -- --n 10000   --seed 1
cargo run --release --bin ann_experiment     -- --n 100000  --seed 1
cargo run --release --bin ann_experiment     -- --n 1000000 --seed 1   # ~1.5 GB, ~1 min
cargo run --release --bin dp_annc_experiment -- --seed 1
```

`generate_data` writes a data set to disk so both experiments can share it:

```bash
cargo run --release --bin generate_data -- --n 100000 --d 128 --queries 200
cargo run --release --bin ann_experiment -- --data data/dimension_128/sample_100000.bin
```
