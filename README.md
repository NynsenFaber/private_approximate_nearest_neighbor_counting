# Approximate Near Neighbour Counting with Differential Privacy

A Rust implementation of **TensorCloseTop-1** (Algorithm 5) from

> Martin Aumüller, Fabrizio Boninsegna, Francesco Silvestri.
> *Differentially Private High-Dimensional Approximate Range Counting, Revisited.*
> FORC 2025. [doi:10.4230/LIPIcs.FORC.2025.15](https://doi.org/10.4230/LIPIcs.FORC.2025.15)

The data structure solves two problems on the unit sphere `S^{d-1}` under inner
product similarity:

* **(α, β)-ANN** (Definition 2) — given a query `q` that has some point at inner
  product `≥ α`, return a point at inner product `≥ β`;
* **(α, β)-ANNC under differential privacy** (Definition 3 + Theorem 13) — return a
  count between `|S ∩ B(q, α)|` and `|S ∩ B(q, β)|`, up to an additive error, while
  the released structure is `(ε, δ)`-differentially private.

Two experiments measure exactly what the theory predicts: the **success rate** of
ANN search, and the **mean absolute error** of private counting.

## Quick start

```bash
# Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

cargo test                                   # 45 unit tests
cargo run --release --bin ann_experiment     # success/failure metric for ANN
cargo run --release --bin dp_annc_experiment # mean absolute error for DP-ANNC
```

Always use `--release`: the debug build is ~18× slower (27.6 s vs 1.5 s on the same
run). Both experiments generate their own synthetic data, finish in a few seconds
with the default arguments, and document every option under `--help`.

## What is implemented

| Paper | Code |
| --- | --- |
| Algorithm 5, TensorCloseTop-1 | [`tensor_data_structures::tensor_close_top1::TensorCloseTop1`](src/tensor_data_structures/tensor_close_top1.rs) |
| Algorithm 4, CloseTop-1 (one factor) | [`tensor_data_structures::close_top1::CloseTop1`](src/tensor_data_structures/close_top1.rs) |
| Definition 11, the partition function `Q` | [`tensor_data_structures::close_top1::FilterSet`](src/tensor_data_structures/close_top1.rs) |
| `search`, the product `B_1 × … × B_t` | [`tensor_data_structures::probe`](src/tensor_data_structures/probe.rs), [`bucket_index`](src/tensor_data_structures/bucket_index.rs) |
| Algorithm 3, ANN → ANNC | `TensorCloseTop1::count` |
| Theorem 13, ANNC → DP-ANNC | [`tensor_data_structures::dp_annc::DpAnnc`](src/tensor_data_structures/dp_annc.rs) |
| Truncated Laplace mechanism (Geng et al.) | [`dp::truncated_laplace`](src/dp/truncated_laplace.rs) |
| Concomitant thresholds `η`, collision band | [`utils::get_threshold`, `utils::collision_band`](src/utils.rs) |


### How the structure works

Construction (`TensorCloseTop1::build`):

1. `t = ⌈log^{1/8}(n) / (1 - α²)⌉` independent factors are created, each with
   `m_sub = ⌈n^{(1/t)·θ/(1-α²)}⌉` Gaussian filters — `t·m_sub = n^{o(1)}` filters in
   total, simulating `m = m_sub^t` buckets.
2. In each factor, a point `x` is assigned to the **first** filter `a` with
   `√(2 log m_sub) - (3/2)·log log m_sub/√(2 log m_sub) ≤ ⟨a, x⟩ ≤ √(2 log m_sub)`.
3. The `t` filter indices form the bucket key of `x`. Every point is stored **at
   most once**, which gives `O(d·n)` space and — decisively for privacy —
   sensitivity `1` for the bucket histogram.

Query: each factor returns `B_i = {j : ⟨a_{i,j}, q⟩ ≥ η}` with
`η = α√(2 log m_sub) - √(2(1-α²) log log m_sub)`; the query then inspects the
buckets of `B_1 × … × B_t`, scanning points (ANN) or summing counters (ANNC).

## Differential privacy

`TensorCloseTop1::release` publishes the counting structure. What leaves the
building is the filters, the thresholds, and a sparse map of noisy counters — no
input point survives the call.

* **Sensitivity 1.** Neighbouring data sets are add/remove-one-point. Since a point
  occupies exactly one bucket, the histograms differ in one coordinate by one.
  (`--sensitivity 2` is available for the substitution relation.)
* **Mechanism.** Truncated Laplace with `b = Δ/ε` and support
  `A = b·ln(1 + (e^ε - 1)/(2δ))`, the optimal `(ε, δ)`-DP additive mechanism. Noise
  is *bounded*, so every released counter is off by at most `A = O(log(1/δ)/ε)` —
  the per-counter error Theorem 13 assumes.
* **Sparse release.** Only non-empty buckets are noised, and a bucket is dropped
  when its noisy value is `≤ 1 + A`. That threshold is what makes releasing the
  *key set* safe: a counter that is `0` on one data set and `1` on a neighbouring
  one never exceeds `1 + A`, so it is suppressed in both cases. The release is
  therefore identical in distribution to "add noise to all `m` counters, then drop
  everything `≤ 1 + A`", which is `(ε, δ)`-DP per coordinate plus post-processing.
* **Query error.** A query sums the counters it hits, so its additive error is at
  most `A` per summed counter. Theorem 13 states this as `A·K` with `K = E[|I(q)|]`;
  in the sparse implementation empty and suppressed buckets contribute exactly
  zero, so the realized bound is `A ×` (number of released counters hit), which the
  experiment reports alongside `|I(q)|`.

All experiments below fix `δ = 10⁻⁶` and vary `ε`; `--delta` changes it.

**Caveats for real deployments.** The noise is drawn from `StdRng` seeded by a
user-supplied number so that experiments are reproducible; a real release must seed
from OS entropy and keep the seed secret. Sampling uses `f64` arithmetic and is
therefore exposed to the floating-point attacks of Mironov (CCS 2012). Each call to
`release` spends a full `(ε, δ)` budget — publishing several releases of the same
data requires composition.

## Experiment 1: ANN success rate

```bash
cargo run --release --bin ann_experiment
```

Every trial is a query `q` that has exactly one planted point at inner product `α`,
in a background of uniformly random unit vectors. The trial **succeeds** if the
query returns a point at inner product `≥ β`, and fails if it returns nothing.
Queries whose data set contains no close point are skipped rather than counted, so
the denominator is always the premise of Definition 2.

Measured on an Apple M1 Pro (8 cores), `d = 128`, `α = 0.7`, `β = 0.4`, 200 trials,
all other parameters from the paper's formulas:

| `n` | `t` | `m_sub` | build | success rate | `\|I(q)\|` | buckets visited | query |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 000 | 3 | 145 | 0.04 s | 0.685 ± 0.033 | 17 455 | 54.6 | 0.094 ms |
| 100 000 | 3 | 502 | 0.94 s | 0.745 ± 0.031 | 318 554 | 278.1 | 0.910 ms |
| 1 000 000 | 3 | 1 741 | 29.2 s | 0.820 ± 0.027 | 5 552 064 | 1 326.4 | 7.300 ms |

`|I(q)|` is the size of the Cartesian product the query covers; "buckets visited"
counts the non-empty ones actually opened, which for search also stops at the first
close point found.

The success rate grows with `n`, as the `1 - o(1)` guarantee of Lemma 17 predicts,
and is still visibly short of 1 at `n = 10^6` — see *Finite-`n` behaviour* below.
In `d = 128` a uniformly random point reaches inner product `0.4` with probability
`≈ 3·10⁻⁶`, so the planted point is essentially the only admissible answer: this is
a strict test, not one that any nearby point can pass.

Useful options: `--n --d --alpha --beta --trials --repeat --seed`, the parameter
overrides `--theta --t --m-sub`, and `--strict`.

### Baseline: linear scan

Every run also times an exact baseline — a single threaded scan of the data set in
storage order, returning the first point at inner product `≥ β` — and reports it
right after the structure's own numbers (`--no-linear` skips it; build time never
enters either measurement):

| `n` | `\|I(q)\|` | ANN query | points scanned | linear query | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 79 | 0.008 ms | 52.5 | 0.006 ms | 0.7× (linear wins) |
| 200 | 107 | 0.008 ms | 99.6 | 0.008 ms | 1.0× |
| 800 | 617 | 0.017 ms | 407.6 | 0.034 ms | 2.0× |
| 10 000 | 17 455 | 0.094 ms | 4 917 | 0.467 ms | 5.0× |
| 100 000 | 318 554 | 0.910 ms | 43 880 | 8.608 ms | 9.5× |
| 1 000 000 | 5 552 064 | 7.300 ms | 317 934 | 70.510 ms | 9.7× |

The **crossover is at `n ≈ 200`** for this `(α, β) = (0.7, 0.4)`, `d = 128` setup.
Past it TensorCloseTop-1 wins by a growing margin, 5–10× in the tested range,
because `|I(q)|` is only the size of the Cartesian product a query *could* touch;
the bucket index opens only the non-empty buckets that actually collide with the
query, so the real work per query grows far slower than `|I(q)|` or `n` — 1 332
points inspected at `n = 10⁶`, against 317 934 scanned by the baseline.

Two things make this comparison fair rather than flattering, and both matter:

* **The planted points are shuffled into the data set.** They are generated after
  the background, so in storage order they would all sit at the very end and the
  scan would have to walk the entire data set before reaching one. That is a worst
  case layout, not a representative one, and it inflated the measured speedup by
  roughly 2× (at `n = 10⁵`: 12.7× before, 9.5× after). `generate` now shuffles once
  with a seeded RNG, so a match sits at a uniformly random position and the scan
  stops after `≈ n/2` points as it should. The structure is indifferent to the
  order — the success rates are bit-for-bit identical either way — so this only
  ever changed the baseline. A unit test pins the property.
* **The baseline is exact.** It always finds a match when one exists, while the
  structure succeeds 74.5% of the time at `n = 10⁵`. The speedup is therefore
  bought with a real accuracy loss, which is the whole point of the `(α, β)`
  relaxation, not a free win.

The crossover also depends on how selective `(α, β)` is: a looser `β` lets the scan
stop earlier without changing the structure's cost much. At `n = 10⁴` with
`β = 0.1`, a uniformly random point already qualifies with probability `≈ 10⁻³` in
`d = 128`, so the scan stops after 7.8 points on average (0.001 ms) and is 18.7×
*faster* than the structure (0.015 ms) — the crossover moves well past `10⁴`. Run
`--beta` at a few values to see this directly.

## Experiment 2: DP-ANNC mean absolute error

```bash
cargo run --release --bin dp_annc_experiment
```

Each query owns a planted cluster of 2 000 points at inner product `≥ α`, mutually
similar at `0.95`, so the quantity to be counted is large enough for a private
answer to be meaningful. The exact answers `|S ∩ B(q, α)|` and `|S ∩ B(q, β)|` are
computed by brute force; the table reports the mean absolute error against the
former, averaged over 20 queries × 5 independent noise draws.

`n = 100 000`, `d = 128`, `α = 0.7`, `β = 0.4`, `δ = 10⁻⁶`, true answer 2 000:

| `ε` | noise `A` | **MAE** | mean estimate | counters summed | `A ×` counters |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 108.70 | 1 398.7 | 613.5 | 2.9 | 319.6 |
| 0.25 | 47.45 | 1 130.4 | 903.2 | 7.1 | 335.0 |
| 0.50 | 25.38 | 915.4 | 1 125.7 | 13.6 | 345.7 |
| 1.00 | 13.66 | 702.9 | 1 359.6 | 25.7 | 351.4 |
| 2.00 | 7.49 | 547.8 | 1 521.7 | 40.4 | 302.4 |
| 4.00 | 4.28 | 431.9 | 1 645.5 | 59.1 | 252.9 |
| 8.00 | 2.64 | 355.9 | 1 744.7 | 82.1 | 216.7 |
| — | non-private (Algorithm 3) | 362.8 | 2 332.6 | 573.8 | — |

Reading the table:

* The MAE decreases monotonically with `ε` and converges to the **non-private**
  error of the same partition (362.8), which is the accuracy ceiling of the LSF
  approximation itself: the filters sweep in ~330 far points per query. At `ε = 8`
  privacy is essentially free; at `ε = 0.1` it costs a factor four in accuracy.
* The dominant privacy cost here is **suppression**, not the added noise: at
  `ε = 0.1` a counter must exceed `1 + A ≈ 110` to be released, so only 2.9 of the
  573.8 non-empty buckets a query hits survive, and the estimate collapses towards
  zero. This is why the private estimate *under*-shoots while the non-private one
  overshoots.
* Because so few counters survive, the realized noise budget (`A ×` counters) stays
  around 300 across the whole range, far below the `A·|I(q)| = 4.4·10⁶` that
  Theorem 13's worst case allows at `ε = 1`.

Useful options: `--n --d --alpha --beta --queries --neighbours --tightness
--epsilons --delta --sensitivity --repeat --seed --strict`, and the same parameter
overrides as above. `--tightness` controls how concentrated the planted cluster is
and therefore how many counters clear the suppression threshold — with a very tight
cluster (`--tightness 0.9999`) the cluster lands in one or two buckets whose
counters dwarf the noise, and the MAE drops to ~300 (309 at `ε = 0.1`, 301 at
`ε = 8`) — better than the non-private answer of the same partition (543), because
suppression discards exactly the singleton buckets that hold far points.

## Synthetic data

`src/data.rs` generates unit vectors uniformly on `S^{d-1}` (normalized Gaussians).
Random unit vectors in high dimension are near-orthogonal, so a random query would
have no near neighbour at all and both experiments would be vacuous; each query
therefore comes with planted points at a prescribed inner product, built as
`x = s·q + √(1-s²)·u` with `u ⊥ q` uniform.

Data sets can be materialized once and shared by both experiments:

```bash
cargo run --release --bin generate_data -- --n 100000 --d 128 --queries 200
cargo run --release --bin ann_experiment -- --data data/dimension_128/sample_100000.bin
```

Generation is seeded and parallel, and reproducible independently of the number of
threads.

## Parameters

| Parameter | Meaning | Default |
| --- | --- | --- |
| `alpha`, `beta` | close / far inner product thresholds, `0 ≤ β < α < 1` | 0.7 / 0.4 |
| `theta` | space-time vs accuracy knob | balanced `ρ = (1-α²)(1-β²)/(1-αβ)²` |
| `t` | concatenation factor | `⌈log^{1/8}(n)/(1-α²)⌉` |
| `m_sub` | filters per factor | `⌈n^{(1/t)·θ/(1-α²)}⌉` |
| `fallback_to_argmax` | keep points that collided with no filter | `true` |

`θ` is the trade-off of Corollary 10: `θ = ρ` balances the number of inspected
buckets against the number of far points swept in; smaller `θ` gives fewer, fatter
buckets (cheaper queries, more far points), larger `θ` the opposite. Note that the
analysis is *dimension-free* — for a Gaussian filter `a`, `(⟨x, a⟩, ⟨q, a⟩)` is
exactly bivariate normal with correlation `⟨x, q⟩` for every `d` — so `d` only
affects how many near neighbours a data set happens to contain.

### Finite-`n` behaviour

The guarantees are asymptotic, and at reachable data set sizes two effects are
worth knowing about; both are measurable with the flags above.

**1. The collision band is wide.** Its width is
`(3/2)·log log m_sub/√(2 log m_sub)`, which vanishes only very slowly. At
`m_sub = 502` the band is `[2.75, 3.53]`, and a point sitting at the lower edge is
found by a query with noticeably smaller probability than one at `√(2 log m_sub)`.
The per-factor success probability is therefore below the asymptotic value, and the
`t` factors multiply it. Larger `m_sub` (larger `θ`, smaller `t`) buys accuracy at
the price of query time: at `n = 10⁵`, `t = 2`, the success rate goes from
0.820 ± 0.027 at `θ = 0.5` to 0.895 ± 0.022 at `θ = ρ`, while `|I(q)|` grows from
1 917 to 535 199 and the build from 0.4 s to 11 s.

**2. Points that collide with no filter.** Lemma 23 bounds this probability by
`m^{-Ω(1)}`, and the hidden constant matters in practice. At `m_sub = 502` the band
is accepted by a single filter with probability `Pr[Z ∈ [2.75, 3.53]] = 0.0028`, so
a point is dropped by a factor with probability `(1 - 0.0028)^502 = 0.25`, and
survives all three factors with probability `0.75³ = 0.42`. Measured at `n = 10⁵`,
`m_sub = 502`, `t = 3`, running with `--strict`:

| mode | points stored | success rate |
| --- | ---: | ---: |
| `--strict` (literal Algorithm 4) | 42.6 % | 0.340 ± 0.034 |
| default (argmax fallback) | 100 % | 0.745 ± 0.031 |

Dropping 25 % of the points *per factor* costs more than half the recall at this
size. The default therefore keeps such a point by assigning it to the filter
maximizing the inner product — the Top-1 rule of Algorithm 2, whose analysis covers
this case asymptotically. This does not affect privacy in any way: the point still
occupies exactly one bucket, so the sensitivity stays 1. Use `--strict` (or
`Config { fallback_to_argmax: false, .. }`) for the literal algorithm.

## Memory overhead

Both experiments print a `memory:` line after the build, breaking the structure's
actual heap footprint down by what it is spent on (measured from allocated
capacities, not lengths, via `TensorCloseTop1::memory_footprint`):

```
memory:      108.67 MiB total = 99.95 MiB raw data + 8.73 MiB overhead
             (1.51 MiB filters, 7.22 MiB bucket index), 8.7% over the raw
             data a linear scan would need
```

`n = 100 000`, `n = 10⁶` (default `d = 128`, `α = 0.7`, `β = 0.4`, `θ = ρ`):

| `n` | raw data | filters | bucket index | overhead | total | overhead / raw data |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 000 | 99.95 MiB | 1.51 MiB | 7.22 MiB | 8.73 MiB | 108.67 MiB | 8.7% |
| 1 000 000 | 999.45 MiB | 5.22 MiB | 72.30 MiB | 77.52 MiB | 1.05 GiB | 7.8% |

`raw data` is what the linear scan baseline above needs and nothing more; it is
the floor every exact method pays and it dominates the total, confirming the
`O(dn)` space bound (Definition 11 / Algorithm 5) empirically. The **overhead is
real but small** — under 9% here — and split two ways:

* **Filters** (`t·m_sub` Gaussian vectors, `n^{o(1)}` of them by construction) —
  negligible at any of the tested sizes, exactly because the paper's whole point
  is that this term does not scale with `n`.
* **Bucket index** — one key (`t` `u32`s) and one point id per stored point,
  which is genuinely necessary: it is what lets a query skip the empty buckets of
  `B_1 × … × B_t` instead of enumerating all `|I(q)|` of them (see *Bucket index*
  below). At `n = 10⁵` this is 7.22 of the 8.73 MiB overhead.


## Implementation notes

* **Bucket index: a prefix tree over sorted keys, built once.** Enumerating
  `B_1 × … × B_t` costs `|I(q)|` lookups, but only the non-empty buckets carry
  information, and with `m = m_sub^t ≫ n` almost the whole product is empty —
  at `n = 10⁵` the product has 318 554 candidate keys, of which only 278 are
  ever occupied. [`BucketIndex`](src/tensor_data_structures/bucket_index.rs)
  avoids ever materializing that product.

  - **Build, once.** `TensorCloseTop1::build` inserts every stored point into a
    `HashMap<BucketKey, Vec<u32>>` — `O(n)` — then `BucketIndex::from_map`
    sorts the resulting `B ≤ n` non-empty entries lexicographically by key —
    `O(B log B)`. `B` is bounded by the number of *stored points*, never by
    `|I(q)|`; the sort runs exactly once, at construction, and every query
    afterwards reuses it.
  - **Query, every time.** A sorted array of length-`t` keys is a prefix tree
    in disguise: a contiguous range that shares the same first `k` components
    *is* the subtree rooted at that prefix, with no pointers or nodes needed
    to represent it. `for_each_match` walks it level by level — at each level
    it merges the query's candidate filter indices for that level against the
    keys in the current range, using binary search (`lower_bound`/
    `upper_bound`) to jump straight past runs of keys that cannot match
    instead of visiting them one by one, then recurses only into the
    sub-ranges that do match. The cost is roughly `O(t · V · log B)`, where
    `V` is the number of buckets actually *visited* — 278 against
    `|I(q)| = 318 554` at `n = 10⁵` — rather than `O(|I(q)|)`. That is a
    13.7× faster query (12.3 ms → 0.90 ms) with identical answers: a unit
    test checks the traversal against the naive enumeration
    ([`probe::ProbeIter`](src/tensor_data_structures/probe.rs)) on random
    inputs, and both visit the matching buckets in lexicographic order.

  The alternative — enumerate every key of `B_1 × … × B_t` and do a hash
  lookup per key — is what the pseudocode literally describes, and it is what
  a first, faithful implementation does. It also throws away exactly the
  saving the tensorization trick was built to give: `m_sub^t` simulated
  buckets from only `t · m_sub` stored filters, read back out one key at a
  time. Sorting the *occupied* buckets once at build time, and walking that
  sorted order at query time instead of the candidate product, is what
  actually realizes the `n^{ρ+o(1)}` query time of Theorem 19, rather than the
  `|I(q)|` of a literal reading of Algorithm 5.
* **Parallelism.** Construction and the brute force ground truth use rayon; queries
  are single threaded so that the reported per-query time is meaningful.
* **Reproducibility.** Everything derives from one seed: filters, data, and noise.
  Re-running an experiment with the same `--seed` reproduces the numbers exactly.
* **Guard rails.** `m_sub` is raised to 16 if the requested value is smaller, since
  the collision band is empty for `m ≤ e^e`; both experiments print a note when that
  happens.

## Repository layout

```
src/
  lib.rs                        module tree
  utils.rs                      inner products, thresholds, sphere sampling, Gaussian tails
  data.rs                       synthetic data sets and brute force ground truth
  cli.rs                        dependency free --key value parsing
  checks.rs                     input validation (used by the legacy structures)
  dp/truncated_laplace.rs       (eps, delta)-DP mechanism and sparse histogram release
  tensor_data_structures/
    close_top1.rs               one CloseTop-1 factor (Algorithm 4) + FilterSet
    tensor_close_top1.rs        TensorCloseTop-1 (Algorithm 5), ANN + exact counting
    dp_annc.rs                  the published private counting structure
    bucket_index.rs             sorted bucket index used by queries
    probe.rs                    bucket keys and naive Cartesian enumeration
    top1.rs, tensor_top1.rs, query.rs      earlier Top-1 based tensorized structure
  simple_data_structures/       earlier non-tensorized Top-1 / CloseTop-1
  bin/
    ann_experiment.rs           success/failure metric
    dp_annc_experiment.rs       mean absolute error metric
    generate_data.rs            writes a data set to disk
    top1.rs, close_top1.rs, tensor_top1.rs   drivers for the earlier structures,
                                             with hard-coded parameters
```

## Tests

```bash
cargo test
```

45 unit tests, covering: the truncated Laplace support/mean and the fact that
singleton buckets are always suppressed; the collision band and the fact that every
stored point really lies inside it; the partition property (no point in two
buckets) that the sensitivity argument depends on; the equivalence of the fast
bucket index with naive Cartesian enumeration; the decomposition of a count into
near and far points; that a private estimate stays within its error bound; and the
geometry of the data generator, including that planted points are shuffled
through the data set rather than parked at its end.
