# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The full paper is in `documents/` so read it if you do not fell 100% sure of what you are implementing.

Rust implementation of TensorCloseTop-1 (Algorithm 5) from Aumüller, Boninsegna, Silvestri, *Differentially Private High-Dimensional Approximate Range Counting, Revisited* (FORC 2025): (α, β)-ANN search and (ε, δ)-DP approximate near neighbour counting on the unit sphere. Crate name is `ann_rust`. MSRV 1.82 (`rust-version` in `Cargo.toml`, enforced by CI).

## Commands

Always use `--release` for tests and experiments: debug is ~18× slower and the tests are arithmetic-heavy.

```bash
cargo test --release --all-targets           # unit tests
cargo test --release --doc                   # doc test in lib.rs
cargo test --release <name_substring>        # single test, e.g. `partition`
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

CI builds with `RUSTFLAGS=-D warnings`, so any compiler warning fails the build. Workflows: `ci.yml` (tests on Linux/macOS/MSRV/beta, rustfmt + docs, coverage via `cargo llvm-cov` uploaded to Codecov, experiment smoke runs, README regeneration) and `clippy.yml` (clippy alone, for its README badge).

Experiments (each has `--help`, and `--algorithm` to pick one of the four; small sizes below match the CI smoke job):

```bash
cargo run --release --bin ann_experiment -- --n 2000 --d 32 --trials 20 --seed 1
cargo run --release --bin dp_annc_experiment -- --n 2000 --d 32 --queries 5 --neighbours 200 --epsilons 1,8 --repeat 2 --seed 1
cargo run --release --bin generate_data -- --n 2000 --d 32 --queries 5 --out data/ci.bin
cargo run --release --bin ann_experiment -- --data data/ci.bin --no-linear
```

`data/` is gitignored. The README tables use `n` up to 10^6 (≈1.5 GB, ~1 min); do not run those casually.

## Architecture

Four algorithms (Top1 = Alg. 2, CloseTop1 = Alg. 4, TensorCloseTop1 = Alg. 5, TensorTop1 = not in paper), each in three layers. One generic implementation per layer; the algorithm is a zero-sized marker type parameter (`lsf::algorithms`), and the public names are type aliases.

| Layer | Generic type | Aliases | Folder |
| --- | --- | --- | --- |
| ANN search | `anns::LsfIndex<A>` | `Top1`, `CloseTop1`, `TensorCloseTop1`, `TensorTop1` | `src/anns/` |
| ANNC (Algorithm 3) | `annc::LsfCounter<A>` | `Top1Counter`, ... | `src/annc/` |
| DP-ANNC (Theorem 13) | `annc::dp::DpLsfCounter<A>` | `DpTop1`, ... | `src/annc/dp/` |

Pipeline: `LsfIndex::build` → `LsfCounter::from(index)` (drops points) or `LsfCounter::build` → `counter.release(mechanism, seed)` / `into_private` → `DpLsfCounter`.

- `src/lsf/`: shared core. `algorithms.rs` markers + `Algorithm` trait (`NAME`, `ASSIGNMENT` Top1/CloseTop1 rule, `TENSORIZED`). `config.rs` `Config` (re-exported at crate root) and `Parameters::resolve::<A>` (single-factor algorithms force `t = 1`, so `m_sub` is their `m`; `m_sub` clamped to ≥ 16). `filters.rs` `FilterSet`, `assign_top1` (argmax), `assign_close_top1` (first filter in band, optional argmax fallback). `partition.rs` `Partition::build::<A>`: the one construction all twelve structures use.
- `lsf/input.rs`: every public `build` runs `check_points` (errors: empty, ragged dimensions, non-finite, zero vector; non-unit points normalized with one stderr warning); every public query entry runs `prepare_query` (panics on wrong dimension/non-finite/zero, normalizes otherwise). Already-unit vectors are left bit-for-bit untouched, which keeps the README numbers reproducible.
- `lsf/bucket_index.rs`: occupied bucket keys sorted once; `for_each_match` walks them as an implicit prefix tree. `lsf/probe.rs`: `ProbeIter`, the naive Cartesian enumeration, is the test oracle for it.
- Each algorithm folder (`anns/top1/`, `annc/dp/tensor_top1/`, ...) has a `mod.rs` that holds only the alias, its rustdoc (construction, query, guarantees, example) and tests. Shared properties are checked by `assert_ann_contract` / `assert_count_contract` / `assert_dp_contract` (in `index.rs`, `counter.rs`, `private_counter.rs`), called from each algorithm's tests; `src/test_support.rs` holds fixtures. `tests/public_api.rs` exercises all twelve aliases through public paths.
- Each algorithm folder's `README.md` is generated from the alias's rustdoc by `tests/algorithm_readmes.rs`. Never edit those READMEs by hand: edit the doc comment, then run `UPDATE_READMES=1 cargo test --test algorithm_readmes` (the test fails when a README is stale; CI regenerates and commits on push to master). The top-level `README.md` is hand-written and its Rust example runs as a doctest (`ReadmeDoctests` in `lib.rs`).
- `annc/dp/truncated_laplace.rs`: mechanism (`b = Δ/ε`, support `A`) and `privatize_histogram` (noise non-empty buckets only, suppress `≤ 1 + A`).
- `data.rs`: synthetic data with planted neighbours, brute-force ground truth, linear-scan baseline, savefile `save`/`load`. `utils.rs`: thresholds, `normal_sf`, seeded sampling, `derive_seed`.
- `experiments/`: the three binaries (registered as `[[bin]]` in `Cargo.toml`). `experiments/common/` holds the `--key value` parser and shared option handling; `--algorithm tensor-close-top1|tensor-top1|close-top1|top1` picks the algorithm (generic `experiment::<A>()`). Single-factor algorithms need `--m-sub` on large `n`: their default `m = n^{θ/(1-α²)}` exceeds `n`, and the experiments refuse > 4 GiB of filters.

## Invariants to preserve

- **Each point occupies at most one bucket.** The privacy proof (sensitivity 1) depends on it, and a test checks the partition property. Any change to bucket assignment must keep this.
- η and the collision band come from the **per-factor** `m_sub`, never from `m_sub^t` (Algorithm 5 line 9). A test pins this.
- **Determinism:** every random draw (filters, data, noise) derives from one master seed through `utils::derive_seed`. Results must not depend on rayon thread count. Use a new stream id for any new randomness; never use `thread_rng`.
- Construction and ground truth are parallel (rayon). Queries stay single-threaded so per-query timings stay meaningful.
- Points and queries are unit vectors; similarity is the inner product; `log` means the natural log.
- The top-level README holds the measured experiment numbers. Update them when behaviour or output changes.
