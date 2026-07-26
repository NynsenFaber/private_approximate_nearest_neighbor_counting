//! Success/failure experiment for `(alpha, beta)`-ANN with TensorCloseTop-1.
//!
//! The data set consists of vectors drawn uniformly from the unit sphere plus, for
//! every query, one planted point at inner product exactly `alpha` from it. A trial
//! *succeeds* when the query returns a point at inner product at least `beta`. The
//! reported success rate estimates the `1 - o(1)` probability of Lemma 17.
//!
//! Run `cargo run --release --bin ann_experiment -- --help` for the options.

use ann_rust::cli::Args;
use ann_rust::data::{
    best_similarity, exact_count, generate, linear_search_first, load, GeneratorConfig, PlantConfig,
};
use ann_rust::tensor_data_structures::tensor_close_top1::{Config, TensorCloseTop1};
use ann_rust::utils::dot_product;
use std::process::exit;
use std::time::Instant;

const OPTIONS: &[&str] = &[
    "help",
    "n",
    "d",
    "alpha",
    "beta",
    "trials",
    "neighbours",
    "tightness",
    "theta",
    "t",
    "m-sub",
    "strict",
    "repeat",
    "seed",
    "data",
    "no-linear",
];

const HELP: &str = "\
Success/failure experiment for (alpha, beta)-ANN with TensorCloseTop-1.

Options (defaults in brackets):
  --n <usize>          number of points in the data set [100000]
  --d <usize>          dimension [128]
  --alpha <f64>        close threshold, points at inner product >= alpha must be found [0.7]
  --beta <f64>         far threshold, points at inner product >= beta may be returned [0.4]
  --trials <usize>     number of queries, each with its own planted neighbour [200]
  --neighbours <usize> planted points per query [1]
  --tightness <f64>    mutual similarity of the planted points, 0 = independent [0.0]
  --theta <f64>        space/time knob; default is the balanced rho
  --t <usize>          concatenation factor; default ceil(log^{1/8}(n) / (1 - alpha^2))
  --m-sub <usize>      filters per factor; default ceil(n^{(1/t) theta / (1 - alpha^2)})
  --strict             drop points that collide with no filter (literal Algorithm 4)
  --repeat <usize>     independent rebuilds of the structure, results are pooled [1]
  --seed <u64>         master seed [1]
  --data <path>        use a data set written by generate_data instead of a fresh one
  --no-linear          skip the linear scan baseline (it is O(n) per query)
";

fn main() {
    if let Err(message) = run() {
        eprintln!("error: {message}");
        exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse(OPTIONS)?;
    if args.has("help") {
        println!("{HELP}");
        return Ok(());
    }

    let n: usize = args.get("n", 100_000)?;
    let d: usize = args.get("d", 128)?;
    let alpha: f64 = args.get("alpha", 0.7)?;
    let beta: f64 = args.get("beta", 0.4)?;
    let trials: usize = args.get("trials", 200)?;
    let neighbours: usize = args.get("neighbours", 1)?;
    let tightness: f64 = args.get("tightness", 0.0)?;
    let repeat: usize = args.get("repeat", 1)?;
    let seed: u64 = args.get("seed", 1)?;
    let run_linear = !args.flag("no-linear", false)?;

    let config = Config {
        alpha,
        beta,
        theta: args.get_optional("theta")?,
        t: args.get_optional("t")?,
        m_sub: args.get_optional("m-sub")?,
        fallback_to_argmax: !args.flag("strict", false)?,
        seed,
    };

    let mut successes = 0usize;
    let mut queries_run = 0usize;
    let mut skipped = 0usize;
    let mut probed_buckets = 0f64;
    let mut matched_buckets = 0f64;
    let mut inspected_points = 0f64;
    let mut query_time = 0f64;
    let mut linear_time = 0f64;
    let mut linear_scanned = 0f64;
    let mut stored_fraction = 0f64;
    let mut context_near = 0f64;
    let mut context_far = 0f64;

    for round in 0..repeat {
        let round_seed = seed.wrapping_add(round as u64);
        let generator = GeneratorConfig {
            n,
            d,
            queries: trials,
            plant: PlantConfig {
                count: neighbours,
                similarity: alpha,
                tightness,
            },
            seed: round_seed,
        };
        let dataset = match args.get_string("data") {
            Some(path) => {
                println!(
                    "[round {}/{}] loading the data set from {path}...",
                    round + 1,
                    repeat
                );
                load(path).map_err(|e| e.to_string())?
            }
            None => {
                println!(
                    "[round {}/{}] generating {} points in dimension {} with {} planted \
                     neighbour(s) per query...",
                    round + 1,
                    repeat,
                    n,
                    d,
                    neighbours
                );
                generate(&generator)?
            }
        };
        let n = dataset.points.len();

        let build_start = Instant::now();
        let structure = TensorCloseTop1::build(
            dataset.points.clone(),
            &Config {
                seed: round_seed,
                ..config.clone()
            },
        )?;
        let build_seconds = build_start.elapsed().as_secs_f64();

        if round == 0 {
            println!("\nParameters\n----------\n{}\n", structure.params.summary());
            if structure.params.m_sub_was_clamped {
                println!(
                    "note: m_sub was raised to the smallest usable value; the requested \
                     value was too small for the collision band to exist\n"
                );
            }
        }
        println!(
            "[round {}/{}] built in {:.2}s: {} of {} points stored ({:.1}%), {} non-empty buckets",
            round + 1,
            repeat,
            build_seconds,
            structure.stored_points(),
            n,
            100. * structure.stored_points() as f64 / n as f64,
            structure.occupied_buckets()
        );
        if round == 0 {
            println!("memory:         {}", structure.memory_footprint().summary());
            // How many points a query may legitimately be answered with, measured on
            // the data set actually under test rather than on a freshly generated one.
            let sample = &dataset.queries[0];
            context_near = exact_count(&dataset.points, sample, alpha) as f64;
            context_far = exact_count(&dataset.points, sample, beta) as f64;
        }
        stored_fraction += structure.stored_points() as f64 / n as f64;

        for query in &dataset.queries {
            // A trial is only meaningful if the data set really contains a point at
            // inner product >= alpha: that is the premise of Definition 2.
            if best_similarity(&dataset.points, query) < alpha - 1e-9 {
                skipped += 1;
                continue;
            }
            let start = Instant::now();
            let outcome = structure.query(query);
            query_time += start.elapsed().as_secs_f64();
            queries_run += 1;
            probed_buckets += outcome.probed_buckets as f64;
            matched_buckets += outcome.matched_buckets as f64;
            inspected_points += outcome.inspected_points as f64;
            if let Some(point) = outcome.point {
                let similarity = dot_product(query, structure.point(point));
                assert!(
                    similarity >= beta,
                    "the structure returned a point at inner product {similarity} < beta"
                );
                successes += 1;
            }

            if run_linear {
                // Naive baseline: a single threaded scan in storage order, timed
                // separately from the structure so build cost never enters either
                // number. It is exact, so it is expected to always find a match
                // among these trials (a close point exists by construction).
                let linear_start = Instant::now();
                let hit = linear_search_first(&dataset.points, query, beta);
                linear_time += linear_start.elapsed().as_secs_f64();
                let index = hit.expect("a point at inner product >= beta must exist");
                linear_scanned += (index + 1) as f64;
            }
        }
    }

    if queries_run == 0 {
        return Err("no query had a close point in the data set".to_string());
    }
    let rate = successes as f64 / queries_run as f64;
    // Standard error of a binomial proportion.
    let standard_error = (rate * (1. - rate) / queries_run as f64).sqrt();

    println!("\nResults\n-------");
    println!("trials:         {queries_run} (queries with a close point in the data set)");
    if skipped > 0 {
        println!("skipped:        {skipped} queries had no point at inner product >= alpha");
    }
    println!("successes:      {successes}/{queries_run}");
    println!("success rate:   {:.4} (+/- {:.4})", rate, standard_error);
    println!("failures:       {}", queries_run - successes);
    println!(
        "points stored:  {:.2}% (average over {} build(s))",
        100. * stored_fraction / repeat as f64,
        repeat
    );
    println!(
        "per query:      |I(q)| = {:.0} buckets, {:.1} of them non-empty, {:.1} points \
         inspected, {:.3} ms",
        probed_buckets / queries_run as f64,
        matched_buckets / queries_run as f64,
        inspected_points / queries_run as f64,
        1000. * query_time / queries_run as f64
    );

    if run_linear {
        let mean_ann_ms = 1000. * query_time / queries_run as f64;
        let mean_linear_ms = 1000. * linear_time / queries_run as f64;
        println!("\nLinear scan baseline (exact, single threaded, build time excluded)");
        println!(
            "per query:      {:.1} points scanned, {:.3} ms",
            linear_scanned / queries_run as f64,
            mean_linear_ms
        );
        if mean_ann_ms < mean_linear_ms {
            println!(
                "comparison:     TensorCloseTop1 is {:.1}x faster than the linear scan",
                mean_linear_ms / mean_ann_ms
            );
        } else {
            println!(
                "comparison:     the linear scan is {:.1}x faster than TensorCloseTop1 at this n",
                mean_ann_ms / mean_linear_ms
            );
        }
    }

    println!(
        "context:        a query has {:.1} point(s) at inner product >= alpha and {:.1} at >= beta",
        context_near, context_far
    );

    Ok(())
}
