//! Mean absolute error experiment for DP-ANNC with TensorCloseTop-1 and the
//! truncated Laplace mechanism.
//!
//! For every query the exact answer `|S ∩ B(q, alpha)|` is computed by brute force
//! and compared with the private estimate. Three quantities are reported per
//! privacy budget:
//!
//! * `MAE`      - mean absolute error against `|S ∩ B(q, alpha)|`;
//! * `interval` - mean distance from the interval `[|S ∩ B(q, alpha)|, |S ∩ B(q, beta)|]`
//!                that Definition 3 declares correct (0 means the answer is valid);
//! * `bound`    - the worst case `A * |I(q)|` additive error of Theorem 13.
//!
//! Run `cargo run --release --bin dp_annc_experiment -- --help` for the options.

use ann_rust::cli::Args;
use ann_rust::data::{exact_count, generate, load, GeneratorConfig, PlantConfig};
use ann_rust::dp::truncated_laplace::TruncatedLaplace;
use ann_rust::tensor_data_structures::tensor_close_top1::{Config, TensorCloseTop1};
use std::process::exit;
use std::time::Instant;

const OPTIONS: &[&str] = &[
    "help",
    "n",
    "d",
    "alpha",
    "beta",
    "queries",
    "neighbours",
    "tightness",
    "epsilons",
    "delta",
    "sensitivity",
    "theta",
    "t",
    "m-sub",
    "strict",
    "repeat",
    "seed",
    "data",
];

const HELP: &str = "\
Mean absolute error experiment for DP-ANNC with TensorCloseTop-1 and truncated Laplace noise.

Options (defaults in brackets):
  --n <usize>          number of points in the data set [100000]
  --d <usize>          dimension [128]
  --alpha <f64>        close threshold [0.7]
  --beta <f64>         far threshold [0.4]
  --queries <usize>    number of queries [20]
  --neighbours <usize> planted points per query, the quantity to be counted [2000]
  --tightness <f64>    mutual similarity of the planted points [0.95]
  --epsilons <list>    comma separated privacy budgets [0.1,0.25,0.5,1,2,4,8]
  --delta <f64>        privacy parameter delta, must be > 0 [1e-6]
  --sensitivity <f64>  1 for add/remove neighbouring, 2 for substitution [1]
  --theta <f64>        space/time knob; default is the balanced rho
  --t <usize>          concatenation factor; default ceil(log^{1/8}(n) / (1 - alpha^2))
  --m-sub <usize>      filters per factor; default ceil(n^{(1/t) theta / (1 - alpha^2)})
  --strict             drop points that collide with no filter (literal Algorithm 4)
  --repeat <usize>     independent noise draws per budget [5]
  --seed <u64>         master seed [1]
  --data <path>        use a data set written by generate_data instead of a fresh one
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
    let queries: usize = args.get("queries", 20)?;
    let neighbours: usize = args.get("neighbours", 2000)?;
    let tightness: f64 = args.get("tightness", 0.95)?;
    let epsilons: Vec<f64> =
        args.get_list("epsilons", vec![0.1, 0.25, 0.5, 1., 2., 4., 8.])?;
    let delta: f64 = args.get("delta", 1e-6)?;
    let sensitivity: f64 = args.get("sensitivity", 1.0)?;
    let repeat: usize = args.get("repeat", 5)?;
    let seed: u64 = args.get("seed", 1)?;

    let config = Config {
        alpha,
        beta,
        theta: args.get_optional("theta")?,
        t: args.get_optional("t")?,
        m_sub: args.get_optional("m-sub")?,
        fallback_to_argmax: !args.flag("strict", false)?,
        seed,
    };

    let dataset = match args.get_string("data") {
        Some(path) => {
            println!("loading the data set from {path}...");
            load(path).map_err(|e| e.to_string())?
        }
        None => {
            println!(
                "generating {n} points in dimension {d} with {neighbours} planted \
                 neighbours per query..."
            );
            generate(&GeneratorConfig {
                n,
                d,
                queries,
                plant: PlantConfig {
                    count: neighbours,
                    similarity: alpha,
                    tightness,
                },
                seed,
            })?
        }
    };
    let n = dataset.points.len();

    // Ground truth: the interval [|S ∩ B(q, alpha)|, |S ∩ B(q, beta)|] of Definition 3.
    println!("computing the exact answers by brute force...");
    let truth: Vec<(u64, u64)> = dataset
        .queries
        .iter()
        .map(|query| {
            (
                exact_count(&dataset.points, query, alpha),
                exact_count(&dataset.points, query, beta),
            )
        })
        .collect();

    let build_start = Instant::now();
    let structure = TensorCloseTop1::build(dataset.points.clone(), &config)?;
    println!("\nParameters\n----------\n{}\n", structure.params.summary());
    if structure.params.m_sub_was_clamped {
        println!(
            "note: m_sub was raised to the smallest usable value; the requested value \
             was too small for the collision band to exist\n"
        );
    }
    println!(
        "built in {:.2}s: {} of {} points stored ({:.1}%), {} non-empty buckets",
        build_start.elapsed().as_secs_f64(),
        structure.stored_points(),
        n,
        100. * structure.stored_points() as f64 / n as f64,
        structure.occupied_buckets()
    );
    println!("memory:      {}", structure.memory_footprint().summary());

    // Non private baseline: the exact answer of the same partition (Algorithm 3).
    let mut baseline_error = 0f64;
    let mut baseline_interval_error = 0f64;
    let mut baseline_estimate = 0f64;
    let mut probed = 0f64;
    let mut matched = 0f64;
    for (query, &(near, far)) in dataset.queries.iter().zip(truth.iter()) {
        let outcome = structure.count(query);
        probed += outcome.probed_buckets as f64;
        matched += outcome.matched_buckets as f64;
        baseline_estimate += outcome.count as f64;
        baseline_error += (outcome.count as f64 - near as f64).abs();
        baseline_interval_error += interval_distance(outcome.count as f64, near, far);
    }
    let queries_run = dataset.queries.len() as f64;
    let mean_probed = probed / queries_run;
    let mean_matched = matched / queries_run;
    let mean_near = truth.iter().map(|&(near, _)| near as f64).sum::<f64>() / queries_run;
    let mean_far = truth.iter().map(|&(_, far)| far as f64).sum::<f64>() / queries_run;

    println!("\nGround truth\n------------");
    println!("mean |S ∩ B(q, alpha)|      = {mean_near:.1}");
    println!("mean |S ∩ B(q, beta)|       = {mean_far:.1}");
    println!("mean buckets probed         = {mean_probed:.1}");
    println!("mean non-empty buckets hit  = {mean_matched:.1}");

    println!("\nNon private counting (Algorithm 3, the accuracy ceiling of the partition)");
    println!(
        "mean answer = {:.1}, MAE = {:.1}, interval error = {:.1}",
        baseline_estimate / queries_run,
        baseline_error / queries_run,
        baseline_interval_error / queries_run
    );

    println!("\nDP-ANNC with truncated Laplace (delta = {delta:.1e}, sensitivity = {sensitivity}, {repeat} noise draws per budget)");
    println!(
        "{:>8}  {:>9}  {:>9}  {:>12}  {:>10}  {:>9}  {:>10}",
        "epsilon", "noise A", "MAE", "interval err", "mean est", "counters", "A x counters"
    );
    println!("{}", "-".repeat(80));

    for epsilon in epsilons {
        let mechanism = TruncatedLaplace::new(epsilon, delta, sensitivity)?;
        let mut absolute_error = 0f64;
        let mut interval_error = 0f64;
        let mut estimate_sum = 0f64;
        let mut matched_sum = 0f64;
        let mut samples = 0f64;

        for draw in 0..repeat {
            // The partition is data independent and stays fixed; only the noise is
            // redrawn, which is what averaging over `repeat` releases measures.
            let private =
                structure.release(mechanism, seed.wrapping_add(1000).wrapping_add(draw as u64));
            for (query, &(near, far)) in dataset.queries.iter().zip(truth.iter()) {
                let outcome = private.query(query);
                absolute_error += (outcome.estimate - near as f64).abs();
                interval_error += interval_distance(outcome.estimate, near, far);
                estimate_sum += outcome.estimate;
                matched_sum += outcome.matched_buckets as f64;
                samples += 1.;
            }
        }

        println!(
            "{:>8.2}  {:>9.2}  {:>9.1}  {:>12.1}  {:>10.1}  {:>9.1}  {:>10.1}",
            epsilon,
            mechanism.bound,
            absolute_error / samples,
            interval_error / samples,
            estimate_sum / samples,
            matched_sum / samples,
            mechanism.bound * matched_sum / samples,
        );
    }

    println!(
        "\nMAE is measured against |S ∩ B(q, alpha)|. 'interval err' is the distance from the\n\
         interval [|S ∩ B(q, alpha)|, |S ∩ B(q, beta)|] that Definition 3 accepts, so 0 means the\n\
         private answer is a valid (alpha, beta)-ANNC answer. 'counters' is the number of released\n\
         counters a query actually summed: empty and suppressed buckets contribute no noise, so\n\
         'A x counters' is the realized worst case noise, while the bound of Theorem 13 uses all\n\
         {mean_probed:.0} probed buckets and reads A x |I(q)| = {:.3e} at epsilon = 1.",
        TruncatedLaplace::new(1.0, delta, sensitivity)?.bound * mean_probed
    );

    Ok(())
}

/// Distance of `estimate` from the interval `[near, far]` of admissible answers.
fn interval_distance(estimate: f64, near: u64, far: u64) -> f64 {
    if estimate < near as f64 {
        near as f64 - estimate
    } else if estimate > far as f64 {
        estimate - far as f64
    } else {
        0.
    }
}
