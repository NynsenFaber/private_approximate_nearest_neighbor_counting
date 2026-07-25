//! Generates a synthetic data set of random unit vectors (with optional planted
//! near neighbours) and stores it on disk, so that several experiments can share
//! exactly the same input.
//!
//! Run `cargo run --release --bin generate_data -- --help` for the options.

use ann_rust::cli::Args;
use ann_rust::data::{generate, save, GeneratorConfig, PlantConfig};
use std::fs::create_dir_all;
use std::process::exit;
use std::time::Instant;

const OPTIONS: &[&str] = &[
    "help",
    "n",
    "d",
    "queries",
    "neighbours",
    "similarity",
    "tightness",
    "seed",
    "out",
];

const HELP: &str = "\
Generates a synthetic data set of unit vectors with planted near neighbours.

Options (defaults in brackets):
  --n <usize>          number of points, planted neighbours included [100000]
  --d <usize>          dimension [128]
  --queries <usize>    number of query vectors [200]
  --neighbours <usize> planted points per query [1]
  --similarity <f64>   inner product of a planted point with its query [0.7]
  --tightness <f64>    mutual similarity of the planted points, 0 = independent [0.0]
  --seed <u64>         master seed [1]
  --out <path>         output file [data/dimension_<d>/sample_<n>.bin]
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
    let config = GeneratorConfig {
        n,
        d,
        queries: args.get("queries", 200)?,
        plant: PlantConfig {
            count: args.get("neighbours", 1)?,
            similarity: args.get("similarity", 0.7)?,
            tightness: args.get("tightness", 0.0)?,
        },
        seed: args.get("seed", 1)?,
    };

    let folder = format!("data/dimension_{d}");
    let path = args
        .get_string("out")
        .map(|path| path.to_string())
        .unwrap_or_else(|| format!("{folder}/sample_{n}.bin"));

    println!("generating {n} unit vectors in dimension {d}...");
    let start = Instant::now();
    let dataset = generate(&config)?;
    println!(
        "generated {} points and {} queries in {:.2}s",
        dataset.points.len(),
        dataset.queries.len(),
        start.elapsed().as_secs_f64()
    );

    if let Some(parent) = std::path::Path::new(&path).parent() {
        create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    save(&path, &dataset).map_err(|e| e.to_string())?;
    println!("saved to {path}");
    Ok(())
}
