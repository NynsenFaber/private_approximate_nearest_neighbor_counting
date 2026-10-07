//! Code shared by the experiment binaries.

// Each binary compiles its own copy of this module and uses a different subset.
#![allow(dead_code)]

pub mod cli;

use ann_rust::{Config, Parameters};
use cli::Args;

/// Help lines for the options read by [`config`] and `--algorithm`.
pub const ALGORITHM_HELP: &str = "\
  --algorithm <name>   tensor-close-top1 (Algorithm 5), tensor-top1, close-top1 (Algorithm 4)
                       or top1 (Algorithm 2) [tensor-close-top1]
  --theta <f64>        space/time knob; default is the balanced rho
  --t <usize>          concatenation factor of the tensor algorithms;
                       default ceil(log^{1/8}(n) / (1 - alpha^2))
  --m-sub <usize>      filters per factor; default ceil(n^{(1/t) theta / (1 - alpha^2)}),
                       with t = 1 for top1 and close-top1, which therefore need it set
  --strict             close-top1 rules: drop points that collide with no filter
                       (literal Algorithm 4)
";

/// The `--algorithm` value, `tensor-close-top1` when absent.
pub fn algorithm(args: &Args) -> &str {
    args.get_string("algorithm").unwrap_or("tensor-close-top1")
}

/// Error for an `--algorithm` value no binary knows.
pub fn unknown_algorithm(name: &str) -> String {
    format!(
        "unknown --algorithm '{name}', expected tensor-close-top1, tensor-top1, close-top1 or top1"
    )
}

/// The structure's [`Config`] from `--theta --t --m-sub --strict`.
pub fn config(args: &Args, alpha: f64, beta: f64, seed: u64) -> Result<Config, String> {
    Ok(Config {
        alpha,
        beta,
        theta: args.get_optional("theta")?,
        t: args.get_optional("t")?,
        m_sub: args.get_optional("m-sub")?,
        fallback_to_argmax: !args.flag("strict", false)?,
        seed,
    })
}

/// Refuses parameters whose filters alone would not fit in memory, which is what
/// the default `m` of Top-1 and CloseTop-1 (`n^{theta / (1 - alpha^2)}`) gives on
/// any sizeable data set.
pub fn check_filter_memory(params: &Parameters) -> Result<(), String> {
    const LIMIT_BYTES: f64 = 4. * 1024. * 1024. * 1024.;
    let bytes = params.stored_filters() as f64 * params.d as f64 * 8.;
    if bytes > LIMIT_BYTES {
        return Err(format!(
            "{} would store {} filters ({:.1} GiB); pass a smaller --m-sub or --theta",
            params.algorithm,
            params.stored_filters(),
            bytes / 1024f64.powi(3)
        ));
    }
    Ok(())
}

/// Printed when `m_sub` was raised to the smallest usable value.
pub const CLAMPED_NOTE: &str = "note: m_sub was raised to the smallest usable value; the \
requested value was too small for the collision band to exist\n";
