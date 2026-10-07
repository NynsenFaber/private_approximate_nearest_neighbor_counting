//! User facing configuration, and the parameters resolved from it for a data set.

use super::algorithms::{Algorithm, Assignment};
use crate::utils::{collision_band, get_threshold, normal_sf};

/// Configuration shared by the four algorithms.
///
/// Every `Option` falls back to the paper's formula, so the usual way to write one
/// is `Config { alpha: 0.7, beta: 0.4, ..Config::default() }`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Points at inner product `>= alpha` from the query must be found.
    pub alpha: f64,
    /// Points at inner product `>= beta` may be returned or counted.
    pub beta: f64,
    /// Space/time-vs-accuracy knob `theta`. `None` uses the balanced choice
    /// `theta = rho = (1 - alpha^2)(1 - beta^2) / (1 - alpha beta)^2` (Corollary 10).
    pub theta: Option<f64>,
    /// Concatenation factor `t` of the tensorized algorithms. `None` uses
    /// `ceil(log^{1/8}(n) / (1 - alpha^2))` (Algorithm 5 line 2). Top-1 and
    /// CloseTop-1 have a single factor and accept only `None` or `Some(1)`.
    pub t: Option<usize>,
    /// Filters per factor. `None` uses `ceil(n^{(1/t) theta / (1 - alpha^2)})`
    /// (Algorithm 5 line 3). With one factor this is the
    /// `m = ceil(n^{theta / (1 - alpha^2)})` of Algorithms 2 and 4.
    pub m_sub: Option<usize>,
    /// CloseTop-1 rule only: store a point that no filter accepts at its argmax
    /// filter instead of dropping it. Not in Algorithm 4; see
    /// [`crate::anns::CloseTop1`] for why it is the default.
    pub fallback_to_argmax: bool,
    /// Master seed. The construction is reproducible from it.
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            alpha: 0.9,
            beta: 0.5,
            theta: None,
            t: None,
            m_sub: None,
            fallback_to_argmax: true,
            seed: 0,
        }
    }
}

/// Parameters of a built structure, resolved from a [`Config`] and the data set.
#[derive(Debug, Clone)]
pub struct Parameters {
    /// Name of the algorithm, e.g. `"TensorCloseTop1"`.
    pub algorithm: &'static str,
    /// Rule used to assign a point to a filter.
    pub assignment: Assignment,
    /// Number of input points.
    pub n: usize,
    /// Dimension of the input points.
    pub d: usize,
    /// Close threshold.
    pub alpha: f64,
    /// Far threshold.
    pub beta: f64,
    /// Trade-off parameter actually used.
    pub theta: f64,
    /// Balanced value `rho`, for reference.
    pub rho: f64,
    /// Number of factors; `1` for Top-1 and CloseTop-1.
    pub t: usize,
    /// Filters per factor.
    pub m_sub: usize,
    /// Query threshold `eta`, computed from `m_sub`.
    pub eta: f64,
    /// Collision band of the CloseTop-1 rule. Unused by the Top-1 rule.
    pub band: (f64, f64),
    /// Whether the CloseTop-1 rule falls back to the argmax filter.
    pub fallback_to_argmax: bool,
    /// `true` if `m_sub` had to be raised to the smallest usable value.
    pub m_sub_was_clamped: bool,
}

/// Smallest number of filters for which the collision band is non-empty
/// (`log log m > 0` requires `m > e`, and the band must fit below `sqrt(2 log m)`).
const MIN_FILTERS_PER_FACTOR: usize = 16;

impl Parameters {
    /// Derives the parameters of algorithm `A` from a configuration and a data set size.
    pub fn resolve<A: Algorithm>(config: &Config, n: usize, d: usize) -> Result<Self, String> {
        let Config {
            alpha,
            beta,
            fallback_to_argmax,
            ..
        } = *config;

        if n == 0 {
            return Err("the data set is empty".to_string());
        }
        if d == 0 {
            return Err("the data set has zero dimensions".to_string());
        }
        if !(alpha < 1.0 && alpha > 0.0) {
            return Err(format!("alpha must lie in (0, 1), got {alpha}"));
        }
        if !(beta >= 0.0 && beta < alpha) {
            return Err(format!("beta must lie in [0, alpha), got {beta}"));
        }
        let rho = (1. - alpha * alpha) * (1. - beta * beta) / (1. - alpha * beta).powi(2);
        let theta = config.theta.unwrap_or(rho);
        if !(theta.is_finite() && theta > 0.0) {
            return Err(format!("theta must be finite and positive, got {theta}"));
        }

        let t = if A::TENSORIZED {
            // Algorithm 5, line 2: t = ceil(log^{1/8}(n) / (1 - alpha^2)).
            config.t.unwrap_or_else(|| {
                let value = (n as f64).ln().powf(1. / 8.) / (1. - alpha * alpha);
                (value.ceil() as usize).max(1)
            })
        } else {
            match config.t {
                None | Some(1) => 1,
                Some(t) => {
                    return Err(format!(
                        "{} has a single factor, so t must be 1 or unset, got {t}",
                        A::NAME
                    ))
                }
            }
        };
        if t == 0 {
            return Err("the concatenation factor t must be positive".to_string());
        }

        // Algorithm 5, line 3: m_sub = ceil(n^{(1/t) theta / (1 - alpha^2)}).
        // With t = 1 this is line 2 of Algorithms 2 and 4.
        let requested = config.m_sub.unwrap_or_else(|| {
            let exponent = theta / ((1. - alpha * alpha) * t as f64);
            (n as f64).powf(exponent).ceil() as usize
        });
        let m_sub = requested.max(MIN_FILTERS_PER_FACTOR);

        Ok(Parameters {
            algorithm: A::NAME,
            assignment: A::ASSIGNMENT,
            n,
            d,
            alpha,
            beta,
            theta,
            rho,
            t,
            m_sub,
            eta: get_threshold(alpha, m_sub),
            band: collision_band(m_sub),
            fallback_to_argmax,
            m_sub_was_clamped: requested < MIN_FILTERS_PER_FACTOR,
        })
    }

    /// Expected number of filters selected per factor, `m_sub * Pr[Z >= eta]`.
    pub fn expected_filters_per_factor(&self) -> f64 {
        self.m_sub as f64 * normal_sf(self.eta)
    }

    /// Expected number of buckets a query inspects, `(m_sub Pr[Z >= eta])^t`.
    pub fn expected_probes(&self) -> f64 {
        self.expected_filters_per_factor().powi(self.t as i32)
    }

    /// Total number of buckets `m = m_sub^t`.
    pub fn total_buckets(&self) -> f64 {
        (self.m_sub as f64).powi(self.t as i32)
    }

    /// Number of filters actually stored, `t * m_sub`.
    pub fn stored_filters(&self) -> usize {
        self.t * self.m_sub
    }

    /// Human readable summary of the resolved parameters.
    pub fn summary(&self) -> String {
        let (threshold, assignment) = match self.assignment {
            Assignment::CloseTop1 => (
                format!(
                    "eta = {:.4}, collision band = [{:.4}, {:.4}]",
                    self.eta, self.band.0, self.band.1
                ),
                format!("fallback to argmax = {}", self.fallback_to_argmax),
            ),
            Assignment::Top1 => (
                format!("eta = {:.4}", self.eta),
                "assignment = argmax filter (Top-1), every point is stored".to_string(),
            ),
        };
        format!(
            "algorithm = {}\n\
             n = {}, d = {}, alpha = {}, beta = {}\n\
             theta = {:.4} (balanced rho = {:.4}), t = {}, m_sub = {}\n\
             {threshold}\n\
             simulated buckets m = m_sub^t = {:.3e}, stored filters t * m_sub = {}\n\
             expected filters per factor = {:.2}, expected probed buckets = {:.1}\n\
             {assignment}",
            self.algorithm,
            self.n,
            self.d,
            self.alpha,
            self.beta,
            self.theta,
            self.rho,
            self.t,
            self.m_sub,
            self.total_buckets(),
            self.stored_filters(),
            self.expected_filters_per_factor(),
            self.expected_probes(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsf::algorithms;

    fn config(alpha: f64, beta: f64) -> Config {
        Config {
            alpha,
            beta,
            ..Config::default()
        }
    }

    /// The tensorized parameters must follow Algorithm 5 lines 2-3.
    #[test]
    fn test_tensorized_parameters_follow_algorithm_5() {
        let params =
            Parameters::resolve::<algorithms::TensorCloseTop1>(&config(0.9, 0.5), 100_000, 64)
                .unwrap();
        let expected_rho = (1. - 0.81) * (1. - 0.25) / (1. - 0.45f64).powi(2);
        assert!((params.rho - expected_rho).abs() < 1e-12);
        assert_eq!(params.theta, params.rho);
        let expected_t = ((100_000f64).ln().powf(1. / 8.) / (1. - 0.81)).ceil() as usize;
        assert_eq!(params.t, expected_t);
        let exponent = params.theta / ((1. - 0.81) * params.t as f64);
        assert_eq!(params.m_sub, (100_000f64).powf(exponent).ceil() as usize);
    }

    /// The single factor algorithms use `m = ceil(n^{theta / (1 - alpha^2)})`
    /// (Algorithms 2 and 4, line 2) and refuse a concatenation factor.
    #[test]
    fn test_single_factor_parameters_follow_algorithms_2_and_4() {
        let base = config(0.9, 0.5);
        let params = Parameters::resolve::<algorithms::Top1>(&base, 1_000, 8).unwrap();
        assert_eq!(params.t, 1);
        let exponent = params.theta / (1. - 0.81);
        assert_eq!(params.m_sub, (1_000f64).powf(exponent).ceil() as usize);
        assert_eq!(params.assignment, Assignment::Top1);

        let one = Config {
            t: Some(1),
            ..base.clone()
        };
        assert!(Parameters::resolve::<algorithms::CloseTop1>(&one, 1_000, 8).is_ok());
        let two = Config { t: Some(2), ..base };
        assert!(Parameters::resolve::<algorithms::CloseTop1>(&two, 1_000, 8).is_err());
        assert!(Parameters::resolve::<algorithms::Top1>(&two, 1_000, 8).is_err());
    }

    /// Invalid thresholds and empty inputs are rejected; `beta = 0` is admissible
    /// (Definition 2 asks for `0 <= beta < alpha < 1`).
    #[test]
    fn test_invalid_inputs_are_rejected() {
        type A = algorithms::TensorTop1;
        assert!(Parameters::resolve::<A>(&config(0.9, 0.5), 0, 64).is_err());
        assert!(Parameters::resolve::<A>(&config(0.9, 0.5), 100, 0).is_err());
        assert!(Parameters::resolve::<A>(&config(0.5, 0.7), 100, 8).is_err());
        assert!(Parameters::resolve::<A>(&config(1.0, 0.5), 100, 8).is_err());
        assert!(Parameters::resolve::<A>(&config(0.5, 0.0), 100, 8).is_ok());
        let bad_theta = Config {
            theta: Some(-1.0),
            ..config(0.9, 0.5)
        };
        assert!(Parameters::resolve::<A>(&bad_theta, 100, 8).is_err());
        let zero_t = Config {
            t: Some(0),
            ..config(0.9, 0.5)
        };
        assert!(Parameters::resolve::<A>(&zero_t, 100, 8).is_err());
    }

    /// The derived quantities follow their formulas, and the summary names the
    /// rule-specific settings.
    #[test]
    fn test_derived_quantities_and_summary() {
        let fixed = Config {
            t: Some(3),
            m_sub: Some(100),
            ..config(0.9, 0.5)
        };
        let close = Parameters::resolve::<algorithms::TensorCloseTop1>(&fixed, 1_000, 8).unwrap();
        assert_eq!(close.stored_filters(), 300);
        assert_eq!(close.total_buckets(), 1e6);
        let per_factor = 100. * normal_sf(close.eta);
        assert!((close.expected_filters_per_factor() - per_factor).abs() < 1e-12);
        assert!((close.expected_probes() - per_factor.powi(3)).abs() < 1e-9);
        let summary = close.summary();
        assert!(summary.starts_with("algorithm = TensorCloseTop1\n"));
        assert!(summary.contains("collision band"));
        assert!(summary.contains("fallback to argmax = true"));

        let top1 = Parameters::resolve::<algorithms::TensorTop1>(&fixed, 1_000, 8).unwrap();
        let summary = top1.summary();
        assert!(!summary.contains("collision band"));
        assert!(summary.contains("argmax filter (Top-1)"));
    }

    /// A tiny `m_sub` is raised to the smallest value with a non-empty band.
    #[test]
    fn test_m_sub_is_clamped() {
        let small = Config {
            m_sub: Some(4),
            ..config(0.9, 0.5)
        };
        let params = Parameters::resolve::<algorithms::CloseTop1>(&small, 100, 8).unwrap();
        assert_eq!(params.m_sub, MIN_FILTERS_PER_FACTOR);
        assert!(params.m_sub_was_clamped);
        assert!(params.band.0 < params.band.1);
    }
}
