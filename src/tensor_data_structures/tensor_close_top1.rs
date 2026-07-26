//! TensorCloseTop-1 (Algorithm 5): the linear space `(alpha, beta)`-ANN data
//! structure, and its exact counting variant (Algorithm 3).
//!
//! The structure concatenates `t` independent [`CloseTop1`] factors, each holding
//! `m_sub = ceil(n^{(1/t) theta / (1 - alpha^2)})` Gaussian filters. A point is
//! mapped to the bucket `(i_1, ..., i_t)` given by the filter it collided with in
//! each factor, so `t * m_sub = n^{o(1)}` filters simulate `m = m_sub^t` buckets.
//! Every point is stored at most once, hence the space is `O(d n)` and — this is
//! what makes the differentially private counting variant cheap — adding or
//! removing one point changes exactly one bucket counter.

use super::bucket_index::BucketIndex;
use super::close_top1::{CloseTop1, FilterSet};
use super::dp_annc::DpAnnc;
use super::probe::{candidate_filters, product_size, BucketKey};
use crate::dp::truncated_laplace::TruncatedLaplace;
use crate::utils::{derive_seed, dot_product, get_threshold, normal_sf};
use rayon::prelude::*;
use std::collections::HashMap;

/// User facing configuration of [`TensorCloseTop1`].
#[derive(Debug, Clone)]
pub struct Config {
    /// Points at inner product `>= alpha` from the query must be found.
    pub alpha: f64,
    /// Points at inner product `>= beta` may be returned/counted.
    pub beta: f64,
    /// Space/time-vs-accuracy knob `theta`. `None` uses the balanced choice
    /// `theta = rho = (1 - alpha^2)(1 - beta^2) / (1 - alpha beta)^2`.
    pub theta: Option<f64>,
    /// Concatenation factor `t`. `None` uses `ceil(log^{1/8}(n) / (1 - alpha^2))`.
    pub t: Option<usize>,
    /// Filters per factor `m_sub`. `None` uses `ceil(n^{(1/t) theta / (1 - alpha^2)})`.
    pub m_sub: Option<usize>,
    /// Store points that collided with no filter at their Top-1 (argmax) filter
    /// instead of dropping them.
    pub fallback_to_argmax: bool,
    /// Master seed; the whole construction is reproducible from it.
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

/// Resolved parameters of a built structure.
#[derive(Debug, Clone)]
pub struct Parameters {
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
    /// Concatenation factor.
    pub t: usize,
    /// Filters per factor.
    pub m_sub: usize,
    /// Query threshold `eta`, computed from `m_sub`.
    pub eta: f64,
    /// Collision band used by each factor.
    pub band: (f64, f64),
    /// Whether non-colliding points fall back to the argmax filter.
    pub fallback_to_argmax: bool,
    /// `true` if `m_sub` had to be raised to the smallest usable value.
    pub m_sub_was_clamped: bool,
}

/// Smallest number of filters for which the collision band is non-empty
/// (`log log m > 0` requires `m > e`, and the band must fit below `sqrt(2 log m)`).
const MIN_FILTERS_PER_FACTOR: usize = 16;

impl Parameters {
    /// Derives the parameters of Algorithm 5 from a configuration and a data set size.
    pub fn resolve(config: &Config, n: usize, d: usize) -> Result<Self, String> {
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

        // Algorithm 5, line 2: t = ceil(log^{1/8}(n) / (1 - alpha^2)).
        let t = config.t.unwrap_or_else(|| {
            let value = (n as f64).ln().powf(1. / 8.) / (1. - alpha * alpha);
            (value.ceil() as usize).max(1)
        });
        if t == 0 {
            return Err("the concatenation factor t must be positive".to_string());
        }

        // Algorithm 5, line 3: m_sub = ceil(n^{(1/t) theta / (1 - alpha^2)}).
        let requested = config.m_sub.unwrap_or_else(|| {
            let exponent = theta / ((1. - alpha * alpha) * t as f64);
            (n as f64).powf(exponent).ceil() as usize
        });
        let m_sub = requested.max(MIN_FILTERS_PER_FACTOR);

        Ok(Parameters {
            n,
            d,
            alpha,
            beta,
            theta,
            rho,
            t,
            m_sub,
            eta: get_threshold(alpha, m_sub),
            band: crate::utils::collision_band(m_sub),
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

    /// Total number of simulated buckets `m = m_sub^t`.
    pub fn total_buckets(&self) -> f64 {
        (self.m_sub as f64).powi(self.t as i32)
    }

    /// Number of filters actually stored, `t * m_sub`.
    pub fn stored_filters(&self) -> usize {
        self.t * self.m_sub
    }

    /// Human readable summary of the resolved parameters.
    pub fn summary(&self) -> String {
        format!(
            "n = {}, d = {}, alpha = {}, beta = {}\n\
             theta = {:.4} (balanced rho = {:.4}), t = {}, m_sub = {}\n\
             eta = {:.4}, collision band = [{:.4}, {:.4}]\n\
             simulated buckets m = m_sub^t = {:.3e}, stored filters t * m_sub = {}\n\
             expected filters per factor = {:.2}, expected probed buckets = {:.1}\n\
             fallback to argmax = {}",
            self.n,
            self.d,
            self.alpha,
            self.beta,
            self.theta,
            self.rho,
            self.t,
            self.m_sub,
            self.eta,
            self.band.0,
            self.band.1,
            self.total_buckets(),
            self.stored_filters(),
            self.expected_filters_per_factor(),
            self.expected_probes(),
            self.fallback_to_argmax,
        )
    }
}

/// Outcome of an `(alpha, beta)`-ANN query.
#[derive(Debug, Clone)]
pub struct AnnOutcome {
    /// Index of a point at inner product `>= beta` from the query, if one was found.
    pub point: Option<usize>,
    /// Size `|I(q)|` of the Cartesian product `B_1 x ... x B_t` the query covers.
    pub probed_buckets: usize,
    /// Number of *non-empty* buckets of that product that were actually visited.
    pub matched_buckets: usize,
    /// Number of points whose inner product with the query was evaluated.
    pub inspected_points: usize,
}

/// Outcome of an exact (non private) `(alpha, beta)`-ANN counting query.
#[derive(Debug, Clone)]
pub struct CountOutcome {
    /// Number of points stored in the inspected buckets.
    pub count: u64,
    /// Size `|I(q)|` of the Cartesian product `B_1 x ... x B_t` the query covers.
    pub probed_buckets: usize,
    /// Number of *non-empty* buckets of that product that were actually visited.
    pub matched_buckets: usize,
}

/// The TensorCloseTop-1 data structure.
pub struct TensorCloseTop1 {
    /// Resolved parameters.
    pub params: Parameters,
    filters: Vec<FilterSet>,
    buckets: BucketIndex<Vec<u32>>,
    data: Vec<Vec<f64>>,
    stored_points: usize,
}

/// Heap memory a built structure actually holds, measured from allocated
/// capacities (not lengths), broken down by what it is spent on.
///
/// `data_bytes` is what any exact baseline needs at minimum — a linear scan does
/// not save anything there, it just skips the rest of this breakdown. Everything
/// else is the price of sub-linear queries.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryFootprint {
    /// The input points, `Vec<Vec<f64>>`.
    pub data_bytes: usize,
    /// The `t * m_sub` Gaussian filters shared by every factor's `search`.
    pub filters_bytes: usize,
    /// The sorted bucket index: one key (`t` `u32`s) per non-empty bucket, plus
    /// the point ids stored in it.
    pub bucket_index_bytes: usize,
}

impl MemoryFootprint {
    /// Total heap memory held by the structure.
    pub fn total(&self) -> usize {
        self.data_bytes + self.filters_bytes + self.bucket_index_bytes
    }

    /// Memory beyond what storing the raw points alone costs, i.e. what an exact
    /// linear-scan baseline over the same data would not need.
    pub fn overhead_bytes(&self) -> usize {
        self.filters_bytes + self.bucket_index_bytes
    }

    /// Overhead as a fraction of the raw data size.
    pub fn overhead_ratio(&self) -> f64 {
        self.overhead_bytes() as f64 / self.data_bytes.max(1) as f64
    }

    /// Human readable breakdown.
    pub fn summary(&self) -> String {
        format!(
            "{} total = {} raw data + {} overhead ({} filters, {} bucket index), \
             {:.1}% over the raw data a linear scan would need",
            format_bytes(self.total()),
            format_bytes(self.data_bytes),
            format_bytes(self.overhead_bytes()),
            format_bytes(self.filters_bytes),
            format_bytes(self.bucket_index_bytes),
            100. * self.overhead_ratio(),
        )
    }
}

/// Formats a byte count with a binary unit (KiB/MiB/GiB), 3 significant digits.
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024. && unit + 1 < UNITS.len() {
        value /= 1024.;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

impl TensorCloseTop1 {
    /// Builds the structure over `data` (points must lie on the unit sphere).
    pub fn build(data: Vec<Vec<f64>>, config: &Config) -> Result<Self, String> {
        let n = data.len();
        let d = data.first().map(|point| point.len()).unwrap_or(0);
        let params = Parameters::resolve(config, n, d)?;
        if data.iter().any(|point| point.len() != d) {
            return Err("all points must have the same dimension".to_string());
        }

        // The t factors are independent: each one gets its own filter seed.
        let substructures: Vec<CloseTop1> = (0..params.t)
            .map(|i| {
                CloseTop1::build(
                    &data,
                    params.m_sub,
                    params.eta,
                    params.fallback_to_argmax,
                    derive_seed(config.seed, i as u64 + 1),
                )
            })
            .collect();

        // Bucket key of every point: the concatenation of its t filter indices.
        // A point missing in any factor is not stored at all (Definition 11 allows
        // the partition to leave points out).
        let keys: Vec<Option<BucketKey>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut key = BucketKey::with_capacity(params.t);
                for factor in &substructures {
                    key.push(factor.bucket_of(i)?);
                }
                Some(key)
            })
            .collect();

        let mut hash_table: HashMap<BucketKey, Vec<u32>> = HashMap::new();
        let mut stored_points = 0usize;
        for (i, key) in keys.into_iter().enumerate() {
            if let Some(key) = key {
                hash_table.entry(key).or_default().push(i as u32);
                stored_points += 1;
            }
        }

        // `match_list` and `band` were only needed to build the keys above — the
        // bucket index already encodes the same point-to-bucket assignment via the
        // point ids it stores, so keeping them alongside would just pay for that
        // information twice. Only the (data independent) filters are needed at
        // query time, so that is all that survives past this point.
        let filters: Vec<FilterSet> = substructures
            .into_iter()
            .map(|factor| factor.filters)
            .collect();

        Ok(TensorCloseTop1 {
            params,
            filters,
            buckets: BucketIndex::from_map(hash_table),
            data,
            stored_points,
        })
    }

    /// Number of input points that ended up in a bucket.
    pub fn stored_points(&self) -> usize {
        self.stored_points
    }

    /// Number of non-empty buckets.
    pub fn occupied_buckets(&self) -> usize {
        self.buckets.len()
    }

    /// The `i`-th input point.
    pub fn point(&self, i: usize) -> &[f64] {
        &self.data[i]
    }

    /// The candidate filters `B_1, ..., B_t` selected by `q` in each factor.
    pub fn candidate_filters(&self, query: &[f64]) -> Vec<Vec<u32>> {
        candidate_filters(self.filters.iter(), query)
    }

    /// Measures the structure's actual heap footprint from allocated capacities
    /// (not lengths, not an asymptotic estimate), broken down by what it is spent
    /// on. See [`MemoryFootprint`].
    pub fn memory_footprint(&self) -> MemoryFootprint {
        let vec_f64 = std::mem::size_of::<Vec<f64>>();
        let data_bytes = self.data.capacity() * vec_f64
            + self
                .data
                .iter()
                .map(|point| point.capacity() * std::mem::size_of::<f64>())
                .sum::<usize>();

        let filters_bytes = self.filters.capacity() * std::mem::size_of::<FilterSet>()
            + self
                .filters
                .iter()
                .map(|set| {
                    set.gaussian_vectors.capacity() * vec_f64
                        + set
                            .gaussian_vectors
                            .iter()
                            .map(|filter| filter.capacity() * std::mem::size_of::<f64>())
                            .sum::<usize>()
                })
                .sum::<usize>();

        let bucket_index_bytes = self
            .buckets
            .memory_bytes(|bucket| bucket.capacity() * std::mem::size_of::<u32>());

        MemoryFootprint {
            data_bytes,
            filters_bytes,
            bucket_index_bytes,
        }
    }

    /// Solves `(alpha, beta)`-ANN: returns a point at inner product `>= beta`, if
    /// one is found in the inspected buckets.
    pub fn query(&self, query: &[f64]) -> AnnOutcome {
        let levels = self.candidate_filters(query);
        let mut inspected_points = 0usize;
        let mut found = None;

        let matched_buckets = self.buckets.for_each_match(&levels, |_, bucket| {
            for &point_id in bucket {
                inspected_points += 1;
                if dot_product(query, &self.data[point_id as usize]) >= self.params.beta {
                    found = Some(point_id as usize);
                    return false;
                }
            }
            true
        });

        AnnOutcome {
            point: found,
            probed_buckets: product_size(&levels),
            matched_buckets,
            inspected_points,
        }
    }

    /// Solves `(alpha, beta)`-ANN counting *without* privacy (Algorithm 3): the sum
    /// of the sizes of the inspected buckets. Since every point is stored at most
    /// once, no point is counted twice.
    pub fn count(&self, query: &[f64]) -> CountOutcome {
        let levels = self.candidate_filters(query);
        let mut count = 0u64;
        let matched_buckets = self.buckets.for_each_match(&levels, |_, bucket| {
            count += bucket.len() as u64;
            true
        });

        CountOutcome {
            count,
            probed_buckets: product_size(&levels),
            matched_buckets,
        }
    }

    /// Releases the counting structure under differential privacy (Theorem 13).
    ///
    /// The returned structure holds only the (data independent) filters and the
    /// noisy counters — no input point survives — so it can be published. Several
    /// releases of the same structure each spend a full `(epsilon, delta)` budget,
    /// which is fine for measuring the error over independent noise draws but must
    /// be accounted for by composition if they are actually published.
    pub fn release(&self, mechanism: TruncatedLaplace, seed: u64) -> DpAnnc {
        let filters = self.filters.clone();
        let counts = self
            .buckets
            .iter()
            .map(|(key, bucket)| (key.clone(), bucket.len() as u64));
        DpAnnc::release(self.params.clone(), filters, counts, mechanism, seed)
    }

    /// Same as [`TensorCloseTop1::release`], consuming the structure so that the
    /// input points are dropped along with it.
    pub fn into_private(self, mechanism: TruncatedLaplace, seed: u64) -> DpAnnc {
        let TensorCloseTop1 {
            params,
            filters,
            buckets,
            ..
        } = self;
        let counts = buckets
            .into_iter_entries()
            .map(|(key, bucket)| (key, bucket.len() as u64));
        DpAnnc::release(params, filters, counts, mechanism, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{plant_neighbours, PlantConfig};
    use crate::tensor_data_structures::probe::ProbeIter;
    use crate::utils::generate_unit_sphere_vectors;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_config(seed: u64) -> Config {
        Config {
            alpha: 0.9,
            beta: 0.5,
            t: Some(3),
            m_sub: Some(64),
            seed,
            ..Config::default()
        }
    }

    /// Parameters must follow Algorithm 5 and reject invalid thresholds.
    #[test]
    fn test_parameter_resolution() {
        let config = Config {
            alpha: 0.9,
            beta: 0.5,
            ..Config::default()
        };
        let params = Parameters::resolve(&config, 100_000, 64).unwrap();
        let expected_rho = (1. - 0.81) * (1. - 0.25) / (1. - 0.45f64).powi(2);
        assert!((params.rho - expected_rho).abs() < 1e-12);
        assert_eq!(params.theta, params.rho);
        let expected_t = ((100_000f64).ln().powf(1. / 8.) / (1. - 0.81)).ceil() as usize;
        assert_eq!(params.t, expected_t);
        let exponent = params.theta / ((1. - 0.81) * params.t as f64);
        assert_eq!(params.m_sub, (100_000f64).powf(exponent).ceil() as usize);

        assert!(Parameters::resolve(&config, 0, 64).is_err());
        let bad = Config {
            alpha: 0.5,
            beta: 0.7,
            ..Config::default()
        };
        assert!(Parameters::resolve(&bad, 100, 8).is_err());
        // beta = 0 is admissible: Definition 2 asks for 0 <= beta < alpha < 1.
        assert!(Parameters::resolve(
            &Config {
                alpha: 0.5,
                beta: 0.0,
                ..Config::default()
            },
            100,
            8
        )
        .is_ok());
    }

    /// Algorithm 5 line 9 computes `eta` from the *per factor* filter count `m̃`,
    /// not from the `m̃^t` simulated total. Getting this wrong would silently
    /// mis-tune every query: a threshold derived from `m̃^t` is far too high, so
    /// `search` would return almost no candidate filters.
    #[test]
    fn test_eta_is_derived_from_m_sub_not_from_the_simulated_total() {
        let params = Parameters::resolve(&test_config(0), 100_000, 64).unwrap();
        assert_eq!(
            params.eta,
            crate::utils::get_threshold(params.alpha, params.m_sub)
        );
        assert!(
            params.eta < crate::utils::get_threshold(params.alpha, params.total_buckets() as usize)
        );
        // Each factor's collision band likewise uses m_sub (Algorithm 5 line 4
        // constructs every factor as CloseTop-1(S, m̃)).
        assert_eq!(params.band, crate::utils::collision_band(params.m_sub));
        // A query must be able to reach the filters points were assigned to.
        assert!(params.eta < params.band.0);
    }

    /// A point is stored only if it collides in *every* factor: its key is the
    /// concatenation of the `t` filter indices, so one miss loses the point. This is
    /// the union bound over `t` factors in the proof of Lemma 15.
    #[test]
    fn test_a_point_missing_in_one_factor_is_not_stored() {
        let data = generate_unit_sphere_vectors(500, 16, 41);
        let strict = Config {
            fallback_to_argmax: false,
            ..test_config(42)
        };
        let structure = TensorCloseTop1::build(data.clone(), &strict).unwrap();
        // Without the fallback some points collide with no filter and are dropped,
        // which is exactly what Algorithm 4 line 9 prescribes.
        assert!(structure.stored_points() < structure.params.n);
        // With the fallback every point is kept.
        let lenient = TensorCloseTop1::build(data, &test_config(42)).unwrap();
        assert_eq!(lenient.stored_points(), lenient.params.n);
    }

    /// Every stored point must be retrievable from its own bucket, and the
    /// structure must store each point at most once (this is what bounds the
    /// sensitivity of the counting query).
    #[test]
    fn test_partition_property() {
        let data = generate_unit_sphere_vectors(400, 16, 11);
        let structure = TensorCloseTop1::build(data, &test_config(12)).unwrap();
        let mut seen = vec![0usize; structure.params.n];
        for (_, bucket) in structure.buckets.iter() {
            for &point_id in bucket {
                seen[point_id as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&times| times <= 1));
        assert_eq!(
            seen.iter().filter(|&&times| times == 1).count(),
            structure.stored_points()
        );
        // With the default fallback every point is stored.
        assert_eq!(structure.stored_points(), structure.params.n);
    }

    /// A query must never return a point below the `beta` threshold, and the count
    /// must equal the number of points stored in the inspected buckets.
    #[test]
    fn test_query_and_count_are_consistent() {
        let mut rng = StdRng::seed_from_u64(21);
        let mut data = generate_unit_sphere_vectors(400, 16, 13);
        let query = crate::utils::random_unit_vector(16, &mut rng);
        data.extend(plant_neighbours(
            &query,
            &PlantConfig {
                count: 10,
                similarity: 0.95,
                tightness: 0.999,
            },
            &mut rng,
        ));
        let structure = TensorCloseTop1::build(data, &test_config(14)).unwrap();

        let ann = structure.query(&query);
        if let Some(point) = ann.point {
            assert!(dot_product(&query, structure.point(point)) >= structure.params.beta);
        }

        let counted = structure.count(&query);
        // Recompute the count by naively enumerating the same Cartesian product,
        // which is the definition the fast bucket index has to agree with.
        let levels = structure.candidate_filters(&query);
        let expected: u64 = ProbeIter::new(&levels, usize::MAX)
            .filter_map(|key| structure.buckets.get(&key))
            .map(|bucket| bucket.len() as u64)
            .sum();
        assert_eq!(counted.count, expected);
        assert_eq!(counted.probed_buckets, product_size(&levels));
        assert!(counted.count <= structure.stored_points() as u64);
    }

    /// The counting query answers `|S ∩ B(q, alpha)| <= ans <= |S ∩ B(q, beta)| + K`,
    /// where `K` is the number of far points swept in by the filters (Lemma 12).
    /// The decomposition of the answer into near and far points is checked here.
    #[test]
    fn test_count_decomposes_into_near_and_far_points() {
        let mut rng = StdRng::seed_from_u64(31);
        let mut data = generate_unit_sphere_vectors(600, 16, 15);
        let query = crate::utils::random_unit_vector(16, &mut rng);
        data.extend(plant_neighbours(
            &query,
            &PlantConfig {
                count: 20,
                similarity: 0.95,
                tightness: 0.99,
            },
            &mut rng,
        ));
        let points_within_beta = crate::data::exact_count(&data, &query, 0.5);
        let structure = TensorCloseTop1::build(data, &test_config(16)).unwrap();
        let counted = structure.count(&query);

        // Split the counted points into the ones the problem allows to report
        // (inner product >= beta) and the far points that inflate the estimate.
        let levels = structure.candidate_filters(&query);
        let mut within_beta = 0u64;
        let mut far_points = 0u64;
        for key in ProbeIter::new(&levels, usize::MAX) {
            if let Some(bucket) = structure.buckets.get(&key) {
                for &point_id in bucket {
                    if dot_product(&query, structure.point(point_id as usize))
                        >= structure.params.beta
                    {
                        within_beta += 1;
                    } else {
                        far_points += 1;
                    }
                }
            }
        }
        assert_eq!(counted.count, within_beta + far_points);
        // The points reported within beta really are a subset of S ∩ B(q, beta).
        assert!(within_beta <= points_within_beta);
    }
}
