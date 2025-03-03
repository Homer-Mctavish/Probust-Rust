use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Represents a probust constraint for a water reservoir control problem with joint probabilistic constraints.
/// The continuous time interval [0, T] is discretized into n subintervals with nodes t₀, t₁, …, tₙ,
/// so that the decision is given by a vector c = (c₁, …, cₙ), where each component represents
/// the integrated water release on the subinterval [tᵢ₋₁, tᵢ]. The reservoir level at node tᵢ is computed as:
///   level(tᵢ) = initial_level + (∑_{j=1}^{i} inflow_j) − (∑_{j=1}^{i} c_j).
/// The joint probabilistic constraint (Equation (38)) requires that for every i the level is in [lower_threshold, upper_threshold].
/// This fits into the finite-dimensional formulation (Equation (40)):
///   min { f(c) | φ(c) ≥ p }.
#[allow(dead_code)]
pub struct ProbustConstraint {
    /// Required safety level (e.g., 0.9 for 90% probability)
    pub safety_level: f64,
    /// Time grid nodes: t₀, t₁, …, tₙ (n = grid.len() - 1 subintervals)
    pub grid: Vec<f64>,
    /// Lower bound on the reservoir level
    pub lower_threshold: f64,
    /// Upper bound on the reservoir level
    pub upper_threshold: f64,
    /// Number of Monte Carlo samples for probability estimation
    pub num_samples: usize,
    /// Covariance matrix σ for the integrated inflows (dimension n×n)
    #[allow(dead_code)]
    pub sigma: DMatrix<f64>,
    /// Precomputed Cholesky factor L (such that σ = L * Lᵀ)
    pub chol: DMatrix<f64>,
}

impl ProbustConstraint {
    /// Constructs a new ProbustConstraint given a covariance matrix σ.
    /// The dimension of σ must equal the number of subintervals (grid.len() - 1).
    pub fn new(
        safety_level: f64,
        grid: Vec<f64>,
        lower_threshold: f64,
        upper_threshold: f64,
        num_samples: usize,
        sigma: DMatrix<f64>,
    ) -> Self {
        let chol = sigma
            .clone()
            .cholesky()
            .expect("Covariance matrix is not positive definite.")
            .l();
        ProbustConstraint {
            safety_level,
            grid,
            lower_threshold,
            upper_threshold,
            num_samples,
            sigma,
            chol,
        }
    }

    /// Evaluates the joint probability φ(c) that the reservoir level remains within the safe interval
    /// [lower_threshold, upper_threshold] at the end of each subinterval.
    ///
    /// For each Monte Carlo sample, a vector of integrated inflows is generated from a centered multivariate
    /// Gaussian distribution (via the Cholesky factor). The reservoir level at node tᵢ is computed as:
    ///   level(tᵢ) = initial_level + (∑_{j=1}^{i} inflow_j) − (∑_{j=1}^{i} c_j).
    /// The sample is feasible if the level is within the safe interval for all subinterval endpoints.
    pub fn evaluate_probability(&self, decision: &Decision) -> f64 {
        let mut rng = rand::thread_rng();
        let n = self.grid.len() - 1; // number of subintervals
        assert_eq!(
            decision.release.len(),
            n,
            "Length of release vector must equal number of subintervals."
        );
        let mut success_count = 0;
        for _ in 0..self.num_samples {
            // Sample a vector z ∈ ℝⁿ from the standard normal distribution.
            let z = DVector::from_iterator(n, (0..n).map(|_| rng.sample(StandardNormal)));
            // Transform z via the Cholesky factor to obtain a sample from N(0, σ).
            let inflow_sample = &self.chol * z;
            let mut cumulative_inflow = 0.0;
            let mut cumulative_release = 0.0;
            let mut feasible = true;
            // Check reservoir level at the end of each subinterval.
            for i in 0..n {
                cumulative_inflow += inflow_sample[i];
                cumulative_release += decision.release[i];
                let level = decision.initial_level + cumulative_inflow - cumulative_release;
                if level < self.lower_threshold || level > self.upper_threshold {
                    feasible = false;
                    break;
                }
            }
            if feasible {
                success_count += 1;
            }
        }
        success_count as f64 / self.num_samples as f64
    }
}

/// Computes the continuous reservoir level at any time t ∈ [0, T] according to:
///
///   l(t) = l₀ + ⟨A(t), ξ⟩ + B(t)
///          − [∑_{j=1}^{i(t)} x_j · (t_j − t_{j-1}) + x_{i(t)+1} · (t − t_{i(t)})],
///
/// where:
///  - A(t) = [A₁(t), …, A_d(t)] with A_j(t) = ∫₀ᵗ α_j(τ)dτ,
///  - B(t) = ∫₀ᵗ β(τ)dτ,
///  - i(t) = max { i | t ≥ grid[i] },
///  - The decision vector “release” is piecewise constant on the subintervals defined by grid,
///  - ξ is a vector of parameters.
///  
/// The functions a_func and b_func are provided to compute A(t) (a vector) and B(t) (a scalar) respectively.
pub fn reservoir_level_continuous<F, G>(
    t: f64,
    l0: f64,
    xi: &Vec<f64>,
    a_func: F,
    b_func: G,
    grid: &Vec<f64>,
    release: &Vec<f64>,
) -> f64
where
    F: Fn(f64) -> Vec<f64>,
    G: Fn(f64) -> f64,
{
    // Compute A(t) and B(t)
    let a_t = a_func(t);
    let b_t = b_func(t);
    // Inner product ⟨A(t), ξ⟩.
    let inner_prod: f64 = a_t.iter().zip(xi.iter()).map(|(a, xi_val)| a * xi_val).sum();
    // Determine subinterval index: i(t) = max { i | t ≥ grid[i] }.
    let n = grid.len() - 1;
    let mut i = 0;
    for j in 0..n {
        if t >= grid[j] && t < grid[j + 1] {
            i = j;
            break;
        }
        if t >= grid[n] {
            i = n - 1;
        }
    }
    // Cumulative release: full intervals plus partial in current interval.
    let mut cum_release = 0.0;
    for j in 0..i {
        cum_release += release[j] * (grid[j + 1] - grid[j]);
    }
    let partial = if t >= grid[i] { t - grid[i] } else { 0.0 };
    cum_release += release[i] * partial;
    // Return the continuous reservoir level.
    l0 + inner_prod + b_t - cum_release
}

/// Checks whether a decision (release vector) satisfies the deterministic constraints for X:
///
///   X = { x ∈ ℝⁿ | 0 ≤ x_i ≤ x̄ for all i, and ∑_{i=1}^{n} x_i ≤ B(T) }.
pub fn check_deterministic_constraints(decision: &Decision, x_bar: f64, B_T: f64) -> bool {
    decision.release.iter().all(|&x| x >= 0.0 && x <= x_bar)
        && decision.release.iter().sum::<f64>() <= B_T
}

/// Represents a decision in the water reservoir control problem.
/// The decision consists of an initial water level and a vector of release amounts,
/// where each release[i] = c_i = ∫_{t_{i-1}}^{t_i} c̃(t) dt for i = 1,…,n.
pub struct Decision {
    pub initial_level: f64,
    pub release: Vec<f64>,
}
