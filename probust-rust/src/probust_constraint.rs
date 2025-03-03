// src/probust_constraint.rs

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Represents a probust constraint for a water reservoir control problem with a joint probabilistic constraint.
/// The time interval [0, T] is discretized into n subintervals with nodes t₀, t₁, …, tₙ.
/// On each subinterval, the water release is assumed to be constant so that the decision is
/// represented by c = (c₁, …, cₙ), where
///    cᵢ = ∫ₜᵢ₋₁^(tᵢ) 𝑐̃(t) dt  for i = 1, …, n.
/// The reservoir level at node tᵢ is computed as:
///    level(tᵢ) = initial_level + (∑ⱼ₌₁ⁱ inflowⱼ) − (∑ⱼ₌₁ⁱ cⱼ).
/// The joint constraint (see Equation (38)) requires that for every i the level remains in
/// the safe interval [lower_threshold, upper_threshold]. Equation (40) then represents the finite-dimensional
/// formulation:
///    min { f(c) | φ(c) ≥ p },
/// where φ(c) = P(level(tᵢ) ∈ [lower_threshold, upper_threshold] ∀ i).
pub struct ProbustConstraint {
    /// Required safety level (e.g., 0.9 for 90% probability)
    pub safety_level: f64,
    /// Time grid nodes: t₀, t₁, …, tₙ (thus n = grid.len() - 1 subintervals)
    pub grid: Vec<f64>,
    /// Lower bound on the reservoir level
    pub lower_threshold: f64,
    /// Upper bound on the reservoir level
    pub upper_threshold: f64,
    /// Number of Monte Carlo samples for probability estimation
    pub num_samples: usize,
    /// Covariance matrix σ for the integrated inflows over each subinterval (dimension n×n)
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
    /// Gaussian distribution (using the Cholesky factor). The reservoir level at node tᵢ is computed as:
    ///    level(tᵢ) = initial_level + (∑ⱼ₌₁ⁱ inflowⱼ) − (∑ⱼ₌₁ⁱ cⱼ).
    /// The sample is feasible if the level is within [lower_threshold, upper_threshold] for all i.
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
            // Transform z using the Cholesky factor to obtain a sample from N(0, σ).
            let inflow_sample = &self.chol * z;

            let mut cumulative_inflow = 0.0;
            let mut cumulative_release = 0.0;
            let mut feasible = true;

            // Check the reservoir level at the end of each subinterval.
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

/// Represents a decision in the water reservoir control problem.
/// The decision consists of an initial water level and a vector of release amounts
/// c = (c₁, …, cₙ), where cᵢ = ∫ₜᵢ₋₁^(tᵢ) 𝑐̃(t) dt (i = 1, …, n).
pub struct Decision {
    pub initial_level: f64,
    pub release: Vec<f64>,
}
