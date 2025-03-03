// src/constraint_optimizer.rs

use crate::probust_constraint::{ProbustConstraint, Decision};
use rand::prelude::*;

/// (Equation 41)
/// Lower-level optimization: given the current ProbustConstraint and Decision,
/// select a candidate time t* from a list of candidate points that minimizes the updated probability
/// when added to the current grid. This corresponds to choosing the most “informative” index.
pub fn optimize_grid(probust: &ProbustConstraint, decision: &Decision, candidate_t: &[f64]) -> f64 {
    let mut best_t = candidate_t[0];
    let mut best_prob = 1.0; // probability values are in [0,1]
    // For each candidate t, form a temporary grid (current grid ∪ {t}) and evaluate the probability.
    for &t in candidate_t {
        let mut new_grid = probust.grid.clone();
        new_grid.push(t);
        new_grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Create a temporary ProbustConstraint with the new grid.
        let temp_constraint = ProbustConstraint {
            grid: new_grid,
            safety_level: probust.safety_level,
            lower_threshold: probust.lower_threshold,
            upper_threshold: probust.upper_threshold,
            num_samples: probust.num_samples,
            sigma: probust.sigma.clone(),
            chol: probust.chol.clone(),
        };
        let prob = temp_constraint.evaluate_probability(decision);
        if prob < best_prob {
            best_prob = prob;
            best_t = t;
        }
    }
    best_t
}

/// (Equation 42)
/// Upper-level optimization: given a ProbustConstraint and an initial Decision,
/// this function attempts to improve the decision vector (here, the release vector)
/// by minimizing an objective function (for instance, total release) while ensuring that
/// the joint probabilistic constraint remains satisfied (i.e. φ(c) ≥ p).
/// We use a simple random perturbation (hill-climbing) approach for demonstration.
pub fn optimize_decision(probust: &ProbustConstraint, initial_decision: Decision, max_iter: usize) -> Decision {
    let mut current_decision = initial_decision;
    let mut best_obj = objective(&current_decision);
    let p_target = probust.safety_level;
    let mut rng = rand::thread_rng();
    let step_size = 0.5; // step size for perturbations

    for _ in 0..max_iter {
        // Generate a candidate decision by perturbing each release component.
        let candidate_release: Vec<f64> = current_decision.release.iter().map(|&r| {
            let perturbation: f64 = rng.gen_range(-step_size..step_size);
            (r + perturbation).max(0.0) // ensure nonnegative release
        }).collect();

        let candidate_decision = Decision {
            initial_level: current_decision.initial_level, // keep initial level fixed
            release: candidate_release,
        };

        // Evaluate the joint probability for the candidate decision.
        let prob = probust.evaluate_probability(&candidate_decision);
        // Only accept the candidate if it is feasible.
        if prob >= p_target {
            let candidate_obj = objective(&candidate_decision);
            if candidate_obj < best_obj {
                current_decision = candidate_decision;
                best_obj = candidate_obj;
            }
        }
    }
    current_decision
}

/// A sample objective function f(c); here, we define it as the total release over all subintervals.
fn objective(decision: &Decision) -> f64 {
    decision.release.iter().sum()
}
