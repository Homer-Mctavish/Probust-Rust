use crate::probust_constraint::{ProbustConstraint, Decision};
use crate::constraint_optimizer::{optimize_decision, optimize_grid};
use nalgebra::DMatrix;

/// Adaptive optimization over equations (41)-(42) until convergence
pub fn adaptive_optimize(
    mut probust: ProbustConstraint,
    mut decision: Decision,
    candidate_times: &[f64],
    x_bar: f64,
    B_T: f64,
    max_outer_iter: usize,
    max_inner_iter: usize,
) -> (ProbustConstraint, Decision) {
    let mut last_prob = probust.evaluate_probability(&decision);
    println!("Initial φ(c) = {:.4}", last_prob);

    for iter in 0..max_outer_iter {
        println!("\n=== Iteration {} ===", iter + 1);

        // --- (1) Grid Update Step: choose new t* from candidate set ---
        let t_star = optimize_grid(&probust, &decision, candidate_times);
        if !probust.grid.contains(&t_star) {
            probust.grid.push(t_star);
            probust.grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("Added t* = {:.4} to grid", t_star);
        }

        // --- (2) Decision Update Step: refine release vector c ---
        decision = optimize_decision(&probust, decision, max_inner_iter);

        // --- (3) Evaluate updated probability ---
        let prob = probust.evaluate_probability(&decision);
        println!("Updated φ(c) = {:.4}", prob);

        // --- (4) Convergence Check ---
        if (prob - last_prob).abs() < 1e-3 {
            println!("Converged.");
            break;
        }
        last_prob = prob;
    }

    (probust, decision)
}

