mod probust_constraint;

use nalgebra::DMatrix;
use probust_constraint::{
    reservoir_level_continuous, check_deterministic_constraints, ProbustConstraint, Decision,
};

/// Sample objective function f(c): here defined as the total release over all subintervals.
fn objective(decision: &Decision) -> f64 {
    decision.release.iter().sum()
}

fn main() {
    // Define the time grid over [0, T].
    // For instance, let T = 10.0 with 5 subintervals, giving 6 nodes: t₀, t₁, ..., t₅.
    let T = 10.0;
    let num_intervals = 5;
    let dt = T / num_intervals as f64;
    let grid: Vec<f64> = (0..=num_intervals).map(|i| i as f64 * dt).collect();

    // Define a decision with piecewise constant water release.
    // Each component represents the integrated release over the subinterval.
    let decision = Decision {
        initial_level: 100.0,
        release: vec![8.0; num_intervals],
    };

    // Define a covariance matrix σ for the integrated inflows over each subinterval.
    // Here we use a diagonal matrix with variance 4.0 (i.e. no covariance among intervals).
    let sigma = DMatrix::from_diagonal_element(num_intervals, num_intervals, 4.0);

    // Create an instance of ProbustConstraint.
    // We require that the reservoir level remains between 90 and 110 with probability at least 0.9.
    let probust_constraint = ProbustConstraint::new(
        0.9,      // safety_level
        grid.clone(),
        90.0,     // lower_threshold
        110.0,    // upper_threshold
        10_000,   // num_samples
        sigma,
    );

    // Evaluate the joint probability φ(c) that the decision satisfies the joint probabilistic constraint.
    let prob = probust_constraint.evaluate_probability(&decision);
    println!("Estimated joint probability: {:.4}", prob);
    if prob >= probust_constraint.safety_level {
        println!("Decision is feasible under the joint probust constraint.");
    } else {
        println!("Decision is NOT feasible under the joint probust constraint.");
    }

    // Check the deterministic constraints for the decision set X:
    // X = { x ∈ ℝⁿ | 0 ≤ x_i ≤ x̄ for all i, and ∑ x_i ≤ B(T) }.
    let x_bar = 10.0; // upper bound per component
    let B_T = 40.0;   // total upper bound
    let deterministic_feasible = check_deterministic_constraints(&decision, x_bar, B_T);
    if deterministic_feasible {
        println!("Decision satisfies the deterministic constraints for set X.");
    } else {
        println!("Decision does NOT satisfy the deterministic constraints for set X.");
    }

    // Evaluate the continuous reservoir level at a specific time t ∈ [0, T]
    // according to Equation (37):
    //   l(t) = l₀ + ⟨A(t), ξ⟩ + B(t) - [∑_{j=1}^{i(t)} x_j*(t_j - t_{j-1}) + x_{i(t)+1}*(t - t_{i(t)})].
    // Define functions for A(t) and B(t). For example:
    // Let A(t) = [t, 0.5*t] for a 2-dimensional parameter ξ and B(t) = 2*t.
    let a_func = |t: f64| -> Vec<f64> { vec![t, 0.5 * t] };
    let b_func = |t: f64| -> f64 { 2.0 * t };
    // Define parameter vector ξ (dimension 2).
    let xi = vec![2.0, 3.0];
    // Choose a time at which to evaluate the reservoir level.
    let t = 7.3;
    let level = reservoir_level_continuous(t, decision.initial_level, &xi, a_func, b_func, &grid, &decision.release);
    println!("At time t = {:.2}, the reservoir level is {:.2}", t, level);

    // Evaluate the objective function f(c) (here, total release).
    let obj = objective(&decision);
    println!("Objective value (total release): {:.4}", obj);
}
