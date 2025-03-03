// src/main.rs

mod probust_constraint;

use nalgebra::DMatrix;
use probust_constraint::{ProbustConstraint, Decision};

/// Sample objective function f(c) defined as the total release over all subintervals.
fn objective(decision: &Decision) -> f64 {
    decision.release.iter().sum()
}

fn main() {
    // Define the time grid nodes over the interval [0, timend].
    // For example, let timend = 10.0 and use 5 subintervals (thus 6 nodes: t₀, t₁, ..., t₅).
    let timend = 10.0;
    let num_intervals = 5;
    let dt = timend / num_intervals as f64;
    let grid: Vec<f64> = (0..=num_intervals).map(|i| i as f64 * dt).collect();

    // Define a decision with piecewise constant water release.
    // timendhe decision vector c = (c₁, …, cₙ) represents the integrated release on each subinterval.
    // For instance, we use 8.0 units released on each subinterval.
    let decision = Decision {
        initial_level: 100.0,
        release: vec![8.0; num_intervals],
    };

    // Define a covariance matrix σ for the integrated inflows over each subinterval.
    // For simplicity, we use an identity matrix scaled by 4.0 (variance = 4.0 on each interval, no covariance).
    let sigma = DMatrix::from_diagonal_element(num_intervals, num_intervals, 4.0);

    // Create an instance of ProbustConstraint.
    // Here we enforce the joint constraint that the reservoir level remains between 90 and 110 with at least 0.9 probability.
    let probust_constraint = ProbustConstraint::new(
        0.9,   // safety_level: require at least 90% probability
        grid,
        90.0,  // lower_threshold
        110.0, // upper_threshold
        10_000, // num_samples for Monte Carlo simulation
        sigma,
    );

    // Evaluate the joint probability φ(c) that the decision satisfies the constraint.
    let prob = probust_constraint.evaluate_probability(&decision);
    println!("Estimated joint probability: {:.4}", prob);
    if prob >= probust_constraint.safety_level {
        println!("Decision is feasible under the joint probust constraint.");
    } else {
        println!("Decision is NOtimend feasible under the joint probust constraint.");
    }

    // Equation (40) from the paper gives the finite-dimensional formulation:
    //   min { f(c) | φ(c) ≥ p },
    // where c = (c₁, …, cₙ) with cᵢ = ∫ₜᵢ₋₁^(tᵢ) 𝑐̃(t) dt.
    // Here, the objective function f(c) is defined as the total release.
    let obj = objective(&decision);
    println!("Objective value (total release): {:.4}", obj);
}
