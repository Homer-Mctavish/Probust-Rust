use std::collections::BTreeMap;
use rand_distr::{Normal, Distribution};
use chrono::{DateTime, Duration, Utc};
use reqwest::blocking::Client;
use serde::Deserialize;

// Define a simple time-series structure for reservoir levels
#[derive(Debug, Clone)]
struct ReservoirState {
    timestamp: DateTime<Utc>,
    inflow: f64,
    level: f64,
}

// Parameters for simulation
const INITIAL_LEVEL: f64 = 1000.0;
const MIN_LEVEL: f64 = 500.0;
const SIMULATION_HORIZON_HOURS: i64 = 48;
const DECISION_INTERVAL_HOURS: i64 = 6;
const NUM_SCENARIOS: usize = 100;

// Simulate inflow using a normal distribution
fn simulate_inflow(mean: f64, std_dev: f64, hours: i64) -> Vec<f64> {
    let normal = Normal::new(mean, std_dev).unwrap();
    (0..hours).map(|_| normal.sample(&mut rand::thread_rng())).collect()
}

// Evaluate a scenario given a control policy (constant consumption)
fn evaluate_scenario(inflows: &[f64], consumption: f64) -> Vec<f64> {
    let mut levels = Vec::new();
    let mut level = INITIAL_LEVEL;
    for &inflow in inflows {
        level += inflow - consumption;
        levels.push(level);
    }
    levels
}

// Query historical inflow mean/std from TimeScaleDB via REST API (simplified for concept)
#[derive(Debug, Deserialize)]
struct TimeScaleResponse {
    mean: f64,
    stddev: f64,
}

fn fetch_inflow_stats_from_timescaledb(client: &Client) -> (f64, f64) {
    let resp: TimeScaleResponse = client
    .get("http://localhost:8000/api/inflow_stats")
    .send()
    .unwrap()
    .json()
    .unwrap();
    (resp.mean, resp.stddev)
}

fn main() {
    let client = Client::new();
    let (mean_inflow, stddev_inflow) = fetch_inflow_stats_from_timescaledb(&client);
    println!("Inflow stats: mean = {}, stddev = {}", mean_inflow, stddev_inflow);

    let now = Utc::now();
    let mut critical_times = BTreeMap::new();

    // Simulate scenarios and find times where level dips below MIN_LEVEL
    for _ in 0..NUM_SCENARIOS {
        let inflows = simulate_inflow(mean_inflow, stddev_inflow, SIMULATION_HORIZON_HOURS);
        let levels = evaluate_scenario(&inflows, 20.0);

        for (i, &level) in levels.iter().enumerate() {
            if level < MIN_LEVEL {
                let timestamp = now + Duration::hours(i as i64);
                *critical_times.entry(timestamp).or_insert(0) += 1;
            }
        }
    }

    println!("\nTime intervals with most constraint violations:");
    for (time, count) in critical_times.iter().filter(|&(_, &v)| v > (NUM_SCENARIOS as f64 * 0.05) as usize) {
        println!("{:?} - Violations in {} scenarios", time, count);
    }

    println!("\nThese are the time intervals to refine with adaptive constraints.");
}
