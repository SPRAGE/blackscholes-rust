//! Batch APIs for pricing and greeks, with optional parallelism via the `parallel` feature.

use crate::{AllGreeks, Greeks, Inputs, Pricing};

/// Compute `calc_all_greeks()` for a slice of inputs, serially.
pub fn all_greeks_batch(inputs: &[Inputs]) -> Vec<Result<AllGreeks, crate::BlackScholesError>> {
    inputs.iter().map(|inp| inp.calc_all_greeks()).collect()
}

/// Compute price for a slice of inputs, serially.
pub fn price_batch(inputs: &[Inputs]) -> Vec<Result<f64, crate::BlackScholesError>> {
    inputs.iter().map(|inp| inp.calc_price()).collect()
}

/// Parallel all-greeks batch if the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
pub fn all_greeks_batch_par(
    inputs: &[Inputs],
) -> Vec<Result<AllGreeks, crate::BlackScholesError>> {
    use rayon::prelude::*;
    inputs.par_iter().map(|inp| inp.calc_all_greeks()).collect()
}

/// Parallel price batch if the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
pub fn price_batch_par(inputs: &[Inputs]) -> Vec<Result<f64, crate::BlackScholesError>> {
    use rayon::prelude::*;
    inputs.par_iter().map(|inp| inp.calc_price()).collect()
}
