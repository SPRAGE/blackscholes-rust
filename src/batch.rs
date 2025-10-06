//! Batch APIs for pricing and greeks, with optional parallelism via the `parallel` feature.

use crate::{AllGreeksGeneric, GreeksGeneric, Inputs, Pricing};

// Precision-selected type aliases for batch outputs
#[cfg(feature = "precision-f64")] type Num = f64;
#[cfg(feature = "precision-f32")] type Num = f32;
#[cfg(feature = "precision-f64")] type AllGreeksSelected = AllGreeksGeneric<f64>;
#[cfg(feature = "precision-f32")] type AllGreeksSelected = AllGreeksGeneric<f32>;

/// Compute `calc_all_greeks()` for a slice of inputs, serially.
pub fn all_greeks_batch(inputs: &[Inputs]) -> Vec<Result<AllGreeksSelected, crate::BlackScholesError>> {
    inputs.iter().map(|inp| inp.calc_all_greeks_generic()).collect()
}

/// Compute price for a slice of inputs, serially.
pub fn price_batch(inputs: &[Inputs]) -> Vec<Result<Num, crate::BlackScholesError>> {
    inputs.iter().map(|inp| inp.calc_price()).collect()
}

/// Parallel all-greeks batch if the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
pub fn all_greeks_batch_par(
    inputs: &[Inputs],
) -> Vec<Result<AllGreeksSelected, crate::BlackScholesError>> {
    use rayon::prelude::*;
    inputs.par_iter().map(|inp| inp.calc_all_greeks_generic()).collect()
}

/// Parallel price batch if the `parallel` feature is enabled.
#[cfg(feature = "parallel")]
pub fn price_batch_par(inputs: &[Inputs]) -> Vec<Result<Num, crate::BlackScholesError>> {
    use rayon::prelude::*;
    inputs.par_iter().map(|inp| inp.calc_price()).collect()
}
