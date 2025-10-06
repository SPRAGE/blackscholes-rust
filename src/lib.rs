//! This library provides an simple, lightweight, and efficient (though not heavily optimized) implementation of the Black-Scholes-Merton model for pricing European options.
//!
//! Provides methods for pricing options, calculating implied volatility, and calculating the first, second, and third order Greeks.
//!
//! ### Example:
//! ```
//! use blackscholes::{Inputs, OptionType, Pricing};
//! let inputs = Inputs::new(OptionType::Call, 100.0, 100.0, None, 0.05, 0.2, 20.0/365.25, Some(0.2));
//! let price = inputs.calc_price().unwrap();
//! ```
//!
//! Criterion benchmark can be ran by running:
//! ```bash
//! cargo bench
//! ```
//!
//! See the [Github Repo](https://github.com/hayden4r4/blackscholes-rust/tree/master) for full source code.  Other implementations such as a [npm WASM package](https://www.npmjs.com/package/@haydenr4/blackscholes_wasm) and a [python module](https://pypi.org/project/blackscholes/) are also available.
//!
//! Errors
//! -------
//! This crate returns a typed error `BlackScholesError` instead of strings. You can match on variants:
//!
//! ```rust
//! use blackscholes::{Inputs, OptionType, Pricing, BlackScholesError};
//!
//! let inputs = Inputs::new(OptionType::Call, 100.0, 100.0, None, 0.05, 0.01, 0.0, Some(0.2));
//! match inputs.calc_price() {
//!     Ok(price) => println!("price = {}", price),
//!     Err(BlackScholesError::TimeToMaturityZero) => println!("t must be > 0"),
//!     Err(e) => println!("error: {}", e),
//! }
//! ```
//!
//! Common variants include `MissingSigma`, `MissingPrice`, `TimeToMaturityZero`, `InvalidLogSK`, and `ConvergenceFailed` (for IV).
//!
//! Parallel batch
//! --------------
//! Enable the `parallel` feature to compute batches with Rayon:
//!
//! ```ignore
//! # features = ["parallel"]
//! use blackscholes::{batch::all_greeks_batch_par, Inputs, OptionType};
//! let inputs = vec![Inputs::new(OptionType::Call, 100.0, 100.0, None, 0.05, 0.01, 0.25, Some(0.2)); 8_000];
//! let results = all_greeks_batch_par(&inputs);
//! ```

pub use greeks::{AllGreeksGeneric, GreeksGeneric};
pub use implied_volatility::ImpliedVolatility;
pub use inputs::OptionType;
// Will migrate to aliasing generic version; keep original struct for transition.
pub use crate::error::BlackScholesError;
use lets_be_rational::normal_distribution::{standard_normal_cdf, standard_normal_pdf};
pub use pricing::Pricing;

mod greeks;
mod implied_volatility;
mod inputs;
mod numeric;
mod generic_inputs;
pub mod lets_be_rational;
mod pricing;
mod error;
pub mod batch;

pub(crate) const DAYS_PER_YEAR: f64 = 365.25;

pub(crate) const A: f64 = 4.626_275_3e-1;
pub(crate) const B: f64 = -1.168_519_2e-2;
pub(crate) const C: f64 = 9.635_418_5e-4;
pub(crate) const D: f64 = 7.535_022_5e-5;
pub(crate) const _E: f64 = 1.424_516_45e-5;
pub(crate) const F: f64 = -2.102_376_9e-5;

// Transitional: expose generic inputs internally; later feature flags will decide alias.
pub use generic_inputs::InputsGeneric;

// Precision feature gating
#[cfg(all(feature = "precision-f32", feature = "precision-f64"))]
compile_error!("Enable only one of precision-f32 or precision-f64 features");

#[cfg(feature = "precision-f64")]
pub type InputsF64 = InputsGeneric<f64>;
#[cfg(feature = "precision-f32")]
pub type InputsF32 = InputsGeneric<f32>;

// Backwards compatible alias `InputsSelected` for feature-based selection
#[cfg(feature = "precision-f64")]
pub type InputsSelected = InputsGeneric<f64>;
#[cfg(feature = "precision-f32")]
pub type InputsSelected = InputsGeneric<f32>;

// Public stable name `Inputs` now maps directly to the feature-selected generic version.
pub type Inputs = InputsSelected;

/// Calculates the d1 and d2 values for the option.
/// # Requires
/// s, k, r, q, t, sigma.
/// # Returns
/// Tuple (f64, f64) of (d1, d2)
#[inline(always)]
#[cfg(feature = "precision-f64")]
pub(crate) fn calc_d1d2(inputs: &Inputs) -> Result<(f64, f64), BlackScholesError> {
    let sigma = inputs.sigma.ok_or(BlackScholesError::MissingSigma)?;
    // Calculating numerator of d1
    let part1 = (inputs.s / inputs.k).ln();

    if !part1.is_finite() {
        return Err(BlackScholesError::InvalidLogSK);
    }

    let part2 = (inputs.r - inputs.q + (sigma.powi(2)) / 2.0) * inputs.t;
    let numd1 = part1 + part2;

    // Calculating denominator of d1 and d2
    if inputs.t == 0.0 {
        return Err(BlackScholesError::TimeToMaturityZero);
    }

    let den = sigma * (inputs.t.sqrt());

    let d1 = numd1 / den;
    let d2 = d1 - den;

    Ok((d1, d2))
}

/// Calculates the nd1 and nd2 values for the option.
/// # Requires
/// s, k, r, q, t, sigma
/// # Returns
/// Tuple (f64, f64) of (nd1, nd2)
#[inline(always)]
#[cfg(feature = "precision-f64")]
pub(crate) fn calc_nd1nd2(inputs: &Inputs) -> Result<(f64, f64), BlackScholesError> {
    let (d1, d2) = calc_d1d2(inputs)?;

    // Calculates the nd1 and nd2 values
    // Checks if OptionType is Call or Put
    match inputs.option_type {
        OptionType::Call => Ok((standard_normal_cdf(d1), standard_normal_cdf(d2))),
        OptionType::Put => Ok((standard_normal_cdf(-d1), standard_normal_cdf(-d2))),
    }
}

/// # Returns
/// f64 of the derivative of the nd1.
#[inline(always)]
#[cfg(feature = "precision-f64")]
pub fn calc_nprimed1(inputs: &Inputs) -> Result<f64, BlackScholesError> {
    let (d1, _) = calc_d1d2(inputs)?;

    // Get the standard n probability density function value of d1
    let nprimed1 = standard_normal_pdf(d1);
    Ok(nprimed1)
}

/// # Returns
/// f64 of the derivative of the nd2.
#[inline(always)]
#[cfg(feature = "precision-f64")]
pub(crate) fn calc_nprimed2(inputs: &Inputs) -> Result<f64, BlackScholesError> {
    let (_, d2) = calc_d1d2(inputs)?;

    // Get the standard n probability density function value of d1
    let nprimed2 = standard_normal_pdf(d2);
    Ok(nprimed2)
}

// ================= Generic transitional helpers =================
use crate::numeric::ModelNum;
// Removed redundant re-import of InputsGeneric (already publicly re-exported above)

#[inline(always)]
pub(crate) fn calc_d1d2_generic<T: ModelNum>(inputs: &InputsGeneric<T>) -> Result<(T, T), BlackScholesError> {
    let sigma = inputs.sigma.ok_or(BlackScholesError::MissingSigma)?;
    let part1 = (inputs.s / inputs.k).ln();
    if !part1.is_finite() {
        return Err(BlackScholesError::InvalidLogSK);
    }
    let part2 = (inputs.r - inputs.q + sigma * sigma / (T::ONE + T::ONE)) * inputs.t; // (sigma^2)/2
    let numd1 = part1 + part2;
    if inputs.t == T::ZERO { return Err(BlackScholesError::TimeToMaturityZero); }
    let sqrt_t = inputs.t.sqrt();
    let den = sigma * sqrt_t;
    let d1 = numd1 / den;
    let d2 = d1 - den;
    Ok((d1, d2))
}

#[inline(always)]
pub(crate) fn calc_nd1nd2_generic<T: ModelNum>(inputs: &InputsGeneric<T>) -> Result<(T, T), BlackScholesError> {
    let (d1, d2) = calc_d1d2_generic(inputs)?;
    match inputs.option_type {
        OptionType::Call => Ok((T::from(standard_normal_cdf(d1.to_f64().unwrap())).unwrap(), T::from(standard_normal_cdf(d2.to_f64().unwrap())).unwrap())),
        OptionType::Put => Ok((T::from(standard_normal_cdf((-d1).to_f64().unwrap())).unwrap(), T::from(standard_normal_cdf((-d2).to_f64().unwrap())).unwrap())),
    }
}

#[inline(always)]
pub(crate) fn calc_nprimed1_generic<T: ModelNum>(inputs: &InputsGeneric<T>) -> Result<T, BlackScholesError> {
    let (d1, _) = calc_d1d2_generic(inputs)?;
    Ok(T::from(standard_normal_pdf(d1.to_f64().unwrap())).unwrap())
}

#[inline(always)]
pub(crate) fn calc_nprimed2_generic<T: ModelNum>(inputs: &InputsGeneric<T>) -> Result<T, BlackScholesError> {
    let (_, d2) = calc_d1d2_generic(inputs)?;
    Ok(T::from(standard_normal_pdf(d2.to_f64().unwrap())).unwrap())
}
