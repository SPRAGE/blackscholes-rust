//! Error types for the Black-Scholes crate.
//!
//! The crate exposes a typed error, [`BlackScholesError`], instead of string errors.
//! This makes it easy to match on specific failure modes without parsing text.
//!
//! Example
//! =======
//!
//! ```
//! use blackscholes::{Inputs, OptionType, Pricing, BlackScholesError};
//!
//! // t = 0.0 will produce a TimeToMaturityZero error
//! let inputs = Inputs::new(OptionType::Call, 100.0, 100.0, None, 0.05, 0.01, 0.0, Some(0.2));
//! match inputs.calc_price() {
//!     Ok(_) => unreachable!("expected an error for t = 0"),
//!     Err(BlackScholesError::TimeToMaturityZero) => {}
//!     Err(e) => panic!("unexpected error: {e}"),
//! }
//! ```
//!
//! Common variants include:
//! - [`BlackScholesError::MissingSigma`]
//! - [`BlackScholesError::MissingPrice`]
//! - [`BlackScholesError::TimeToMaturityZero`]
//! - [`BlackScholesError::InvalidLogSK`]
//! - [`BlackScholesError::ConvergenceFailed`]

use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BlackScholesError {
    /// Volatility (sigma) was required but not provided.
    MissingSigma,
    /// Option price (p) was required but not provided.
    MissingPrice,
    /// Time to maturity (t) was zero; must be > 0.
    TimeToMaturityZero,
    /// ln(s/k) was not finite (e.g., s or k invalid). Check inputs.
    InvalidLogSK,
    /// Iterative solver failed to converge (e.g., implied volatility).
    ConvergenceFailed,
    /// Generic typed input error with a static message.
    InvalidInput(&'static str),
}

impl fmt::Display for BlackScholesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlackScholesError::MissingSigma => write!(f, "Expected Some(f64) for sigma, received None"),
            BlackScholesError::MissingPrice => write!(f, "Option price (p) is required"),
            BlackScholesError::TimeToMaturityZero => write!(f, "Time to maturity (t) is 0"),
            BlackScholesError::InvalidLogSK => write!(f, "Log(s/k) is not finite"),
            BlackScholesError::ConvergenceFailed => write!(f, "Failed to converge"),
            BlackScholesError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for BlackScholesError {}
