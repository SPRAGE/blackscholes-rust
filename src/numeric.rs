//! Numeric abstraction layer to enable generic computations over f32/f64 (and later SIMD types).
//! Initially minimal: only what current pricing/greeks/IV paths need.

use core::ops::{Add, Sub, Mul, Div, Neg};
use num_traits::{Float, FloatConst};

/// Trait alias capturing required numeric behaviour for the model.
pub trait ModelNum:
    Copy + Clone + core::fmt::Debug + Float + FloatConst + Send + Sync + 'static +
    Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> + Neg<Output=Self>
{
    const DAYS_PER_YEAR: Self; // 365.25
    const ONE_HUNDRED: Self;   // 100.0 constant used for percentage adjustments
    const ZERO: Self;          // 0
    const ONE: Self;           // 1
    const HALF: Self;          // 0.5
}

impl ModelNum for f64 {
    const DAYS_PER_YEAR: Self = 365.25;
    const ONE_HUNDRED: Self = 100.0;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
}

impl ModelNum for f32 {
    const DAYS_PER_YEAR: Self = 365.25; // minor rounding difference acceptable
    const ONE_HUNDRED: Self = 100.0;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
}
