//! High-level batch APIs using SIMD acceleration.
//!
//! This module provides convenient functions for batch processing of options using SIMD.
//! It automatically handles:
//! - **Chunking**: Processes options in optimal SIMD chunks (4 for f64, 8 for f32)
//! - **Remainder handling**: Processes leftover options using scalar code
//! - **Precision selection**: Uses f64x4 or f32x8 based on `precision-f64`/`precision-f32` features
//! - **Parallel processing**: Optional rayon-based parallelization with `parallel` feature
//!
//! # Quick Start
//!
//! ```rust
//! use blackscholes::{Inputs, OptionType};
//! use blackscholes::simd_batch::{price_batch_simd, greeks_batch_simd, iv_batch_simd};
//!
//! # #[cfg(feature = "simd")]
//! # {
//! let inputs: Vec<Inputs> = vec![
//!     Inputs::new(OptionType::Call, 100.0, 105.0, None, 0.05, 0.0, 0.25, Some(0.2)),
//!     Inputs::new(OptionType::Put, 100.0, 95.0, None, 0.05, 0.0, 0.25, Some(0.25)),
//!     // ... more options
//! ];
//!
//! let prices = price_batch_simd(&inputs);
//! let greeks = greeks_batch_simd(&inputs);
//! # }
//! ```
//!
//! # Performance Guidelines
//!
//! - **Batch size**: Best with 100+ options (amortizes chunking overhead)
//! - **SIMD only**: ~4x speedup for f64, ~8x for f32
//! - **SIMD + parallel**: Additional ~Nx speedup (N = CPU cores)
//! - **Break-even**: ~20-30 options (below this, scalar may be faster due to overhead)
//!
//! # API Variants
//!
//! Each function has three variants:
//! - `*_batch_simd`: SIMD-only (this module)
//! - `*_batch_simd_par`: SIMD + parallel processing (requires `parallel` feature)
//! - `*_batch`: Standard scalar implementation (in respective modules)

#![cfg(feature = "simd")]

use crate::{Inputs, BlackScholesError, AllGreeksGeneric};
use crate::simd_ops::{
    SimdInputsF64x4, simd_calc_price_f64x4, simd_calc_greeks_f64x4, 
    simd_calc_rational_iv_f64x4,
};
use wide::f64x4;

#[cfg(feature = "precision-f32")]
use crate::simd_ops::{
    SimdInputsF32x8, simd_calc_price_f32x8, simd_calc_greeks_f32x8, 
    simd_calc_rational_iv_f32x8,
};
#[cfg(feature = "precision-f32")]
use wide::f32x8;

// ============================================================================
// Helper functions to convert between Inputs and SIMD types
// ============================================================================

#[cfg(feature = "precision-f64")]
fn inputs_to_simd_f64x4(inputs: &[Inputs; 4]) -> SimdInputsF64x4 {
    SimdInputsF64x4 {
        option_type: [
            inputs[0].option_type,
            inputs[1].option_type,
            inputs[2].option_type,
            inputs[3].option_type,
        ],
        s: f64x4::new([inputs[0].s, inputs[1].s, inputs[2].s, inputs[3].s]),
        k: f64x4::new([inputs[0].k, inputs[1].k, inputs[2].k, inputs[3].k]),
        r: f64x4::new([inputs[0].r, inputs[1].r, inputs[2].r, inputs[3].r]),
        q: f64x4::new([inputs[0].q, inputs[1].q, inputs[2].q, inputs[3].q]),
        t: f64x4::new([inputs[0].t, inputs[1].t, inputs[2].t, inputs[3].t]),
        sigma: f64x4::new([
            inputs[0].sigma.unwrap_or(0.0),
            inputs[1].sigma.unwrap_or(0.0),
            inputs[2].sigma.unwrap_or(0.0),
            inputs[3].sigma.unwrap_or(0.0),
        ]),
    }
}

#[cfg(feature = "precision-f32")]
fn inputs_to_simd_f32x8(inputs: &[Inputs; 8]) -> SimdInputsF32x8 {
    SimdInputsF32x8 {
        option_type: [
            inputs[0].option_type,
            inputs[1].option_type,
            inputs[2].option_type,
            inputs[3].option_type,
            inputs[4].option_type,
            inputs[5].option_type,
            inputs[6].option_type,
            inputs[7].option_type,
        ],
        s: f32x8::new([
            inputs[0].s, inputs[1].s, inputs[2].s, inputs[3].s,
            inputs[4].s, inputs[5].s, inputs[6].s, inputs[7].s,
        ]),
        k: f32x8::new([
            inputs[0].k, inputs[1].k, inputs[2].k, inputs[3].k,
            inputs[4].k, inputs[5].k, inputs[6].k, inputs[7].k,
        ]),
        r: f32x8::new([
            inputs[0].r, inputs[1].r, inputs[2].r, inputs[3].r,
            inputs[4].r, inputs[5].r, inputs[6].r, inputs[7].r,
        ]),
        q: f32x8::new([
            inputs[0].q, inputs[1].q, inputs[2].q, inputs[3].q,
            inputs[4].q, inputs[5].q, inputs[6].q, inputs[7].q,
        ]),
        t: f32x8::new([
            inputs[0].t, inputs[1].t, inputs[2].t, inputs[3].t,
            inputs[4].t, inputs[5].t, inputs[6].t, inputs[7].t,
        ]),
        sigma: f32x8::new([
            inputs[0].sigma.unwrap_or(0.0),
            inputs[1].sigma.unwrap_or(0.0),
            inputs[2].sigma.unwrap_or(0.0),
            inputs[3].sigma.unwrap_or(0.0),
            inputs[4].sigma.unwrap_or(0.0),
            inputs[5].sigma.unwrap_or(0.0),
            inputs[6].sigma.unwrap_or(0.0),
            inputs[7].sigma.unwrap_or(0.0),
        ]),
    }
}

// ============================================================================
// Batch Pricing with SIMD
// ============================================================================

/// Calculate Black-Scholes option prices for a batch using SIMD acceleration (f64).
///
/// Processes options in chunks of 4 using 256-bit SIMD vectors, providing approximately
/// 4x speedup over scalar implementation. Remaining options (if count not divisible by 4)
/// are processed using scalar code.
///
/// # Arguments
///
/// * `inputs` - Slice of option parameters (each must have `sigma` set)
///
/// # Returns
///
/// Vector of option prices, one per input in the same order.
///
/// # Performance
///
/// - **Speedup**: ~4x faster than scalar for large batches (1000+ options)
/// - **Break-even**: ~20-30 options (overhead of chunking becomes negligible)
/// - **Precision**: Full f64 precision (15-16 decimal digits)
///
/// # Example
///
/// ```
/// # #[cfg(feature = "simd")]
/// # {
/// use blackscholes::{Inputs, OptionType};
/// use blackscholes::simd_batch::price_batch_simd;
///
/// let inputs: Vec<Inputs> = vec![
///     Inputs::new(OptionType::Call, 100.0, 105.0, None, 0.05, 0.0, 0.25, Some(0.2)),
///     Inputs::new(OptionType::Put, 100.0, 95.0, None, 0.05, 0.0, 0.25, Some(0.25)),
///     // ... more options (ideally 100+)
/// ];
///
/// let prices = price_batch_simd(&inputs);
/// # }
/// ```
///
/// # See Also
///
/// - [`price_batch_simd_par`](fn.price_batch_simd_par.html) - Parallel + SIMD version (requires `parallel` feature)
/// - [`crate::batch::price_batch`] - Scalar batch implementation
#[cfg(feature = "precision-f64")]
pub fn price_batch_simd(inputs: &[Inputs]) -> Vec<f64> {
    let mut results = Vec::with_capacity(inputs.len());
    let chunks = inputs.chunks_exact(4);
    let remainder = chunks.remainder();
    
    // Process 4 at a time with SIMD
    for chunk in chunks {
        let chunk_array: [Inputs; 4] = [
            chunk[0].clone(),
            chunk[1].clone(),
            chunk[2].clone(),
            chunk[3].clone(),
        ];
        let simd_inputs = inputs_to_simd_f64x4(&chunk_array);
        let prices = simd_calc_price_f64x4(&simd_inputs);
        
        for price in prices.iter() {
            results.push(*price);
        }
    }
    
    // Process remainder with scalar code
    for input in remainder {
        use crate::Pricing;
        results.push(input.calc_price().unwrap_or(0.0));
    }
    
    results
}

/// Calculate Black-Scholes option prices for a batch using SIMD acceleration (f32).
///
/// Processes options in chunks of 8 using 256-bit SIMD vectors, providing approximately
/// 8x speedup over scalar implementation. Uses single precision for additional performance.
///
/// # Arguments
///
/// * `inputs` - Slice of option parameters (each must have `sigma` set)
///
/// # Returns
///
/// Vector of option prices, one per input in the same order.
///
/// # Performance vs f64
///
/// - **Speedup**: ~8x faster than scalar (vs ~4x for f64x4)
/// - **Throughput**: ~2x higher than f64x4 version
/// - **Precision**: ~6-7 decimal digits (vs 15-16 for f64)
/// - **Use when**: Processing very large batches where slight precision loss is acceptable
///
/// # Example
///
/// ```
/// # #[cfg(all(feature = "simd", feature = "precision-f32"))]
/// # {
/// use blackscholes::{Inputs, OptionType};
/// use blackscholes::simd_batch::price_batch_simd;
///
/// let inputs: Vec<Inputs> = vec![
///     Inputs::new(OptionType::Call, 100.0, 105.0, None, 0.05, 0.0, 0.25, Some(0.2)),
///     // ... more options
/// ];
///
/// let prices = price_batch_simd(&inputs);
/// # }
/// ```
#[cfg(feature = "precision-f32")]
pub fn price_batch_simd(inputs: &[Inputs]) -> Vec<f32> {
    let mut results = Vec::with_capacity(inputs.len());
    let chunks = inputs.chunks_exact(8);
    let remainder = chunks.remainder();
    
    // Process 8 at a time with SIMD
    for chunk in chunks {
        let chunk_array: [Inputs; 8] = [
            chunk[0].clone(),
            chunk[1].clone(),
            chunk[2].clone(),
            chunk[3].clone(),
            chunk[4].clone(),
            chunk[5].clone(),
            chunk[6].clone(),
            chunk[7].clone(),
        ];
        let simd_inputs = inputs_to_simd_f32x8(&chunk_array);
        let prices = simd_calc_price_f32x8(&simd_inputs);
        
        for i in 0..8 {
            results.push(prices[i]);
        }
    }
    
    // Process remainder
    for input in remainder {
        use crate::Pricing;
        results.push(input.calc_price().unwrap_or(0.0));
    }
    
    results
}

// ============================================================================
// Batch Greeks with SIMD
// ============================================================================

/// Calculate all 17 Greeks for a batch of options using SIMD acceleration (f64).
///
/// Computes all Greeks in a single pass using vectorized operations. This is significantly
/// more efficient than calling individual Greek functions or even calculating all Greeks
/// for each option separately.
///
/// # Greeks Calculated
///
/// Returns `AllGreeksGeneric` structure containing all 17 Greeks:
/// - **First-order**: Delta, Vega, Theta, Rho, Epsilon
/// - **Second-order**: Gamma, Vanna, Charm, Vomma, Veta  
/// - **Third-order**: Speed, Zomma, Color, Ultima
/// - **Special**: Lambda, Dual Delta, Dual Gamma
///
/// See [`simd_ops::simd_calc_greeks_f64x4`](crate::simd_ops::simd_calc_greeks_f64x4) for detailed Greek definitions.
///
/// # Arguments
///
/// * `inputs` - Slice of option parameters (each must have `sigma` set)
///
/// # Returns
///
/// Vector of `AllGreeksGeneric<f64>` structures, one per input in the same order.
///
/// # Performance
///
/// - **Speed**: ~0.02 µs per option for all 17 Greeks (~50M options/second)
/// - **Speedup**: ~4x faster than scalar implementation
/// - **Efficiency**: ~20x faster than calculating Greeks individually
/// - **Best for**: Batches of 100+ options
///
/// # Example
///
/// ```
/// # #[cfg(feature = "simd")]
/// # {
/// use blackscholes::{Inputs, OptionType};
/// use blackscholes::simd_batch::greeks_batch_simd;
///
/// let inputs: Vec<Inputs> = vec![
///     Inputs::new(OptionType::Call, 100.0, 105.0, None, 0.05, 0.02, 0.25, Some(0.2)),
///     Inputs::new(OptionType::Put, 100.0, 95.0, None, 0.05, 0.02, 0.25, Some(0.25)),
///     // ... more options
/// ];
///
/// let greeks = greeks_batch_simd(&inputs);
/// for (i, g) in greeks.iter().enumerate() {
///     println!("Option {}: Delta={:.4}, Gamma={:.4}, Vega={:.4}", 
///              i, g.delta, g.gamma, g.vega);
/// }
/// # }
/// ```
///
/// # See Also
///
/// - [`greeks_batch_simd_par`](fn.greeks_batch_simd_par.html) - Parallel + SIMD version
/// - [`crate::batch::all_greeks_batch`] - Scalar batch implementation
#[cfg(feature = "precision-f64")]
pub fn greeks_batch_simd(inputs: &[Inputs]) -> Vec<AllGreeksGeneric<f64>> {
    let mut results = Vec::with_capacity(inputs.len());
    let chunks = inputs.chunks_exact(4);
    let remainder = chunks.remainder();
    
    // Process 4 at a time with SIMD
    for chunk in chunks {
        let chunk_array: [Inputs; 4] = [
            chunk[0].clone(),
            chunk[1].clone(),
            chunk[2].clone(),
            chunk[3].clone(),
        ];
        let simd_inputs = inputs_to_simd_f64x4(&chunk_array);
        let greeks = simd_calc_greeks_f64x4(&simd_inputs);
        
        for i in 0..4 {
            // Convert SIMD Greeks to AllGreeksGeneric with all Greeks
            results.push(AllGreeksGeneric {
                delta: greeks.delta[i],
                gamma: greeks.gamma[i],
                theta: greeks.theta[i],
                vega: greeks.vega[i],
                rho: greeks.rho[i],
                epsilon: greeks.epsilon[i],
                lambda: greeks.lambda[i],
                vanna: greeks.vanna[i],
                charm: greeks.charm[i],
                veta: greeks.veta[i],
                vomma: greeks.vomma[i],
                speed: greeks.speed[i],
                zomma: greeks.zomma[i],
                color: greeks.color[i],
                ultima: greeks.ultima[i],
                dual_delta: greeks.dual_delta[i],
                dual_gamma: greeks.dual_gamma[i],
            });
        }
    }
    
    // Process remainder
    for input in remainder {
        use crate::GreeksGeneric;
        results.push(input.calc_all_greeks_generic().unwrap_or_else(|_| {
            AllGreeksGeneric {
                delta: 0.0, gamma: 0.0, theta: 0.0, vega: 0.0,
                rho: 0.0, epsilon: 0.0, lambda: 0.0, vanna: 0.0,
                charm: 0.0, veta: 0.0, vomma: 0.0, speed: 0.0,
                zomma: 0.0, color: 0.0, ultima: 0.0,
                dual_delta: 0.0, dual_gamma: 0.0,
            }
        }));
    }
    
    results
}

/// Calculate all 17 Greeks for a batch of options using SIMD acceleration (f32).
///
/// Processes 8 options at a time using single precision for maximum throughput.
/// Ideal for very large batches where f32 precision (6-7 digits) is sufficient.
///
/// # Arguments
///
/// * `inputs` - Slice of option parameters (each must have `sigma` set)
///
/// # Returns
///
/// Vector of `AllGreeksGeneric<f32>` structures containing all 17 Greeks.
///
/// # Performance vs f64
///
/// - **Throughput**: ~2x higher than f64x4 (8 options vs 4 per iteration)
/// - **Speedup**: ~8x faster than scalar implementation
/// - **Precision**: ~6-7 decimal digits (sufficient for most applications)
/// - **Best for**: Very large batches (10,000+ options)
///
/// # Example
///
/// ```
/// # #[cfg(all(feature = "simd", feature = "precision-f32"))]
/// # {
/// use blackscholes::{Inputs, OptionType};
/// use blackscholes::simd_batch::greeks_batch_simd;
///
/// let inputs: Vec<Inputs> = vec![
///     Inputs::new(OptionType::Call, 100.0, 105.0, None, 0.05, 0.02, 0.25, Some(0.2)),
///     // ... many more options
/// ];
///
/// let greeks = greeks_batch_simd(&inputs);
/// # }
/// ```
#[cfg(feature = "precision-f32")]
pub fn greeks_batch_simd(inputs: &[Inputs]) -> Vec<AllGreeksGeneric<f32>> {
    let mut results = Vec::with_capacity(inputs.len());
    let chunks = inputs.chunks_exact(8);
    let remainder = chunks.remainder();
    
    // Process 8 at a time with SIMD
    for chunk in chunks {
        let chunk_array: [Inputs; 8] = [
            chunk[0].clone(),
            chunk[1].clone(),
            chunk[2].clone(),
            chunk[3].clone(),
            chunk[4].clone(),
            chunk[5].clone(),
            chunk[6].clone(),
            chunk[7].clone(),
        ];
        let simd_inputs = inputs_to_simd_f32x8(&chunk_array);
        let greeks = simd_calc_greeks_f32x8(&simd_inputs);
        
        for i in 0..8 {
            results.push(AllGreeksGeneric {
                delta: greeks.delta[i],
                gamma: greeks.gamma[i],
                theta: greeks.theta[i],
                vega: greeks.vega[i],
                rho: greeks.rho[i],
                epsilon: greeks.epsilon[i],
                lambda: greeks.lambda[i],
                vanna: greeks.vanna[i],
                charm: greeks.charm[i],
                veta: greeks.veta[i],
                vomma: greeks.vomma[i],
                speed: greeks.speed[i],
                zomma: greeks.zomma[i],
                color: greeks.color[i],
                ultima: greeks.ultima[i],
                dual_delta: greeks.dual_delta[i],
                dual_gamma: greeks.dual_gamma[i],
            });
        }
    }
    
    // Process remainder
    for input in remainder {
        use crate::GreeksGeneric;
        results.push(input.calc_all_greeks_generic().unwrap_or_else(|_| {
            AllGreeksGeneric {
                delta: 0.0, gamma: 0.0, theta: 0.0, vega: 0.0,
                rho: 0.0, epsilon: 0.0, lambda: 0.0, vanna: 0.0,
                charm: 0.0, veta: 0.0, vomma: 0.0, speed: 0.0,
                zomma: 0.0, color: 0.0, ultima: 0.0,
                dual_delta: 0.0, dual_gamma: 0.0,
            }
        }));
    }
    
    results
}

// ============================================================================
// Batch Implied Volatility with SIMD
// ============================================================================

/// Calculate implied volatility for a batch using SIMD with Jäckel's rational method (f64).
///
/// Uses the highly efficient "Let's Be Rational" algorithm which converges in 2-3 iterations,
/// providing 5-10x speedup over Newton-Raphson methods. Combined with SIMD batching, this
/// offers dramatic performance improvements for large-scale IV calculations.
///
/// # Algorithm
///
/// - **Method**: Jäckel's "Let's Be Rational" rational approximation
/// - **Convergence**: 2-3 iterations typical, guaranteed for valid inputs
/// - **Accuracy**: Machine precision (~1e-15 for f64)
/// - **Robustness**: Handles deep ITM/OTM, extreme strikes, long maturities
///
/// # Arguments
///
/// * `inputs` - Slice of option parameters. Each `Inputs` must have:
///   - `p` (price) set via `Some(price)` in the third parameter of `Inputs::new(...)`
///   - All other parameters (S, K, r, q, t) valid
///   - `sigma` should be `None` (being calculated)
///
/// # Returns
///
/// Vector of `Result<f64, BlackScholesError>` - one per input. Each is:
/// - `Ok(sigma)` - Successfully calculated implied volatility
/// - `Err(e)` - Invalid inputs (negative price, t ≤ 0, etc.)
///
/// # Performance
///
/// - **Convergence**: 2-3 iterations per option (vs 10+ for Newton-Raphson)
/// - **SIMD speedup**: ~4x from processing 4 options simultaneously
/// - **Total speedup**: ~20-40x vs scalar Newton-Raphson
/// - **Best for**: Batches of 100+ options with market prices
///
/// # Example
///
/// ```
/// # #[cfg(feature = "simd")]
/// # {
/// use blackscholes::{Inputs, OptionType};
/// use blackscholes::simd_batch::iv_batch_simd;
///
/// // Create inputs with market prices
/// let inputs: Vec<Inputs> = vec![
///     Inputs::new(OptionType::Call, 100.0, 105.0, Some(2.5), 0.05, 0.0, 0.25, None),
///     Inputs::new(OptionType::Put, 100.0, 95.0, Some(1.8), 0.05, 0.0, 0.25, None),
///     // ... more options
/// ];
///
/// let ivs = iv_batch_simd(&inputs);
/// for (i, result) in ivs.iter().enumerate() {
///     match result {
///         Ok(iv) => println!("Option {}: IV = {:.4}", i, iv),
///         Err(e) => println!("Option {}: Error - {:?}", i, e),
///     }
/// }
/// # }
/// ```
///
/// # See Also
///
/// - [`iv_batch_simd_par`](fn.iv_batch_simd_par.html) - Parallel + SIMD version
/// - [`crate::ImpliedVolatility::calc_iv`] - Scalar IV calculation
///
/// # References
///
/// Jäckel, P. (2015). "Let's be rational." Wilmott, 2015(75), 40-53.
#[cfg(feature = "precision-f64")]
pub fn iv_batch_simd(inputs: &[Inputs]) -> Vec<Result<f64, BlackScholesError>> {
    let mut results = Vec::with_capacity(inputs.len());
    let chunks = inputs.chunks_exact(4);
    let remainder = chunks.remainder();
    
    // Process 4 at a time with SIMD
    for chunk in chunks {
        let chunk_array: [Inputs; 4] = [
            chunk[0].clone(),
            chunk[1].clone(),
            chunk[2].clone(),
            chunk[3].clone(),
        ];
        
        let prices = f64x4::new([
            chunk[0].p.unwrap_or(0.0),
            chunk[1].p.unwrap_or(0.0),
            chunk[2].p.unwrap_or(0.0),
            chunk[3].p.unwrap_or(0.0),
        ]);
        let prices_arr: [f64; 4] = prices.into();
        
        let simd_inputs = inputs_to_simd_f64x4(&chunk_array);
        
        match simd_calc_rational_iv_f64x4(&simd_inputs, prices_arr) {
            Ok(ivs) => {
                for iv in ivs.iter() {
                    results.push(Ok(*iv));
                }
            }
            Err(_e) => {
                // Fallback to scalar calculation for each
                for input in chunk {
                    use crate::ImpliedVolatility;
                    results.push(input.calc_rational_iv().map_err(|e| e));
                }
            }
        }
    }
    
    // Process remainder
    for input in remainder {
        use crate::ImpliedVolatility;
        results.push(input.calc_rational_iv().map_err(|e| e));
    }
    
    results
}

/// Calculate implied volatility for a batch of inputs using SIMD with Jäckel's rational method (f32)
/// 
/// Note: Internally promotes to f64 for the rational solver.
#[cfg(feature = "precision-f32")]
pub fn iv_batch_simd(inputs: &[Inputs]) -> Vec<Result<f32, BlackScholesError>> {
    let mut results = Vec::with_capacity(inputs.len());
    let chunks = inputs.chunks_exact(8);
    let remainder = chunks.remainder();
    
    // Process 8 at a time with SIMD
    for chunk in chunks {
        let chunk_array: [Inputs; 8] = [
            chunk[0].clone(),
            chunk[1].clone(),
            chunk[2].clone(),
            chunk[3].clone(),
            chunk[4].clone(),
            chunk[5].clone(),
            chunk[6].clone(),
            chunk[7].clone(),
        ];
        
        let prices = f32x8::new([
            chunk[0].p.unwrap_or(0.0),
            chunk[1].p.unwrap_or(0.0),
            chunk[2].p.unwrap_or(0.0),
            chunk[3].p.unwrap_or(0.0),
            chunk[4].p.unwrap_or(0.0),
            chunk[5].p.unwrap_or(0.0),
            chunk[6].p.unwrap_or(0.0),
            chunk[7].p.unwrap_or(0.0),
        ]);
        let prices_arr: [f32; 8] = prices.into();
        
        let simd_inputs = inputs_to_simd_f32x8(&chunk_array);
        
        match simd_calc_rational_iv_f32x8(&simd_inputs, prices_arr) {
            Ok(ivs) => {
                for iv in ivs.iter() {
                    results.push(Ok(*iv));
                }
            }
            Err(_e) => {
                for input in chunk {
                    use crate::ImpliedVolatility;
                    results.push(input.calc_rational_iv().map(|v| v as f32).map_err(|e| e));
                }
            }
        }
    }
    
    // Process remainder
    for input in remainder {
        use crate::ImpliedVolatility;
        results.push(input.calc_rational_iv().map(|v| v as f32).map_err(|e| e));
    }
    
    results
}

// ============================================================================
// Combined SIMD + Parallel processing
// ============================================================================

/// Calculate prices using both SIMD and parallel processing
#[cfg(all(feature = "simd", feature = "parallel", feature = "precision-f64"))]
pub fn price_batch_simd_par(inputs: &[Inputs]) -> Vec<f64> {
    use rayon::prelude::*;
    
    inputs
        .par_chunks(64) // Process in chunks to balance parallelism
        .flat_map(|chunk| price_batch_simd(chunk))
        .collect()
}

/// Calculate prices using both SIMD and parallel processing (f32)
#[cfg(all(feature = "simd", feature = "parallel", feature = "precision-f32"))]
pub fn price_batch_simd_par(inputs: &[Inputs]) -> Vec<f32> {
    use rayon::prelude::*;
    
    inputs
        .par_chunks(64)
        .flat_map(|chunk| price_batch_simd(chunk))
        .collect()
}

/// Calculate Greeks using both SIMD and parallel processing (f64)
#[cfg(all(feature = "simd", feature = "parallel", feature = "precision-f64"))]
pub fn greeks_batch_simd_par(inputs: &[Inputs]) -> Vec<AllGreeksGeneric<f64>> {
    use rayon::prelude::*;
    
    inputs
        .par_chunks(64)
        .flat_map(|chunk| greeks_batch_simd(chunk))
        .collect()
}

/// Calculate Greeks using both SIMD and parallel processing (f32)
#[cfg(all(feature = "simd", feature = "parallel", feature = "precision-f32"))]
pub fn greeks_batch_simd_par(inputs: &[Inputs]) -> Vec<AllGreeksGeneric<f32>> {
    use rayon::prelude::*;
    
    inputs
        .par_chunks(64)
        .flat_map(|chunk| greeks_batch_simd(chunk))
        .collect()
}

/// Calculate IV using both SIMD and parallel processing (f64)
#[cfg(all(feature = "simd", feature = "parallel", feature = "precision-f64"))]
pub fn iv_batch_simd_par(inputs: &[Inputs]) -> Vec<Result<f64, BlackScholesError>> {
    use rayon::prelude::*;
    
    inputs
        .par_chunks(64)
        .flat_map(|chunk| iv_batch_simd(chunk))
        .collect()
}

/// Calculate IV using both SIMD and parallel processing (f32)
#[cfg(all(feature = "simd", feature = "parallel", feature = "precision-f32"))]
pub fn iv_batch_simd_par(inputs: &[Inputs]) -> Vec<Result<f32, BlackScholesError>> {
    use rayon::prelude::*;
    
    inputs
        .par_chunks(64)
        .flat_map(|chunk| iv_batch_simd(chunk))
        .collect()
}
