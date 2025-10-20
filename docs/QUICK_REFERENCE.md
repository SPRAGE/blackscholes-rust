# Quick Reference Card

Fast lookup for common tasks in the `blackscholes` crate.

## Installation

```toml
[dependencies]
# Basic usage (f64 precision)
blackscholes = "0.24"

# With SIMD acceleration
blackscholes = { version = "0.24", features = ["simd"] }

# With parallel processing
blackscholes = { version = "0.24", features = ["parallel"] }

# With SIMD + parallel
blackscholes = { version = "0.24", features = ["simd", "parallel"] }

# Single precision (f32)
blackscholes = { version = "0.24", default-features = false, features = ["precision-f32"] }
```

## Single Option Operations

### Price an Option

```rust
use blackscholes::{Inputs, OptionType, Pricing};

let option = Inputs::new(
    OptionType::Call,
    100.0,  // spot price
    105.0,  // strike price
    None,   // market price (not needed for pricing)
    0.05,   // risk-free rate
    0.02,   // dividend yield
    0.25,   // time to maturity (years)
    Some(0.2),  // volatility
);

let price = option.calc_price()?;
```

### Calculate a Single Greek

```rust
use blackscholes::GreeksGeneric;

let delta = option.calc_delta()?;
let gamma = option.calc_gamma()?;
let vega = option.calc_vega()?;
```

### Calculate All Greeks at Once

```rust
let all_greeks = option.calc_all_greeks_generic()?;
println!("Delta: {}, Gamma: {}, Vega: {}", 
         all_greeks.delta, all_greeks.gamma, all_greeks.vega);
```

### Calculate Implied Volatility

```rust
use blackscholes::ImpliedVolatility;

let option_with_price = Inputs::new(
    OptionType::Call,
    100.0, 105.0,
    Some(2.5),  // market price
    0.05, 0.02, 0.25,
    None,  // volatility to be calculated
);

// Standard Newton-Raphson method
let iv = option_with_price.calc_iv()?;

// Faster "Let's Be Rational" method (2-3 iterations)
let iv = option_with_price.calc_rational_iv()?;
```

## Batch Operations (Scalar)

### Price Multiple Options

```rust
use blackscholes::batch::price_batch;

let inputs: Vec<Inputs> = vec![/* ... */];
let prices = price_batch(&inputs);
```

### Calculate Greeks for Multiple Options

```rust
use blackscholes::batch::all_greeks_batch;

let greeks = all_greeks_batch(&inputs);
```

### Parallel Processing (requires `parallel` feature)

```rust
use blackscholes::batch::{price_batch_par, all_greeks_batch_par};

let prices = price_batch_par(&inputs);  // Uses rayon
let greeks = all_greeks_batch_par(&inputs);
```

## SIMD Operations (requires `simd` feature)

### Price Options with SIMD

```rust
use blackscholes::simd_batch::price_batch_simd;

let inputs: Vec<Inputs> = vec![/* ... at least 100 for best performance */];
let prices = price_batch_simd(&inputs);  // ~4x faster
```

### Calculate All Greeks with SIMD

```rust
use blackscholes::simd_batch::greeks_batch_simd;

let greeks = greeks_batch_simd(&inputs);
// Returns Vec<AllGreeksGeneric<f64>> with all 17 Greeks
```

### Calculate IV with SIMD (Let's Be Rational)

```rust
use blackscholes::simd_batch::iv_batch_simd;

let inputs_with_prices: Vec<Inputs> = vec![/* ... */];
let ivs = iv_batch_simd(&inputs_with_prices);
// Returns Vec<Result<f64, BlackScholesError>>
```

### SIMD + Parallel (requires both features)

```rust
use blackscholes::simd_batch::{
    price_batch_simd_par,
    greeks_batch_simd_par,
    iv_batch_simd_par,
};

let prices = price_batch_simd_par(&inputs);  // ~32x+ faster on 8 cores
let greeks = greeks_batch_simd_par(&inputs);
let ivs = iv_batch_simd_par(&inputs_with_prices);
```

## All 17 Greeks

When you call `calc_all_greeks_generic()` or `greeks_batch_simd()`, you get:

### First-Order Greeks (Price Sensitivity)
- **delta**: ∂V/∂S - Change in option price per $1 change in underlying
- **vega**: ∂V/∂σ - Change in option price per 1% change in volatility
- **theta**: ∂V/∂t - Change in option price per day
- **rho**: ∂V/∂r - Change in option price per 1% change in interest rate
- **epsilon**: ∂V/∂q - Change in option price per 1% change in dividend yield

### Second-Order Greeks (Curvature)
- **gamma**: ∂²V/∂S² - Rate of change of delta
- **vanna**: ∂²V/∂S∂σ - Change in delta per 1% change in volatility
- **charm**: ∂²V/∂S∂t - Change in delta over time
- **vomma**: ∂²V/∂σ² - Change in vega per 1% change in volatility
- **veta**: ∂²V/∂σ∂t - Change in vega over time

### Third-Order Greeks (Stability)
- **speed**: ∂³V/∂S³ - Rate of change of gamma
- **zomma**: ∂³V/∂S²∂σ - Change in gamma per 1% change in volatility
- **color**: ∂³V/∂S²∂t - Change in gamma over time
- **ultima**: ∂³V/∂σ³ - Change in vomma per 1% change in volatility

### Special Greeks
- **lambda**: Ω - Leverage/elasticity (percentage change in option price vs underlying)
- **dual_delta**: ∂V/∂K - Sensitivity to strike price
- **dual_gamma**: ∂²V/∂K² - Second-order sensitivity to strike price

## Error Handling

```rust
use blackscholes::BlackScholesError;

match option.calc_price() {
    Ok(price) => println!("Price: {}", price),
    Err(BlackScholesError::MissingSigma) => {
        eprintln!("Volatility is required");
    }
    Err(BlackScholesError::TimeToMaturityZero) => {
        eprintln!("Time to maturity must be > 0");
    }
    Err(e) => eprintln!("Error: {:?}", e),
}
```

Common errors:
- `MissingSigma` - Volatility required for pricing/Greeks
- `MissingPrice` - Market price required for IV calculation
- `TimeToMaturityZero` - t must be > 0
- `ConvergenceFailed` - IV solver did not converge
- `InvalidLogSK` - Invalid spot/strike combination

## Performance Tips

### When to Use What

| Use Case | Best Choice | Expected Speedup |
|----------|-------------|------------------|
| Single option | Regular API | - |
| 10-100 options | `*_batch` | Minimal |
| 100-1000 options | `*_batch_simd` | ~4x (f64), ~8x (f32) |
| 1000-10000 options | `*_batch_simd_par` | ~32x+ (8 cores) |
| 10000+ options | `*_batch_simd_par` + f32 | ~64x+ |

### Feature Combinations

```toml
# Maximum speed, memory not a concern
features = ["simd", "parallel", "precision-f64"]

# Maximum throughput, slight precision loss OK
features = ["simd", "parallel", "precision-f32"]

# Memory constrained, still want speed
features = ["simd", "precision-f32"]

# Simple batch processing
features = ["parallel"]
```

### Optimization Checklist

✅ Use SIMD for batches of 100+ options  
✅ Use parallel for 1000+ options  
✅ Consider f32 for very large batches (10k+)  
✅ Pre-allocate vectors for batch operations  
✅ Use `calc_all_greeks_generic()` instead of individual Greek calls  
✅ Use `calc_rational_iv()` instead of `calc_iv()` when possible  
✅ Build with `--release` flag  
✅ Consider `RUSTFLAGS="-C target-cpu=native"` for maximum SIMD performance

## Precision Selection

```toml
# Default: f64 (15-16 decimal digits)
blackscholes = "0.24"

# Explicit f64
blackscholes = { version = "0.24", default-features = false, features = ["precision-f64"] }

# f32 (6-7 decimal digits, 2x memory efficiency)
blackscholes = { version = "0.24", default-features = false, features = ["precision-f32"] }
```

**When to use f32:**
- Processing millions of options
- Memory is constrained
- 6-7 digit precision is sufficient
- Maximum throughput needed

**When to use f64:**
- Default choice
- Critical risk calculations
- Extreme strikes or long maturities
- Maximum precision needed

## See Also

- **[Full SIMD Guide](SIMD.md)** - Detailed SIMD documentation
- **[API Documentation](https://docs.rs/blackscholes)** - Complete API reference
- **[Benchmarking Guide](BENCHMARKING.md)** - Performance testing
- **[Examples](../examples/)** - Working code examples
