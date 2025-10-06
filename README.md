![Crates.io](https://img.shields.io/crates/v/blackscholes) ![Docs.rs](https://docs.rs/blackscholes/badge.svg) ![License](https://img.shields.io/crates/l/blackscholes)

# BlackScholes

A lightweight Black‑Scholes‑Merton pricing + full Greeks (1st–3rd order) + implied volatility (Newton & Jäckel rational) with selectable numeric precision (f64 or f32) and optional batch/parallel APIs.

## Features at a Glance

- European vanilla call/put pricing (analytic)
- Full Greeks: delta, gamma, theta, vega, rho, epsilon, lambda, vanna, charm, veta, vomma, speed, zomma, color, ultima, dual-delta, dual-gamma
- Two IV solvers:

    - `calc_iv` (Modified Corrado-Miller initial guess + Newton)
    - `calc_rational_iv` (Jäckel “Let’s be rational”, promotes internally to f64 for accuracy)
- Optional batch helpers (with optional Rayon parallelism via `parallel` feature)

## Selecting Precision

Exactly one precision feature must be active; `precision-f64` is the default.

```toml
[dependencies]
# Default (f64)
blackscholes = "*"

# Explicit selection
# blackscholes = { version = "*", default-features = false, features = ["precision-f64"] }
# blackscholes = { version = "*", default-features = false, features = ["precision-f32"] }
```

Aliases exposed:

| Alias | Meaning |
|-------|---------|
| `Inputs` | Feature-selected generic (`InputsGeneric<f64>` or `<f32>`) |

`calc_rational_price` / `calc_rational_iv` always return `f64` (f32 inputs are promoted internally).

## Minimal Example

```rust
use blackscholes::{Inputs, OptionType, Pricing, GreeksGeneric, ImpliedVolatility};

let inputs = Inputs::new(

    OptionType::Call,
    100.0,
    100.0,

    None,   // price not needed when pricing
    0.05,   // risk-free rate
    0.01,   // dividend yield

    0.5,    // time to maturity (years)
    Some(0.2),
);

let price = inputs.calc_price().unwrap();
let delta = inputs.calc_delta().unwrap();
let all = inputs.calc_all_greeks_generic().unwrap();

// Implied volatility from observed price
let mut iv_inputs = inputs.clone();
iv_inputs.p = Some(price * 1.02); // pretend observed market price

iv_inputs.sigma = None;
let iv = iv_inputs.calc_iv(0.0005).unwrap();
```

## Batch APIs

```rust
use blackscholes::{batch::all_greeks_batch, Inputs, OptionType, Pricing, GreeksGeneric};

let v: Vec<Inputs> = (0..4).map(|i| {
    Inputs::new(OptionType::Call, 100.0 + i as f64, 100.0, None, 0.05, 0.0, 0.25, Some(0.2))
}).collect();

let greeks = all_greeks_batch(&v); // Vec<Result<AllGreeksGeneric<_>, _>>
```

Enable parallel processing:

```toml
[dependencies]
blackscholes = { version = "*", features = ["parallel"] }
```

Then call `batch::all_greeks_batch_par` or `batch::price_batch_par`.

## Implied Volatility Paths

| Method | Notes |
|--------|-------|
| `calc_iv(tol)` | Generic Newton; tolerance is in price difference space (scaled by vega) |
| `calc_rational_iv()` | Jäckel method; fast convergence (2–3 iterations) always returns `f64` |

## All Greeks Struct

`calc_all_greeks_generic()` returns `AllGreeksGeneric<T>` with fields accessible directly (no allocations / maps):

```rust
let g = inputs.calc_all_greeks_generic().unwrap();
println!("delta={} gamma={} vega={}", g.delta, g.gamma, g.vega);
```

## Error Handling

Typed errors via `BlackScholesError`:

```rust
use blackscholes::{Inputs, OptionType, Pricing, BlackScholesError};

let bad = Inputs::new(OptionType::Call, 100.0, 0.0, None, 0.05, 0.0, 30.0/365.25, Some(0.2));

match bad.calc_price() {
    Ok(p) => println!("{}", p),
    Err(BlackScholesError::InvalidLogSK) => eprintln!("invalid S/K log"),
    Err(e) => eprintln!("error: {}", e),
}

```

Common variants: `MissingSigma`, `MissingPrice`, `TimeToMaturityZero`, `InvalidLogSK`, `ConvergenceFailed`.

## Performance Notes

Micro-benchmarks (indicative, not guarantees) show tens of nanoseconds for single pricing / first-order Greeks on modern x86_64 when using `f64`. For repeatable numbers run:

```
cargo bench
```

and consult project benchmark dashboards (if published).

## Feature Summary

| Feature | Effect |
|---------|--------|
| `precision-f64` (default) | Use `f64` throughout |
| `precision-f32` | Use `f32` core (rational IV/pricing still returns `f64`) |
| `parallel` | Enables Rayon-based parallel batch helpers |

## Compatibility & Migration

Earlier versions exposed a concrete `Inputs` struct (f64). The crate now provides a generic `InputsGeneric<T>` internally with `Inputs` as a stable, feature-selected alias. No code changes are required for typical usage; just ensure only one precision feature is enabled.

## License

MIT – see [LICENSE](LICENSE.md).

## Related Packages

| Ecosystem | Link |
|-----------|------|
| Python | https://pypi.org/project/blackscholes-python/ |
| WASM / npm | https://www.npmjs.com/package/@haydenr4/blackscholes_wasm |

---
Feel free to open issues or PRs for additional Greeks, performance tweaks, or API improvements.
# BlackScholes
A library providing Black-Scholes option pricing, Greek calculations, and implied-volatility solver.

[![Crates.io](https://img.shields.io/crates/v/blackscholes.svg)](https://crates.io/crates/blackscholes)
[![Documentation](https://docs.rs/blackscholes/badge.svg)](https://docs.rs/blackscholes)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-GitHub%20Pages-blue)](https://przemyslawolszewski.github.io/bs-rs/)

A Black-Scholes option pricing, Greek calculation, and implied volatility calculation library.

The library handles both European and American style options for the following option types:
- Vanilla Put/Call
- Binary Put/Call
- Binary OT Range (In/Out)
- Barrier


# Performance Optimization

This library is optimized for both single-option pricing and high-throughput batch processing. We've implemented a comprehensive benchmarking infrastructure to measure and improve performance.

## Key Performance Characteristics

- **Single Option Pricing**: ~35-40 ns per option
- **Rational Pricing Method**: ~55-65 ns per option
- **Delta Calculation**: ~30-35 ns per option
- **Gamma Calculation**: ~14-15 ns per option
- **Batch Processing**: Scales linearly up to large batch sizes
- **All Greeks Calculation**: ~2 ms per 1000 options

## Benchmarking Infrastructure

The library includes a comprehensive benchmarking system for performance tracking:

- **Interactive Charts**: Professional benchmark visualizations on [GitHub Pages](https://przemyslawolszewski.github.io/bs-rs/)
- **Automated Regression Detection**: CI-integrated tests that fail on performance regressions (>10% threshold)
- **Historical Tracking**: Continuous monitoring of performance trends over time
- **Pull Request Comments**: Automatic performance comparison comments on PRs

View live benchmark results at: https://przemyslawolszewski.github.io/bs-rs/

## Precision Selection (f64 vs f32)

As of the upcoming release the core engine is generic over the float type. You can choose precision via Cargo features:

```toml
[dependencies]
blackscholes = { version = "*", default-features = false, features = ["precision-f64"] }
# or for 32-bit floats
blackscholes = { version = "*", default-features = false, features = ["precision-f32"] }
```

Exactly one of `precision-f64` (default) or `precision-f32` must be enabled. They are mutually exclusive.

Public helpers:
- `Inputs` (legacy, concrete f64) – deprecated and will be removed in a future major version.
- `InputsSelected` – feature‑selected alias of the generic `InputsGeneric<T>`.
- `InputsF64` / `InputsF32` – explicit aliases if you need to write code that depends on precision at compile time.

`calc_rational_iv()` always computes using 64-bit precision under the hood; the f32 path promotes to f64 internally and casts back. The standard Newton IV solver (`calc_iv`) runs natively in the selected precision.

## Usage Examples

```rust
use blackscholes::{InputsSelected as Inputs, OptionType, Pricing, Greeks, ImpliedVolatility};

// Basic option pricing
let inputs = Inputs::new(
    OptionType::Call,   // Call option
    100.0,              // Spot price
    100.0,              // Strike price
    None,               // Option price (not needed for pricing)
    0.05,               // Risk-free rate
    0.01,               // Dividend yield
    1.0,                // Time to maturity (in years)
    Some(0.2),          // Volatility
);

// Calculate option price
let price = inputs.calc_price().unwrap();
println!("Option price: {}", price);

// Calculate option Greeks
let delta = inputs.calc_delta().unwrap();
let gamma = inputs.calc_gamma().unwrap();
let theta = inputs.calc_theta().unwrap();
let vega = inputs.calc_vega().unwrap();
let rho = inputs.calc_rho().unwrap();

println!("Delta: {}, Gamma: {}, Vega: {}", delta, gamma, vega);

// Calculate implied volatility from price
let mut iv_inputs = Inputs::new(
    OptionType::Call,
    100.0,
    100.0,
    Some(10.0),  // Option price
    0.05,
    0.01,
    1.0,
    None,        // Volatility is what we're solving for
);

let iv = iv_inputs.calc_iv(0.0001).unwrap();
println!("Implied volatility: {}", iv);
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# blackscholes
[![Crates.io](https://img.shields.io/crates/v/blackscholes)](https://crates.io/crates/blackscholes)
[![Docs.rs](https://docs.rs/blackscholes/badge.svg)](https://docs.rs/blackscholes)
![License](https://img.shields.io/crates/l/blackscholes)
  
This library provides a simple, lightweight, and efficient (though not heavily optimized) implementation of the Black-Scholes-Merton model for pricing European options.  
  
Includes all first, second, and third order Greeks.  

Implements both:  

- calc_iv() in the ImpliedVolatility trait which uses [Modified Corrado-Miller by Piotr Płuciennik (2007)](https://sin.put.poznan.pl/files/download/37938) for the initial volatility guess and the Newton Raphson algorithm to solve for the implied volatility.
- calc_rational_iv() in the ImpliedVolatility trait which uses "Let's be rational" method from ["Let's be rational" (2016) by Peter Jackel](http://www.jaeckel.org/LetsBeRational.pdf).  Utilizing Jackel's C++ implementation to get convergence within 2 iterations with 64-bit floating point accuracy.
  
## Usage  
  
View the [docs](https://docs.rs/blackscholes) for usage and examples.  
  
**Other packages available:**  
Python: [Pypi](https://pypi.org/project/blackscholes-python/)  
WASM: [npm](https://www.npmjs.com/package/@haydenr4/blackscholes_wasm)  

## API note

`calc_all_greeks()` returns a typed struct `AllGreeks` (not a HashMap) for zero-allocation and faster access to fields like `delta`, `gamma`, etc.

### Deprecation Notice

`Inputs` (the concrete f64 struct) is deprecated. Migrate to `InputsSelected` (preferred) or `InputsF64` / `InputsF32`. The concrete struct will be removed after a transition period to reduce maintenance duplication.

## Build performance

This repo enables LTO, codegen-units=1, opt-level=3, panic=abort and `target-cpu=native` (via `.cargo/config.toml`) for maximum throughput. Use `cargo bench` to run Criterion benchmarks.

## Errors

The library uses a typed error enum `BlackScholesError` instead of string errors. This makes it easy to handle specific failure modes:

```rust
use blackscholes::{Inputs, OptionType, Pricing, BlackScholesError};

let inputs = Inputs::new(OptionType::Call, 100.0, 0.0, None, 0.05, 0.0, 30.0/365.25, Some(0.2));
match inputs.calc_price() {
    Ok(p) => println!("{}", p),
    Err(BlackScholesError::MissingSigma) => eprintln!("volatility is required"),
    Err(e) => eprintln!("error: {}", e),
}
```

Common variants: `MissingSigma`, `MissingPrice`, `TimeToMaturityZero`, `InvalidLogSK`, `ConvergenceFailed`.
