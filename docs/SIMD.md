# SIMD Acceleration Guide

This document provides comprehensive documentation for the SIMD (Single Instruction, Multiple Data) acceleration features in the `blackscholes` crate.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Technical Details](#technical-details)
- [Best Practices](#best-practices)
- [Benchmarking](#benchmarking)
- [Troubleshooting](#troubleshooting)

## Overview

The SIMD module provides vectorized implementations of Black-Scholes calculations, allowing you to process multiple options simultaneously using modern CPU vector instructions.

### Key Features

- **Vectorized Processing**: Process 4 (f64) or 8 (f32) options at once
- **All 17 Greeks**: Complete Greeks calculation in SIMD
- **Fast IV**: Jäckel's "Let's Be Rational" method (2-3 iterations)
- **Portable**: Works on x86 (AVX/AVX2), ARM (NEON), and other architectures
- **Parallel Ready**: Combines with rayon for multi-threaded processing
- **Zero-Copy**: No heap allocations during computation

### Supported Operations

| Operation | Description | SIMD Types |
|-----------|-------------|------------|
| **Pricing** | European call/put pricing | `simd_calc_price_f64x4`, `simd_calc_price_f32x8` |
| **Greeks** | All 17 Greeks calculation | `simd_calc_greeks_f64x4`, `simd_calc_greeks_f32x8` |
| **Implied Vol** | IV using Jäckel's method | `simd_calc_rational_iv_f64x4`, `simd_calc_rational_iv_f32x8` |

## Quick Start

### Enable the Feature

Add to your `Cargo.toml`:

```toml
[dependencies]
blackscholes = { version = "0.24", features = ["simd"] }
```

### Basic Usage

```rust
use blackscholes::{Inputs, OptionType};
use blackscholes::simd_batch::{price_batch_simd, greeks_batch_simd, iv_batch_simd};

fn main() {
    // Create a batch of options
    let inputs: Vec<Inputs> = (0..16)
        .map(|i| {
            Inputs::new(
                OptionType::Call,
                100.0,                    // spot price
                95.0 + i as f64 * 2.0,   // varying strikes
                None,
                0.05,                     // risk-free rate
                0.02,                     // dividend yield
                0.25,                     // time to maturity
                Some(0.2),                // volatility
            )
        })
        .collect();

    // Price using SIMD
    let prices = price_batch_simd(&inputs);
    println!("Prices: {:?}", prices);

    // Calculate Greeks using SIMD
    let greeks = greeks_batch_simd(&inputs);
    println!("First option delta: {}", greeks[0].delta);
    println!("First option gamma: {}", greeks[0].gamma);
    
    // Calculate implied volatility
    let iv_inputs: Vec<Inputs> = prices.iter().enumerate()
        .map(|(i, &price)| {
            let mut inp = inputs[i].clone();
            inp.option_price = Some(price);
            inp.sigma = None;
            inp
        })
        .collect();
    
    let ivs = iv_batch_simd(&iv_inputs);
    for (i, iv_result) in ivs.iter().enumerate() {
        match iv_result {
            Ok(iv) => println!("Option {}: IV = {:.6}", i, iv),
            Err(e) => println!("Option {}: Error - {:?}", i, e),
        }
    }
}
```

### With Parallel Processing

For large batches, combine SIMD with parallel processing:

```rust
use blackscholes::simd_batch::price_batch_simd_par;

// Process 100,000 options
let large_batch: Vec<Inputs> = create_large_batch(100_000);
let prices = price_batch_simd_par(&large_batch);
```

## Performance

### Speedup Comparison

Compared to scalar (non-SIMD) processing:

| Operation | Batch Size | Scalar Time | SIMD Time | Speedup |
|-----------|------------|-------------|-----------|---------|
| Pricing | 10,000 | ~800 µs | ~200 µs | **4.0x** |
| Greeks (6) | 10,000 | ~1,500 µs | ~400 µs | **3.75x** |
| Greeks (17) | 10,000 | ~2,000 µs | ~500 µs | **4.0x** |
| Implied Vol | 10,000 | ~5,000 µs | ~1,200 µs | **4.2x** |

*Benchmarked on Intel Core i7 (AVX2), f64 precision*

### Per-Option Performance

| Operation | Time per Option | Throughput |
|-----------|----------------|------------|
| Pricing | ~0.02 µs | ~50M options/sec |
| All Greeks | ~0.05 µs | ~20M options/sec |
| Implied Vol | ~0.12 µs | ~8M options/sec |

### Memory Bandwidth

- **f64x4**: ~256 bytes per 4 options (64 bytes/option)
- **f32x8**: ~128 bytes per 8 options (16 bytes/option)
- Cache-friendly: processes in 4 or 8-option chunks

## API Reference

### High-Level Batch APIs

Located in `blackscholes::simd_batch`:

#### `price_batch_simd(inputs: &[Inputs]) -> Vec<f64>`

Price a batch of options using SIMD.

```rust
let inputs: Vec<Inputs> = create_options();
let prices = price_batch_simd(&inputs);
```

**Features:**
- Automatic chunking (4 for f64, 8 for f32)
- Scalar fallback for remainder
- No heap allocations during SIMD processing

#### `greeks_batch_simd(inputs: &[Inputs]) -> Vec<AllGreeksGeneric<f64>>`

Calculate all 17 Greeks using SIMD.

```rust
let greeks = greeks_batch_simd(&inputs);

// Access any Greek
println!("Delta: {}", greeks[0].delta);
println!("Vanna: {}", greeks[0].vanna);
println!("Speed: {}", greeks[0].speed);
```

**Greeks Calculated:**
- **First-order**: Delta, Vega, Theta, Rho, Epsilon, Lambda
- **Second-order**: Gamma, Vanna, Charm, Vomma, Veta
- **Third-order**: Speed, Zomma, Color, Ultima
- **Exotic**: Dual Delta, Dual Gamma

#### `iv_batch_simd(inputs: &[Inputs]) -> Vec<Result<f64, BlackScholesError>>`

Calculate implied volatility using Jäckel's "Let's Be Rational" method.

```rust
let ivs = iv_batch_simd(&inputs);
for (i, result) in ivs.iter().enumerate() {
    match result {
        Ok(iv) => println!("IV {}: {:.6}", i, iv),
        Err(e) => eprintln!("Error {}: {:?}", i, e),
    }
}
```

**Features:**
- 2-3 iterations typical convergence
- Handles edge cases (deep ITM/OTM)
- No tolerance parameter needed

#### Parallel Versions

Add `features = ["simd", "parallel"]` to use:

- `price_batch_simd_par(inputs: &[Inputs]) -> Vec<f64>`
- `greeks_batch_simd_par(inputs: &[Inputs]) -> Vec<AllGreeksGeneric<f64>>`
- `iv_batch_simd_par(inputs: &[Inputs]) -> Vec<Result<f64, BlackScholesError>>`

**Configuration:**
- Splits batch into 64-option chunks
- Processes chunks in parallel using rayon
- Each chunk uses SIMD internally

### Low-Level SIMD APIs

Located in `blackscholes::simd_ops`:

#### Data Structures

```rust
pub struct SimdInputsF64x4 {
    pub option_type: [OptionType; 4],
    pub s: f64x4,           // spot prices
    pub k: f64x4,           // strike prices
    pub r: f64x4,           // risk-free rates
    pub q: f64x4,           // dividend yields
    pub t: f64x4,           // times to maturity
    pub sigma: f64x4,       // volatilities
}

pub struct SimdGreeksF64x4 {
    pub delta: [f64; 4],
    pub gamma: [f64; 4],
    pub theta: [f64; 4],
    pub vega: [f64; 4],
    pub rho: [f64; 4],
    pub epsilon: [f64; 4],
    pub lambda: [f64; 4],
    pub vanna: [f64; 4],
    pub charm: [f64; 4],
    pub veta: [f64; 4],
    pub vomma: [f64; 4],
    pub speed: [f64; 4],
    pub zomma: [f64; 4],
    pub color: [f64; 4],
    pub ultima: [f64; 4],
    pub dual_delta: [f64; 4],
    pub dual_gamma: [f64; 4],
}
```

#### Core Functions

```rust
// Pricing
pub fn simd_calc_price_f64x4(inputs: &SimdInputsF64x4) -> [f64; 4]

// Greeks
pub fn simd_calc_greeks_f64x4(inputs: &SimdInputsF64x4) -> SimdGreeksF64x4

// Implied Volatility
pub fn simd_calc_rational_iv_f64x4(
    inputs: &SimdInputsF64x4,
    prices: [f64; 4],
) -> Result<[f64; 4], BlackScholesError>

// Utility functions
pub fn simd_normal_cdf_f64x4(x: f64x4) -> f64x4
pub fn simd_normal_pdf_f64x4(x: f64x4) -> f64x4
pub fn simd_calc_d1d2_f64x4(inputs: &SimdInputsF64x4) -> (f64x4, f64x4)
```

## Technical Details

### SIMD Vector Types

The crate uses the `wide` crate for portable SIMD:

- **f64x4**: Four 64-bit floats (256 bits total)
  - AVX/AVX2 on x86_64
  - NEON on ARM64 (with emulation)
  
- **f32x8**: Eight 32-bit floats (256 bits total)
  - AVX/AVX2 on x86_64
  - NEON on ARM64 (with emulation)

### Normal Distribution Approximation

Uses **Abramowitz & Stegun** formula 7.1.26 for the error function:

```
CDF(x) = 0.5 * (1 + erf(x/√2))
```

**Accuracy**: Maximum error < 7.5×10⁻⁸

### Implied Volatility Algorithm

**Jäckel's "Let's Be Rational"** method:

1. Initial guess using rational approximation
2. Halley's method refinement
3. Special handling for extreme moneyness
4. Monotonic convergence guarantee

**Advantages**:
- 2-3 iterations (vs 10+ for Newton-Raphson)
- More stable for deep ITM/OTM
- No tolerance parameter needed

**Reference**: Jäckel, P. (2015). "Let's Be Rational". Wilmott Magazine.

### Memory Layout

**Input Conversion** (AoS → SoA):
```
Array of Structs (Input):
[{s:100,k:95,r:0.05,...}, {s:100,k:100,r:0.05,...}, ...]

Struct of Arrays (SIMD):
{
    s: [100, 100, 100, 100],
    k: [95, 100, 105, 110],
    r: [0.05, 0.05, 0.05, 0.05],
    ...
}
```

This layout enables efficient SIMD processing with minimal register spills.

## Best Practices

### When to Use SIMD

✅ **Good Use Cases:**
- Pricing > 100 options
- Calculating Greeks for option chains
- IV surface construction
- Risk management (portfolio Greeks)
- Monte Carlo variance reduction

❌ **Not Recommended:**
- Single option calculations (use scalar)
- Very small batches (< 16 options)
- When precision-f32 isn't sufficient

### Batch Size Recommendations

| Batch Size | Recommendation |
|------------|----------------|
| < 16 | Use scalar functions |
| 16 - 1,000 | SIMD only |
| 1,000 - 10,000 | SIMD preferred |
| > 10,000 | SIMD + Parallel |

### Precision Choice

**f64 (default)**:
- ✅ Full precision (~15-17 decimal digits)
- ✅ Suitable for all use cases
- ❌ Half the throughput vs f32

**f32** (`features = ["precision-f32"]`):
- ✅ 2x throughput (8 at once vs 4)
- ✅ Sufficient for most trading (~7 decimal digits)
- ❌ May accumulate errors in complex calculations

### Optimization Tips

1. **Pre-allocate vectors** for results:
   ```rust
   let mut prices = Vec::with_capacity(inputs.len());
   ```

2. **Reuse input vectors** when possible

3. **Use parallel** for > 10,000 options:
   ```rust
   let prices = price_batch_simd_par(&large_batch);
   ```

4. **Profile first**: Use `cargo bench` to verify SIMD benefits

## Benchmarking

### Running Benchmarks

```bash
# Run all SIMD benchmarks
cargo bench --features simd

# Run specific benchmark
cargo bench --features simd pricing

# Compare SIMD vs scalar
cargo bench --features simd -- pricing
```

### Example Benchmark

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use blackscholes::{Inputs, OptionType};
use blackscholes::simd_batch::price_batch_simd;

fn benchmark_simd_pricing(c: &mut Criterion) {
    let inputs: Vec<Inputs> = (0..10000)
        .map(|i| {
            Inputs::new(
                OptionType::Call,
                100.0,
                95.0 + (i % 20) as f64,
                None,
                0.05,
                0.02,
                0.25,
                Some(0.2),
            )
        })
        .collect();

    c.bench_function("simd_price_10k", |b| {
        b.iter(|| price_batch_simd(black_box(&inputs)))
    });
}

criterion_group!(benches, benchmark_simd_pricing);
criterion_main!(benches);
```

## Troubleshooting

### Common Issues

#### "Feature 'simd' not enabled"

**Solution**: Add feature to Cargo.toml:
```toml
blackscholes = { version = "0.24", features = ["simd"] }
```

#### Performance not as expected

**Possible causes**:
1. Batch size too small (< 100 options)
2. CPU doesn't support AVX2 (check with `lscpu | grep avx2`)
3. Debug build (use `--release`)
4. Memory allocation overhead (pre-allocate)

**Debug**:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features simd
```

#### Accuracy concerns

**Check**:
- Compare SIMD vs scalar results
- Run validation tests:
  ```bash
  cargo run --example test_all_greeks_simd --features simd --release
  ```

#### Compilation errors with `wide` crate

**Solution**: Update dependencies:
```bash
cargo update
cargo clean
cargo build --features simd
```

### Platform-Specific Notes

**x86_64**:
- Best performance with AVX2 (Haswell+)
- Use `-C target-cpu=native` for optimization
- SSE2 fallback available (slower)

**ARM64**:
- Uses NEON with some emulation
- Performance may vary by chip
- Test on target hardware

**WebAssembly**:
- SIMD support via WASM SIMD proposal
- May require specific flags
- Verify browser support

## Further Reading

- [Jäckel's "Let's Be Rational" paper](https://www.wilmott.com/lets-be-rational/)
- [Abramowitz & Stegun Handbook](https://personal.math.ubc.ca/~cbm/aands/)
- [Wide crate documentation](https://docs.rs/wide/)
- [Benchmarking guide](BENCHMARKING.md)

## Examples

See the [`examples/`](../examples) directory:

- `simd_example.rs` - Comprehensive SIMD demonstration
- `test_all_greeks_simd.rs` - Validation of all Greeks
- `debug_simd.rs` - SIMD vs scalar comparison

Run with:
```bash
cargo run --example simd_example --features simd --release
```
