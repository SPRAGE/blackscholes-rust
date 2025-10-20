# Documentation Index

This directory contains comprehensive documentation for the `blackscholes` Rust crate.

## Core Documentation

### [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⭐ START HERE
Fast lookup for common tasks:
- Installation snippets for all feature combinations
- Code examples for pricing, Greeks, and IV
- All 17 Greeks with descriptions
- Error handling patterns
- Performance tips and feature selection guide

**When to read**: First time using the crate, or need a quick reminder of syntax.

### [SIMD.md](SIMD.md)
Complete guide to SIMD (Single Instruction Multiple Data) acceleration features:
- Quick start guide
- Performance benchmarks and comparisons
- Complete API reference (high-level and low-level)
- Technical details (algorithms, memory layouts)
- Best practices and optimization tips
- Troubleshooting and platform-specific notes

**When to read**: If you're processing large batches of options (100+) and want maximum performance.

### [BENCHMARKING.md](BENCHMARKING.md)
Guide to running and interpreting performance benchmarks:
- How to run criterion benchmarks
- Understanding benchmark results
- Comparing different configurations (SIMD vs scalar, f64 vs f32)
- Performance tuning tips

**When to read**: When evaluating performance or choosing between features.

## API Documentation

The complete API reference is hosted on [docs.rs](https://docs.rs/blackscholes):
- All function signatures and parameters
- Return types and error handling
- Usage examples for each API
- Type aliases and feature flags

## Quick Start by Use Case

### I want to price a single option
→ See [README.md](../README.md) "Minimal Example" section

### I want to calculate Greeks for a single option
→ See [API docs - GreeksGeneric trait](https://docs.rs/blackscholes/latest/blackscholes/trait.GreeksGeneric.html)

### I want to calculate implied volatility
→ See [API docs - ImpliedVolatility trait](https://docs.rs/blackscholes/latest/blackscholes/trait.ImpliedVolatility.html)

### I want to process batches of 100+ options fast
→ Read [SIMD.md](SIMD.md) Quick Start section

### I want to process batches using all CPU cores
→ Enable `parallel` feature and use `*_batch_par` functions in [batch module](https://docs.rs/blackscholes/latest/blackscholes/batch/index.html)

### I want both SIMD and parallel processing
→ Enable both `simd` and `parallel` features, use `*_batch_simd_par` functions in [simd_batch module](https://docs.rs/blackscholes/latest/blackscholes/simd_batch/index.html)

### I want to optimize memory usage
→ Use `precision-f32` feature instead of default `precision-f64` (see [README.md](../README.md) "Selecting Precision")

### I want maximum performance
→ Read [SIMD.md](SIMD.md) "Performance" and "Best Practices" sections

## Examples

The [`examples/`](../examples/) directory contains working code examples:

### SIMD Examples
- `simd_example.rs` - Basic SIMD usage with all features
- `debug_simd.rs` - SIMD validation and debugging
- `test_all_greeks_simd.rs` - Comprehensive Greeks validation

### General Examples
Check the examples directory for additional usage patterns.

## Feature Flags

| Feature | Description | Documentation |
|---------|-------------|---------------|
| `precision-f64` | Use f64 precision (default) | [README.md](../README.md) |
| `precision-f32` | Use f32 precision | [README.md](../README.md) |
| `simd` | Enable SIMD acceleration | [SIMD.md](SIMD.md) |
| `parallel` | Enable rayon parallelism | [README.md](../README.md), [API docs](https://docs.rs/blackscholes) |

## Performance Summary

For those who just want the numbers:

| Configuration | Throughput (options/sec) | Speedup vs Scalar |
|---------------|-------------------------|-------------------|
| Scalar (f64) | ~12.5M | 1x (baseline) |
| SIMD f64x4 | ~50M | ~4x |
| SIMD f32x8 | ~100M | ~8x |
| SIMD + Parallel (8 cores) | ~400M+ | ~32x+ |

*Benchmarks for calculating all 17 Greeks. See [SIMD.md](SIMD.md) for detailed performance analysis.*

## Getting Help

1. **Check the docs**: Start with [README.md](../README.md) and [docs.rs](https://docs.rs/blackscholes)
2. **SIMD questions**: See [SIMD.md](SIMD.md) troubleshooting section
3. **Performance issues**: Read [BENCHMARKING.md](BENCHMARKING.md) and [SIMD.md](SIMD.md) best practices
4. **API questions**: Check [docs.rs API reference](https://docs.rs/blackscholes)
5. **Found a bug?**: Open an issue on [GitHub](https://github.com/hayden4r4/blackscholes-rust)

## Contributing

See [BENCHMARKING.md](BENCHMARKING.md) for performance testing guidelines if you're contributing optimizations.
