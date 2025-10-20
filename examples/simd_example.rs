//! Example demonstrating SIMD-accelerated Black-Scholes calculations
//! 
//! Compile with SIMD support:
//! ```bash
//! cargo run --example simd_example --features simd
//! ```
//! 
//! Or with both SIMD and parallel processing:
//! ```bash
//! cargo run --example simd_example --features "simd,parallel" --release
//! ```

#[cfg(feature = "simd")]
fn main() {
    use blackscholes::{Inputs, OptionType};
    use blackscholes::simd_batch::{price_batch_simd, greeks_batch_simd, iv_batch_simd};
    
    println!("Black-Scholes SIMD Example\n");
    println!("==========================\n");
    
    // Create a batch of options to price
    let inputs: Vec<Inputs> = (0..16)
        .map(|i| {
            let strike = 100.0 + (i as f64) * 5.0;
            Inputs::new(
                OptionType::Call,
                100.0,        // spot price
                strike,       // strike price
                None,         // option price (not needed for pricing)
                0.05,         // risk-free rate
                0.02,         // dividend yield
                0.25,         // time to maturity (3 months)
                Some(0.2),    // volatility
            )
        })
        .collect();
    
    println!("Pricing {} options using SIMD acceleration...", inputs.len());
    
    // Price using SIMD
    let prices = price_batch_simd(&inputs);
    println!("\nOption Prices:");
    for (i, price) in prices.iter().enumerate() {
        println!("  Option {}: Strike={:.2}, Price={:.4}", 
                 i, inputs[i].k, price);
    }
    
    // Calculate Greeks using SIMD
    println!("\nCalculating Greeks using SIMD...");
    let greeks = greeks_batch_simd(&inputs);
    println!("\nFirst option Greeks:");
    println!("  Delta:      {:.6}", greeks[0].delta);
    println!("  Gamma:      {:.6}", greeks[0].gamma);
    println!("  Theta:      {:.6}", greeks[0].theta);
    println!("  Vega:       {:.6}", greeks[0].vega);
    println!("  Rho:        {:.6}", greeks[0].rho);
    println!("  Epsilon:    {:.6}", greeks[0].epsilon);
    println!("  Lambda:     {:.6}", greeks[0].lambda);
    println!("  Vanna:      {:.6}", greeks[0].vanna);
    println!("  Charm:      {:.6}", greeks[0].charm);
    println!("  Veta:       {:.6}", greeks[0].veta);
    println!("  Vomma:      {:.6}", greeks[0].vomma);
    println!("  Speed:      {:.6}", greeks[0].speed);
    println!("  Zomma:      {:.6}", greeks[0].zomma);
    println!("  Color:      {:.6}", greeks[0].color);
    println!("  Ultima:     {:.6}", greeks[0].ultima);
    println!("  Dual Delta: {:.6}", greeks[0].dual_delta);
    println!("  Dual Gamma: {:.6}", greeks[0].dual_gamma);
    
    // Calculate implied volatility using SIMD
    println!("\nCalculating Implied Volatility using SIMD (Let's Be Rational method)...");
    let iv_inputs: Vec<Inputs> = prices.iter().enumerate().map(|(i, &price)| {
        Inputs::new(
            OptionType::Call,
            100.0,
            inputs[i].k,
            Some(price),  // Use calculated price
            0.05,
            0.02,
            0.25,
            None,         // No sigma - we're calculating it
        )
    }).collect();
    
    let ivs = iv_batch_simd(&iv_inputs);
    println!("\nImplied Volatilities (should be close to 0.2):");
    for (i, iv_result) in ivs.iter().enumerate() {
        match iv_result {
            Ok(iv) => println!("  Option {}: IV={:.6}", i, iv),
            Err(e) => println!("  Option {}: Error - {:?}", i, e),
        }
    }
    
    // Demonstrate performance with larger batch
    println!("\n\nPerformance Test:");
    println!("==================");
    
    let large_batch: Vec<Inputs> = (0..10000)
        .map(|i| {
            Inputs::new(
                if i % 2 == 0 { OptionType::Call } else { OptionType::Put },
                100.0,
                95.0 + (i % 20) as f64,
                None,
                0.05,
                0.02,
                0.25,
                Some(0.15 + (i % 10) as f64 * 0.01),
            )
        })
        .collect();
    
    use std::time::Instant;
    
    let start = Instant::now();
    let batch_prices = price_batch_simd(&large_batch);
    let simd_duration = start.elapsed();
    
    println!("\nPriced {} options using SIMD in {:?}", 
             large_batch.len(), simd_duration);
    println!("Average time per option: {:.2} µs", 
             simd_duration.as_micros() as f64 / large_batch.len() as f64);
    
    // Verify a few results
    println!("\nSample prices from large batch:");
    for i in (0..10).step_by(2) {
        println!("  Option {}: {:.4}", i, batch_prices[i]);
    }
    
    #[cfg(feature = "parallel")]
    {
        use blackscholes::simd_batch::price_batch_simd_par;
        
        println!("\n\nSIMD + Parallel Processing Test:");
        println!("==================================");
        
        let start = Instant::now();
        let par_prices = price_batch_simd_par(&large_batch);
        let par_duration = start.elapsed();
        
        println!("Priced {} options using SIMD+Parallel in {:?}", 
                 large_batch.len(), par_duration);
        println!("Average time per option: {:.2} µs", 
                 par_duration.as_micros() as f64 / large_batch.len() as f64);
        println!("Speedup: {:.2}x", 
                 simd_duration.as_secs_f64() / par_duration.as_secs_f64());
    }
}

#[cfg(not(feature = "simd"))]
fn main() {
    println!("This example requires the 'simd' feature to be enabled.");
    println!("Please run with: cargo run --example simd_example --features simd");
}
