//! Debug SIMD calculations

#[cfg(feature = "simd")]
fn main() {
    use blackscholes::{Inputs, OptionType, Pricing};
    use blackscholes::simd_batch::price_batch_simd;
    
    // Create options
    let inputs: Vec<Inputs> = vec![
        Inputs::new(OptionType::Call, 100.0, 100.0, None, 0.05, 0.02, 0.25, Some(0.2)),
        Inputs::new(OptionType::Call, 100.0, 105.0, None, 0.05, 0.02, 0.25, Some(0.2)),
        Inputs::new(OptionType::Call, 100.0, 110.0, None, 0.05, 0.02, 0.25, Some(0.2)),
        Inputs::new(OptionType::Call, 100.0, 115.0, None, 0.05, 0.02, 0.25, Some(0.2)),
    ];
    
    // Price using non-SIMD
    println!("Non-SIMD prices:");
    for (i, input) in inputs.iter().enumerate() {
        let price = input.calc_price().unwrap();
        println!("  Option {}: Strike={:.2}, Price={:.6}", i, input.k, price);
    }
    
    // Price using SIMD
    println!("\nSIMD prices:");
    let simd_prices = price_batch_simd(&inputs);
    for (i, price) in simd_prices.iter().enumerate() {
        println!("  Option {}: Strike={:.2}, Price={:.6}", i, inputs[i].k, price);
    }
}

#[cfg(not(feature = "simd"))]
fn main() {
    println!("This example requires the 'simd' feature.");
}
