//! Detailed debug of SIMD calculations

#[cfg(feature = "simd")]
fn main() {
    use blackscholes::{Inputs, OptionType, Pricing};
    use blackscholes::simd_ops::{simd_calc_d1d2_f64x4, simd_calc_price_f64x4, simd_normal_cdf_f64x4, SimdInputsF64x4};
    use wide::f64x4;
    
    // Single test option
    let input = Inputs::new(
        OptionType::Call,
        100.0,  // spot
        100.0,  // strike  
        None,
        0.05,   // r
        0.02,   // q
        0.25,   // t
        Some(0.2),  // sigma
    );
    
    println!("Testing Option:");
    println!("  S={}, K={}, r={}, q={}, t={}, sigma={}", 
             input.s, input.k, input.r, input.q, input.t, input.sigma.unwrap());
    
    // Calculate using non-SIMD
    let price = input.calc_price().unwrap();
    println!("\nNon-SIMD price: {:.6}", price);
    
    // Calculate using SIMD (fill 4 identical options)
    let simd_inputs = SimdInputsF64x4 {
        option_type: [OptionType::Call; 4],
        s: f64x4::splat(100.0),
        k: f64x4::splat(100.0),
        r: f64x4::splat(0.05),
        q: f64x4::splat(0.02),
        t: f64x4::splat(0.25),
        sigma: f64x4::splat(0.2),
    };
    
    // Check d1, d2
    let (d1, d2) = simd_calc_d1d2_f64x4(&simd_inputs);
    let d1_arr: [f64; 4] = d1.into();
    let d2_arr: [f64; 4] = d2.into();
    println!("\nSIMD d1: {:.6}", d1_arr[0]);
    println!("SIMD d2: {:.6}", d2_arr[0]);
    
    // Check normal CDF values
    let nd1 = simd_normal_cdf_f64x4(d1);
    let nd2 = simd_normal_cdf_f64x4(d2);
    let nd1_arr: [f64; 4] = nd1.into();
    let nd2_arr: [f64; 4] = nd2.into();
    println!("\nSIMD N(d1): {:.6}", nd1_arr[0]);
    println!("SIMD N(d2): {:.6}", nd2_arr[0]);
    
    // Calculate final price
    let prices = simd_calc_price_f64x4(&simd_inputs);
    println!("\nSIMD price: {:.6}", prices[0]);
    
    println!("\nDifference: {:.6}", (price - prices[0]).abs());
}

#[cfg(not(feature = "simd"))]
fn main() {
    println!("This example requires the 'simd' feature.");
}
