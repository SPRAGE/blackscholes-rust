//! Test to verify SIMD Greeks match non-SIMD calculations

#[cfg(feature = "simd")]
fn main() {
    use blackscholes::{Inputs, OptionType, GreeksGeneric};
    use blackscholes::simd_batch::greeks_batch_simd;
    
    println!("Testing SIMD Greeks vs Non-SIMD Greeks\n");
    println!("==========================================\n");
    
    // Test various scenarios
    let test_cases = vec![
        // (spot, strike, r, q, t, sigma, option_type, description)
        (100.0, 100.0, 0.05, 0.02, 0.25, 0.2, OptionType::Call, "ATM Call"),
        (100.0, 105.0, 0.05, 0.02, 0.25, 0.2, OptionType::Call, "OTM Call"),
        (100.0, 95.0, 0.05, 0.02, 0.25, 0.2, OptionType::Call, "ITM Call"),
        (100.0, 100.0, 0.05, 0.02, 0.25, 0.2, OptionType::Put, "ATM Put"),
        (100.0, 105.0, 0.05, 0.02, 0.25, 0.2, OptionType::Put, "ITM Put"),
        (100.0, 95.0, 0.05, 0.02, 0.25, 0.2, OptionType::Put, "OTM Put"),
        (100.0, 100.0, 0.05, 0.02, 1.0, 0.3, OptionType::Call, "Long dated high vol"),
        (100.0, 100.0, 0.10, 0.05, 0.5, 0.15, OptionType::Call, "Different rates"),
    ];
    
    let mut max_errors = std::collections::HashMap::new();
    max_errors.insert("delta", 0.0);
    max_errors.insert("gamma", 0.0);
    max_errors.insert("theta", 0.0);
    max_errors.insert("vega", 0.0);
    max_errors.insert("rho", 0.0);
    max_errors.insert("epsilon", 0.0);
    max_errors.insert("lambda", 0.0);
    max_errors.insert("vanna", 0.0);
    max_errors.insert("charm", 0.0);
    max_errors.insert("veta", 0.0);
    max_errors.insert("vomma", 0.0);
    max_errors.insert("speed", 0.0);
    max_errors.insert("zomma", 0.0);
    max_errors.insert("color", 0.0);
    max_errors.insert("ultima", 0.0);
    max_errors.insert("dual_delta", 0.0);
    max_errors.insert("dual_gamma", 0.0);
    
    for (s, k, r, q, t, sigma, opt_type, desc) in test_cases {
        let input = Inputs::new(opt_type, s, k, None, r, q, t, Some(sigma));
        
        // Calculate using non-SIMD
        let non_simd = input.calc_all_greeks_generic().unwrap();
        
        // Calculate using SIMD
        let simd_results = greeks_batch_simd(&[input.clone()]);
        let simd = &simd_results[0];
        
        println!("Test: {}", desc);
        println!("  S={}, K={}, r={}, q={}, t={}, σ={}, type={:?}", s, k, r, q, t, sigma, opt_type);
        
        // Compare each Greek
        let fields = [
            ("delta", non_simd.delta, simd.delta),
            ("gamma", non_simd.gamma, simd.gamma),
            ("theta", non_simd.theta, simd.theta),
            ("vega", non_simd.vega, simd.vega),
            ("rho", non_simd.rho, simd.rho),
            ("epsilon", non_simd.epsilon, simd.epsilon),
            ("lambda", non_simd.lambda, simd.lambda),
            ("vanna", non_simd.vanna, simd.vanna),
            ("charm", non_simd.charm, simd.charm),
            ("veta", non_simd.veta, simd.veta),
            ("vomma", non_simd.vomma, simd.vomma),
            ("speed", non_simd.speed, simd.speed),
            ("zomma", non_simd.zomma, simd.zomma),
            ("color", non_simd.color, simd.color),
            ("ultima", non_simd.ultima, simd.ultima),
            ("dual_delta", non_simd.dual_delta, simd.dual_delta),
            ("dual_gamma", non_simd.dual_gamma, simd.dual_gamma),
        ];
        
        let mut has_error = false;
        for (name, non_simd_val, simd_val) in fields.iter() {
            let diff = (non_simd_val - simd_val).abs();
            let rel_error = if non_simd_val.abs() > 1e-10 {
                diff / non_simd_val.abs()
            } else {
                diff
            };
            
            // Update max error
            if let Some(max_err) = max_errors.get_mut(name) {
                if rel_error > *max_err {
                    *max_err = rel_error;
                }
            }
            
            // Flag errors > 0.01% (0.0001)
            if rel_error > 0.0001 {
                println!("  ⚠️  {}: non_simd={:.8}, simd={:.8}, diff={:.2e}, rel_err={:.2e}",
                         name, non_simd_val, simd_val, diff, rel_error);
                has_error = true;
            }
        }
        
        if !has_error {
            println!("  ✅ All Greeks match within tolerance");
        }
        println!();
    }
    
    println!("\n==========================================");
    println!("Maximum Relative Errors:");
    println!("==========================================");
    for (name, max_err) in max_errors.iter() {
        println!("  {:<12}: {:.2e}", name, max_err);
    }
    
    // Check if all errors are within acceptable range
    let all_pass = max_errors.values().all(|&e| e < 0.001); // 0.1% tolerance
    if all_pass {
        println!("\n✅ ALL TESTS PASSED - SIMD Greeks match non-SIMD within 0.1%");
    } else {
        println!("\n⚠️  Some Greeks have errors > 0.1%");
    }
}

#[cfg(not(feature = "simd"))]
fn main() {
    println!("This test requires the 'simd' feature.");
}
