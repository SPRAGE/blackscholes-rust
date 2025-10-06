#![cfg(feature = "precision-f32")]
use blackscholes::{OptionType, InputsSelected, Pricing, ImpliedVolatility, GreeksGeneric};

// Smoke test ensuring generic f32 path compiles and produces finite outputs.
#[test]
fn f32_pricing_greeks_iv_smoke() {
    let inputs = InputsSelected::new(OptionType::Call,
        100.0f32, 100.0f32, None, 0.05f32, 0.01f32, 30.0f32/365.25f32, Some(0.2f32));
    let price = inputs.calc_price().expect("price");
    assert!(price.is_finite());
    let greeks = inputs.calc_all_greeks_generic().expect("greeks");
    assert!(greeks.vega.is_finite());
    let mut iv_inputs = inputs.clone();
    iv_inputs.p = Some(price);
    let iv = iv_inputs.calc_iv(0.0005f32).expect("iv");
    assert!(iv > 0.0f32 && iv.is_finite());
}
