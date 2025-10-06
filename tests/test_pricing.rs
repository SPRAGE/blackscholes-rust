use assert_approx_eq::assert_approx_eq;
use blackscholes::{Inputs, OptionType, Pricing};

#[cfg(feature = "precision-f32")]
const EPS: f32 = 0.001;
#[cfg(feature = "precision-f64")]
const EPS: f64 = 0.001;

const INPUTS_CALL_OTM: Inputs = Inputs {
    option_type: OptionType::Call,
    s: 100.0,
    k: 110.0,
    p: None,
    r: 0.05,
    q: 0.05,
    t: 20.0 / 365.25,
    sigma: Some(0.2),
};
const INPUTS_CALL_ITM: Inputs = Inputs {
    option_type: OptionType::Call,
    s: 100.0,
    k: 90.0,
    p: None,
    r: 0.05,
    q: 0.05,
    t: 20.0 / 365.25,
    sigma: Some(0.2),
};
const INPUTS_PUT_OTM: Inputs = Inputs {
    option_type: OptionType::Put,
    s: 100.0,
    k: 90.0,
    p: None,
    r: 0.05,
    q: 0.05,
    t: 20.0 / 365.25,
    sigma: Some(0.2),
};
const INPUTS_PUT_ITM: Inputs = Inputs {
    option_type: OptionType::Put,
    s: 100.0,
    k: 110.0,
    p: None,
    r: 0.05,
    q: 0.05,
    t: 20.0 / 365.25,
    sigma: Some(0.2),
};

const INPUTS_BRANCH_CUT: Inputs = Inputs {
    option_type: OptionType::Put,
    s: 100.0,
    k: 100.0,
    p: None,
    r: 0.0,
    q: 0.0,
    sigma: Some(0.421),
    t: 1.0,
};

#[test]
fn price_call_otm() {
    #[cfg(feature = "precision-f32")] { assert_approx_eq!(INPUTS_CALL_OTM.calc_price().unwrap(), 0.0376_f32, EPS); }
    #[cfg(feature = "precision-f64")] { assert_approx_eq!(INPUTS_CALL_OTM.calc_price().unwrap(), 0.0376_f64, EPS); }
}
#[test]
fn price_call_itm() {
    #[cfg(feature = "precision-f32")] { assert_approx_eq!(INPUTS_CALL_ITM.calc_price().unwrap(), 9.9913_f32, EPS); }
    #[cfg(feature = "precision-f64")] { assert_approx_eq!(INPUTS_CALL_ITM.calc_price().unwrap(), 9.9913_f64, EPS); }
}

#[test]
fn price_put_otm() {
    #[cfg(feature = "precision-f32")] { assert_approx_eq!(INPUTS_PUT_OTM.calc_price().unwrap(), 0.01867_f32, EPS); }
    #[cfg(feature = "precision-f64")] { assert_approx_eq!(INPUTS_PUT_OTM.calc_price().unwrap(), 0.01867_f64, EPS); }
}
#[test]
fn price_put_itm() {
    #[cfg(feature = "precision-f32")] { assert_approx_eq!(INPUTS_PUT_ITM.calc_price().unwrap(), 10.0103_f32, EPS); }
    #[cfg(feature = "precision-f64")] { assert_approx_eq!(INPUTS_PUT_ITM.calc_price().unwrap(), 10.0103_f64, EPS); }
}

#[test]
fn price_using_lets_be_rational() {
    // compare the results from calc_price() and calc_rational_price() for the options above
    // rational price always returns f64; cast predicted when in f32 mode
    let rp_call_otm = INPUTS_CALL_OTM.calc_rational_price().unwrap();
    let price_call_otm = INPUTS_CALL_OTM.calc_price().unwrap() as f64;
    assert_approx_eq!(price_call_otm, rp_call_otm, 0.001);

    let rp_call_itm = INPUTS_CALL_ITM.calc_rational_price().unwrap();
    let price_call_itm = INPUTS_CALL_ITM.calc_price().unwrap() as f64;
    assert_approx_eq!(price_call_itm, rp_call_itm, 0.001);

    let rp_put_otm = INPUTS_PUT_OTM.calc_rational_price().unwrap();
    let price_put_otm = INPUTS_PUT_OTM.calc_price().unwrap() as f64;
    assert_approx_eq!(price_put_otm, rp_put_otm, 0.001);

    let rp_put_itm = INPUTS_PUT_ITM.calc_rational_price().unwrap();
    let price_put_itm = INPUTS_PUT_ITM.calc_price().unwrap() as f64;
    assert_approx_eq!(price_put_itm, rp_put_itm, 0.001);
}

#[test]
fn test_rational_price_near_branch_cut() {
    let expect = 16.67224_f64; // rational still f64 specific
    assert_approx_eq!(INPUTS_BRANCH_CUT.calc_rational_price().unwrap(), expect, 0.001);
}
