use std::f64::consts::PI;

use num_traits::Float;
use statrs::consts::SQRT_2PI;

use crate::{
    lets_be_rational::implied_volatility_from_a_transformed_rational_guess,
    pricing::Pricing, BlackScholesError, A, B, C, D, _E, F,
};
use crate::lets_be_rational::normal_distribution::standard_normal_pdf;
use crate::generic_inputs::InputsGeneric;
use crate::numeric::ModelNum;
// generic helpers
use crate::{calc_d1d2_generic, calc_nprimed1_generic};

pub trait ImpliedVolatility<T>: Pricing<T>
where
    T: ModelNum,
{
    fn calc_iv(&self, tolerance: T) -> Result<T, BlackScholesError>;
    fn calc_rational_iv(&self) -> Result<f64, BlackScholesError>;
}

impl<T: ModelNum> ImpliedVolatility<T> for InputsGeneric<T> {
    /// Calculates the implied volatility of the option.
    /// Tolerance is the max error allowed for the implied volatility,
    /// the lower the tolerance the more iterations will be required.
    /// Recommended to be a value between 0.001 - 0.0001 for highest efficiency/accuracy.
    /// Initializes estimation of sigma using Brenn and Subrahmanyam (1998) method of calculating initial iv estimation.
    /// Uses Newton Raphson algorithm to calculate implied volatility.
    /// # Requires
    /// s, k, r, q, t, p
    /// # Returns
    /// f64 of the implied volatility of the option.
    /// # Example:
    /// ```
    /// use blackscholes::{Inputs, OptionType, ImpliedVolatility};
    /// let inputs = Inputs::new(OptionType::Call, 100.0, 100.0, Some(0.5), 0.05, 0.2, 20.0/365.25, None);
    /// let iv = inputs.calc_iv(0.0001).unwrap();
    /// ```
    /// Initial estimation of sigma using Modified Corrado-Miller from ["A MODIFIED CORRADO-MILLER IMPLIED VOLATILITY ESTIMATOR" (2007) by Piotr P√luciennik](https://sin.put.poznan.pl/files/download/37938) method of calculating initial iv estimation.
    /// A more accurate method is the "Let's be rational" method from ["Let’s be rational" (2016) by Peter Jackel](http://www.jaeckel.org/LetsBeRational.pdf)
    /// however this method is much more complicated, it is available as calc_rational_iv().
    #[allow(non_snake_case)]
    fn calc_iv(&self, tolerance: T) -> Result<T, BlackScholesError> {
        let mut inputs = self.clone();
        let p = self.p.ok_or(BlackScholesError::MissingPrice)?;
        // Initialize estimation of sigma using Brenn and Subrahmanyam (1998) method of calculating initial iv estimation.
        // commented out to replace with modified corrado-miller method.
        // let mut sigma: f64 = (PI2 / inputs.t).sqrt() * (p / inputs.s);

        // Use f64 path for initial guess; then cast to T.
        let s_f = inputs.s.to_f64().unwrap();
        let k_f = inputs.k.to_f64().unwrap();
        let r_f = inputs.r.to_f64().unwrap();
        let q_f = inputs.q.to_f64().unwrap();
        let t_f = inputs.t.to_f64().unwrap();
        let p_f = p.to_f64().unwrap();
        if t_f == 0.0 { return Err(BlackScholesError::TimeToMaturityZero); }
        let X = k_f * (-r_f * t_f).exp();
        let fminusX = s_f - X;
        let fplusX = s_f + X;
        let oneoversqrtT = 1.0 / t_f.sqrt();
        let x = oneoversqrtT * (SQRT_2PI / fplusX);
        let y = p_f - (s_f - k_f) / 2.0 + ((p_f - fminusX / 2.0).powi(2) - fminusX.powi(2) / PI).sqrt();
        let mut sigma_f = oneoversqrtT
            * (SQRT_2PI / fplusX)
            * (p_f - fminusX / 2.0 + ((p_f - fminusX / 2.0).powi(2) - fminusX.powi(2) / PI).sqrt())
            + A + B / x + C * y + D / x.powi(2) + _E * y.powi(2) + F * y / x;
        if !sigma_f.is_finite() { return Err(BlackScholesError::ConvergenceFailed); }
        let mut sigma: T = T::from(sigma_f).unwrap();
        let mut diff = T::from(100.0).unwrap();

        // Uses Newton Raphson algorithm to calculate implied volatility.
        // Test if the difference between calculated option price and actual option price is > tolerance,
        // if so then iterate until the difference is less than tolerance
        while diff.abs() > tolerance {
            inputs.sigma = Some(sigma);
            let price = inputs.calc_price()?; // T
            diff = price - p;
            let nprimed1 = calc_nprimed1_generic(&inputs)?;
            let vega_scaled = T::from(0.01).unwrap() * inputs.s * (-inputs.q * inputs.t).exp() * inputs.t.sqrt() * nprimed1;
            let vega_raw = vega_scaled * T::ONE_HUNDRED;
            sigma = sigma - diff / vega_raw;
            if !sigma.is_finite() { return Err(BlackScholesError::ConvergenceFailed); }
        }
        Ok(sigma)
    }

    /// Calculates the implied volatility of the option.
    /// # Requires
    /// s, k, r, t, p
    /// # Returns
    /// f64 of the implied volatility of the option.
    /// # Example:
    /// ```
    /// use blackscholes::{Inputs, OptionType, ImpliedVolatility};
    /// let inputs = Inputs::new(OptionType::Call, 100.0, 100.0, Some(0.2), 0.05, 0.05, 20.0/365.25, None);
    /// let iv = inputs.calc_rational_iv().unwrap();
    /// ```

    fn calc_rational_iv(&self) -> Result<f64, BlackScholesError> {
        // Promote generic inputs to f64 for Jackel rational solver
        let p = self.p.ok_or(BlackScholesError::MissingPrice)?.to_f64().unwrap();
        let t = self.t.to_f64().unwrap();
        if t == 0.0 { return Err(BlackScholesError::TimeToMaturityZero); }
        let r_f64 = self.r.to_f64().unwrap();
        let q_f64 = self.q.to_f64().unwrap();
        let s_f64 = self.s.to_f64().unwrap();
        let k_f64 = self.k.to_f64().unwrap();
        let p_adj = p * (r_f64 * t).exp();
        let f = s_f64 * ((r_f64 - q_f64) * t).exp();
        let sigma = implied_volatility_from_a_transformed_rational_guess(
            p_adj, f, k_f64, t, self.option_type);
        if sigma.is_nan() || sigma.is_infinite() || sigma < 0.0 { return Err(BlackScholesError::ConvergenceFailed); }
        Ok(sigma)
    }
}
