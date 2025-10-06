use num_traits::Float;

use crate::{lets_be_rational, BlackScholesError, OptionType};
use crate::lets_be_rational::normal_distribution::{standard_normal_cdf, standard_normal_pdf};
// Import generic helper function for pricing
use crate::calc_nd1nd2_generic;
use crate::numeric::ModelNum;
use crate::generic_inputs::InputsGeneric;

pub trait Pricing<T>
where
    T: Float,
{
    fn calc_price(&self) -> Result<T, BlackScholesError>;
    fn calc_rational_price(&self) -> Result<f64, BlackScholesError>;
}

// Legacy f64-specific Pricing impl removed; generic impl below (InputsGeneric<f64>) covers f64.

// Generic implementation (transitional). Rational price kept f64-only for now.
impl<T: ModelNum> Pricing<T> for InputsGeneric<T> {
    fn calc_price(&self) -> Result<T, BlackScholesError> {
        let (nd1, nd2) = calc_nd1nd2_generic(self)?;
        let e_negqt = (-self.q * self.t).exp();
        let e_negrt = (-self.r * self.t).exp();
        let zero = T::ZERO;
        let call_val = nd1 * self.s * e_negqt - nd2 * self.k * e_negrt;
        let put_val = nd2 * self.k * e_negrt - nd1 * self.s * e_negqt;
        let price = match self.option_type {
            OptionType::Call => if call_val > zero { call_val } else { zero },
            OptionType::Put => if put_val > zero { put_val } else { zero },
        };
        Ok(price)
    }

    fn calc_rational_price(&self) -> Result<f64, BlackScholesError> { // fallback: promote to f64
        // Only meaningful if T can convert to f64
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let forward = self.s.to_f64().unwrap() * ((self.r.to_f64().unwrap() - self.q.to_f64().unwrap()) * self.t.to_f64().unwrap()).exp();
        let undiscounted = lets_be_rational::black(forward, self.k.to_f64().unwrap(), sigma.to_f64().unwrap(), self.t.to_f64().unwrap(), self.option_type);
        Ok(undiscounted * (-self.r.to_f64().unwrap() * self.t.to_f64().unwrap()).exp())
    }
}
