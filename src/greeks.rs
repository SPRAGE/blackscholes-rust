use crate::{OptionType, Pricing, BlackScholesError};
use crate::lets_be_rational::normal_distribution::{standard_normal_cdf, standard_normal_pdf};
use crate::{calc_d1d2_generic, calc_nd1nd2_generic, calc_nprimed1_generic, calc_nprimed2_generic};
use crate::numeric::ModelNum;
use crate::generic_inputs::InputsGeneric;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AllGreeksGeneric<T: ModelNum> {
    pub delta: T,
    pub gamma: T,
    pub theta: T,
    pub vega: T,
    pub rho: T,
    pub epsilon: T,
    pub lambda: T,
    pub vanna: T,
    pub charm: T,
    pub veta: T,
    pub vomma: T,
    pub speed: T,
    pub zomma: T,
    pub color: T,
    pub ultima: T,
    pub dual_delta: T,
    pub dual_gamma: T,
}

pub trait GreeksGeneric<T>: Pricing<T>
where
    T: ModelNum,
{
    fn calc_delta(&self) -> Result<T, BlackScholesError>;
    fn calc_gamma(&self) -> Result<T, BlackScholesError>;
    fn calc_theta(&self) -> Result<T, BlackScholesError>;
    fn calc_vega(&self) -> Result<T, BlackScholesError>;
    fn calc_rho(&self) -> Result<T, BlackScholesError>;
    fn calc_epsilon(&self) -> Result<T, BlackScholesError>;
    fn calc_lambda(&self) -> Result<T, BlackScholesError>;
    fn calc_vanna(&self) -> Result<T, BlackScholesError>;
    fn calc_charm(&self) -> Result<T, BlackScholesError>;
    fn calc_veta(&self) -> Result<T, BlackScholesError>;
    fn calc_vomma(&self) -> Result<T, BlackScholesError>;
    fn calc_speed(&self) -> Result<T, BlackScholesError>;
    fn calc_zomma(&self) -> Result<T, BlackScholesError>;
    fn calc_color(&self) -> Result<T, BlackScholesError>;
    fn calc_ultima(&self) -> Result<T, BlackScholesError>;
    fn calc_dual_delta(&self) -> Result<T, BlackScholesError>;
    fn calc_dual_gamma(&self) -> Result<T, BlackScholesError>;
    fn calc_all_greeks_generic(&self) -> Result<AllGreeksGeneric<T>, BlackScholesError>;
}

impl<T: ModelNum> GreeksGeneric<T> for InputsGeneric<T> {
    fn calc_delta(&self) -> Result<T, BlackScholesError> {
        let (nd1, _) = calc_nd1nd2_generic(self)?;
        let sign = T::from(self.option_type as i8).unwrap();
        Ok(sign * (-self.q * self.t).exp() * nd1)
    }
    fn calc_gamma(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let nprimed1 = calc_nprimed1_generic(self)?;
        Ok((-self.q * self.t).exp() * nprimed1 / (self.s * sigma * self.t.sqrt()))
    }
    fn calc_theta(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let nprimed1 = calc_nprimed1_generic(self)?;
        let (nd1, nd2) = calc_nd1nd2_generic(self)?;
        let sign = T::from(self.option_type as i8).unwrap();
        let theta = (-(self.s * sigma * (-self.q * self.t).exp() * nprimed1 / (T::from(2.0).unwrap() * self.t.sqrt()))
            - self.r * self.k * (-self.r * self.t).exp() * nd2 * sign
            + self.q * self.s * (-self.q * self.t).exp() * nd1 * sign)
            / T::DAYS_PER_YEAR;
        Ok(theta)
    }
    fn calc_vega(&self) -> Result<T, BlackScholesError> {
        let nprimed1 = calc_nprimed1_generic(self)?;
        Ok(T::from(0.01).unwrap() * self.s * (-self.q * self.t).exp() * self.t.sqrt() * nprimed1)
    }
    fn calc_rho(&self) -> Result<T, BlackScholesError> {
        let (_, nd2) = calc_nd1nd2_generic(self)?;
        let sign = T::from(self.option_type as i8).unwrap();
        Ok(sign * self.k * self.t * (-self.r * self.t).exp() * nd2 / T::ONE_HUNDRED)
    }
    fn calc_epsilon(&self) -> Result<T, BlackScholesError> {
        let (nd1, _) = calc_nd1nd2_generic(self)?;
        let sign = T::from(self.option_type as i8).unwrap();
        Ok(-self.s * self.t * (-self.q * self.t).exp() * nd1 * sign)
    }
    fn calc_lambda(&self) -> Result<T, BlackScholesError> {
        let delta = self.calc_delta()?;
        Ok(delta * self.s / self.calc_price()?)
    }
    fn calc_vanna(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let nprimed1 = calc_nprimed1_generic(self)?;
        let (_, d2) = calc_d1d2_generic(self)?;
        Ok(d2 * (-self.q * self.t).exp() * nprimed1 * T::from(-0.01).unwrap() / sigma)
    }
    fn calc_charm(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let nprimed1 = calc_nprimed1_generic(self)?;
        let (nd1, _) = calc_nd1nd2_generic(self)?;
        let (_, d2) = calc_d1d2_generic(self)?;
        let e_negqt = (-self.q * self.t).exp();
        let sign = T::from(self.option_type as i8).unwrap();
        Ok(sign * self.q * e_negqt * nd1
            - e_negqt * nprimed1 * (T::from(2.0).unwrap() * (self.r - self.q) * self.t - d2 * sigma * self.t.sqrt())
                / (T::from(2.0).unwrap() * self.t * sigma * self.t.sqrt()))
    }
    fn calc_veta(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let nprimed1 = calc_nprimed1_generic(self)?;
        let (d1, d2) = calc_d1d2_generic(self)?;
        let e_negqt = (-self.q * self.t).exp();
        Ok(-self.s * e_negqt * nprimed1 * self.t.sqrt()
            * (self.q + ((self.r - self.q) * d1) / (sigma * self.t.sqrt())
                - ((T::ONE + d1 * d2) / (T::from(2.0).unwrap() * self.t))))
    }
    fn calc_vomma(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let (d1, d2) = calc_d1d2_generic(self)?;
        Ok(self.calc_vega()? * ((d1 * d2) / sigma))
    }
    fn calc_speed(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let (d1, _) = calc_d1d2_generic(self)?;
        let gamma = self.calc_gamma()?;
        Ok(-gamma / self.s * (d1 / (sigma * self.t.sqrt()) + T::ONE))
    }
    fn calc_zomma(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let (d1, d2) = calc_d1d2_generic(self)?;
        let gamma = self.calc_gamma()?;
        Ok(gamma * ((d1 * d2 - T::ONE) / sigma))
    }
    fn calc_color(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let (d1, d2) = calc_d1d2_generic(self)?;
        let nprimed1 = calc_nprimed1_generic(self)?;
        let e_negqt = (-self.q * self.t).exp();
        Ok(-e_negqt
            * (nprimed1 / (T::from(2.0).unwrap() * self.s * self.t * sigma * self.t.sqrt()))
            * (T::from(2.0).unwrap() * self.q * self.t
                + T::ONE
                + (T::from(2.0).unwrap() * (self.r - self.q) * self.t - d2 * sigma * self.t.sqrt())
                    / (sigma * self.t.sqrt())
                    * d1))
    }
    fn calc_ultima(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let (d1, d2) = calc_d1d2_generic(self)?;
        let vega = self.calc_vega()?;
        Ok(-vega / sigma.powi(2) * (d1 * d2 * (T::ONE - d1 * d2) + d1 * d1 + d2 * d2))
    }
    fn calc_dual_delta(&self) -> Result<T, BlackScholesError> {
        let (_, nd2) = calc_nd1nd2_generic(self)?;
        let e_negqt = (-self.q * self.t).exp();
        Ok(match self.option_type {
            OptionType::Call => -e_negqt * nd2,
            OptionType::Put => e_negqt * nd2,
        })
    }
    fn calc_dual_gamma(&self) -> Result<T, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        let nprimed2 = calc_nprimed2_generic(self)?;
        let e_negqt = (-self.q * self.t).exp();
        Ok(e_negqt * (nprimed2 / (self.k * sigma * self.t.sqrt())))
    }
    fn calc_all_greeks_generic(&self) -> Result<AllGreeksGeneric<T>, BlackScholesError> {
        let sigma = self.sigma.ok_or(BlackScholesError::MissingSigma)?;
        if self.t == T::ZERO {
            return Err(BlackScholesError::TimeToMaturityZero);
        }
        let sqrt_t = self.t.sqrt();
        let e_negqt = (-self.q * self.t).exp();
        let e_negrt = (-self.r * self.t).exp();
        let (d1, d2) = calc_d1d2_generic(self)?;
        let nprimed1 = T::from(standard_normal_pdf(d1.to_f64().unwrap())).unwrap();
        let nprimed2 = T::from(standard_normal_pdf(d2.to_f64().unwrap())).unwrap();
        let (nd1, nd2) = match self.option_type {
            OptionType::Call => (
                T::from(standard_normal_cdf(d1.to_f64().unwrap())).unwrap(),
                T::from(standard_normal_cdf(d2.to_f64().unwrap())).unwrap(),
            ),
            OptionType::Put => (
                T::from(standard_normal_cdf((-d1).to_f64().unwrap())).unwrap(),
                T::from(standard_normal_cdf((-d2).to_f64().unwrap())).unwrap(),
            ),
        };
        let sign = T::from(self.option_type as i8).unwrap();
        let delta = sign * e_negqt * nd1;
        let gamma = e_negqt * nprimed1 / (self.s * sigma * sqrt_t);
        let theta = (-(self.s * sigma * e_negqt * nprimed1 / (T::from(2.0).unwrap() * sqrt_t))
            - self.r * self.k * e_negrt * nd2 * sign
            + self.q * self.s * e_negqt * nd1 * sign)
            / T::DAYS_PER_YEAR;
        let vega = T::from(0.01).unwrap() * self.s * e_negqt * sqrt_t * nprimed1;
        let rho = sign * self.k * self.t * e_negrt * nd2 / T::ONE_HUNDRED;
        let epsilon = -self.s * self.t * e_negqt * nd1 * sign;
        // price for lambda
        let price_call = nd1 * self.s * e_negqt - nd2 * self.k * e_negrt;
        let price_put = nd2 * self.k * e_negrt - nd1 * self.s * e_negqt;
        let zero = T::ZERO;
        let price = match self.option_type {
            OptionType::Call => if price_call > zero { price_call } else { zero },
            OptionType::Put => if price_put > zero { price_put } else { zero },
        };
        let lambda = delta * self.s / price;
        let vanna = d2 * e_negqt * nprimed1 * T::from(-0.01).unwrap() / sigma;
        let charm = sign * self.q * e_negqt * nd1
            - e_negqt * nprimed1 * (T::from(2.0).unwrap() * (self.r - self.q) * self.t - d2 * sigma * sqrt_t)
                / (T::from(2.0).unwrap() * self.t * sigma * sqrt_t);
        let veta = -self.s * e_negqt * nprimed1 * sqrt_t
            * (self.q + ((self.r - self.q) * d1) / (sigma * sqrt_t)
                - ((T::ONE + d1 * d2) / (T::from(2.0).unwrap() * self.t)));
        let vomma = vega * ((d1 * d2) / sigma);
        let speed = -gamma / self.s * (d1 / (sigma * sqrt_t) + T::ONE);
        let zomma = gamma * ((d1 * d2 - T::ONE) / sigma);
        let color = -e_negqt
            * (nprimed1 / (T::from(2.0).unwrap() * self.s * self.t * sigma * sqrt_t))
            * (T::from(2.0).unwrap() * self.q * self.t
                + T::ONE
                + (T::from(2.0).unwrap() * (self.r - self.q) * self.t - d2 * sigma * sqrt_t)
                    / (sigma * sqrt_t)
                    * d1);
        let ultima = -vega / sigma.powi(2)
            * (d1 * d2 * (T::ONE - d1 * d2) + d1 * d1 + d2 * d2);
        let dual_delta = match self.option_type {
            OptionType::Call => -e_negqt * nd2,
            OptionType::Put => e_negqt * nd2,
        };
        let dual_gamma = e_negqt * (nprimed2 / (self.k * sigma * sqrt_t));
        Ok(AllGreeksGeneric {
            delta,
            gamma,
            theta,
            vega,
            rho,
            epsilon,
            lambda,
            vanna,
            charm,
            veta,
            vomma,
            speed,
            zomma,
            color,
            ultima,
            dual_delta,
            dual_gamma,
        })
    }
}

