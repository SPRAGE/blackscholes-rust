//! Generic version of Inputs allowing different numeric types.
//! This is an internal transitional module; public type alias will expose selected precision.

use core::fmt::{Display, Formatter, Result as FmtResult};

use crate::{OptionType};
use crate::numeric::ModelNum;

#[derive(Debug, Clone, PartialEq)]
pub struct InputsGeneric<T: ModelNum> {
    pub option_type: OptionType,
    pub s: T,
    pub k: T,
    pub p: Option<T>,
    pub r: T,
    pub q: T,
    pub t: T,
    pub sigma: Option<T>,
}

impl<T: ModelNum> InputsGeneric<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(option_type: OptionType, s: T, k: T, p: Option<T>, r: T, q: T, t: T, sigma: Option<T>) -> Self {
        Self { option_type, s, k, p, r, q, t, sigma }
    }
}

impl<T: ModelNum + Display> Display for InputsGeneric<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        writeln!(f, "Option type: {}", self.option_type)?;
        writeln!(f, "Stock price: {}", self.s)?;
        writeln!(f, "Strike price: {}", self.k)?;
        match self.p { Some(p) => writeln!(f, "Option price: {}", p)?, None => writeln!(f, "Option price: None")?, };
        writeln!(f, "Risk-free rate: {}", self.r)?;
        writeln!(f, "Dividend yield: {}", self.q)?;
        writeln!(f, "Time to maturity: {}", self.t)?;
        match self.sigma { Some(sig) => writeln!(f, "Volatility: {}", sig)?, None => writeln!(f, "Volatility: None")?, };
        Ok(())
    }
}
