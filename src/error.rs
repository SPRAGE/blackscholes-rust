use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum BlackScholesError {
    MissingSigma,
    MissingPrice,
    TimeToMaturityZero,
    InvalidLogSK,         // ln(s/k) produced +/-inf or NaN
    ConvergenceFailed,    // generic IV convergence failure
    InvalidInput(&'static str), // generic typed input error
}

impl fmt::Display for BlackScholesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlackScholesError::MissingSigma => write!(f, "Expected Some(f64) for sigma, received None"),
            BlackScholesError::MissingPrice => write!(f, "Option price (p) is required"),
            BlackScholesError::TimeToMaturityZero => write!(f, "Time to maturity (t) is 0"),
            BlackScholesError::InvalidLogSK => write!(f, "Log(s/k) is not finite"),
            BlackScholesError::ConvergenceFailed => write!(f, "Failed to converge"),
            BlackScholesError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for BlackScholesError {}
