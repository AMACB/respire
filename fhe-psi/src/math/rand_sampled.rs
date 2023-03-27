//! Traits for randomly sampling from types (esp. rings).
use rand::Rng;

// TODO: test randomness (especially discrete gaussian)

/// Used for sampling elements uniformly.
pub trait RandUniformSampled {
    /// Sample an element uniformly.
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self;
}

/// Used for sampling elements uniformly from `{0, 1}`.
pub trait RandZeroOneSampled {
    /// Sample an element uniformly from `{0, 1}`.
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self;
}

/// Used for sampling elements with the discrete gaussian distribution.
pub trait RandDiscreteGaussianSampled {
    /// Sample an element with the discrete gaussian distribution.
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self;
}
