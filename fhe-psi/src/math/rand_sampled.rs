use rand::Rng;

// TODO: test randomness (especially discrete gaussian)

pub trait RandUniformSampled {
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self;
}

pub trait RandZeroOneSampled {
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self;
}

pub trait RandDiscreteGaussianSampled {
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self;
}
