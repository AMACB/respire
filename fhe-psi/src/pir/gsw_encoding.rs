use crate::fhe::gadget::build_gadget;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use crate::pir::encoding::EncodingScheme;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub struct GSWEncoding<
    const N: usize,
    const N_PLUS_ONE: usize,
    const M: usize,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const M: usize,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncodingScheme
    for GSWEncoding<N, N_PLUS_ONE, M, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = Z_N_CycloRaw<D, Q>;
    type Ciphertext = Matrix<N_PLUS_ONE, M, Z_N_CycloRaw<D, Q>>;
    type SecretKey = Matrix<N, 1, Z_N_CycloRaw<D, Q>>;

    fn keygen() -> Self::SecretKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, Z_N_CycloRaw<D, Q>> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn encode(s: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_T: Matrix<1, M, Z_N_CycloRaw<D, Q>> = Matrix::rand_uniform(&mut rng);
        let E: Matrix<N, M, Z_N_CycloRaw<D, Q>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let C: Matrix<N_PLUS_ONE, M, Z_N_CycloRaw<D, Q>> =
            &Matrix::stack(&a_T, &(&(s * &a_T) + &E))
                + &(&build_gadget::<Z_N_CycloRaw<D, Q>, N_PLUS_ONE, M, Q, G_BASE, G_LEN>() * mu);
        C
    }
}
