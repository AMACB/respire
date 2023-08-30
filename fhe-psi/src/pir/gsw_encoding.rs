use crate::fhe::gadget::build_gadget;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::utils::floor_log;
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use crate::pir::encoding::EncodingScheme;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use crate::math::z_n_cyclo_ntt::Z_N_CycloNTT;

pub struct GSWEncoding<
    const N: usize,
    const N_PLUS_ONE: usize,
    const M: usize,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

pub struct GSWEncodingParamsRaw {
    pub N: usize,
    pub Q: u64,
    pub D: usize,
    pub W: u64,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

impl GSWEncodingParamsRaw {
    pub const fn expand(&self) -> GSWEncodingParams {
        let T = floor_log(self.G_BASE, self.Q) + 1;
        GSWEncodingParams {
            N: self.N,
            N_PLUS_ONE: self.N + 1,
            M: (self.N + 1) * T,
            Q: self.Q,
            D: self.D,
            W: self.W,
            G_BASE: self.G_BASE,
            G_LEN: T,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
        }
    }
}

pub struct GSWEncodingParams {
    pub N: usize,
    pub N_PLUS_ONE: usize,
    pub M: usize,
    pub Q: u64,
    pub D: usize,
    pub W: u64,
    pub G_BASE: u64,
    pub G_LEN: usize,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

#[macro_export]
macro_rules! gsw_encoding {
    ($params: expr) => {
        GSWEncoding<
            {$params.N},
            {$params.N_PLUS_ONE},
            {$params.M},
            {$params.Q},
            {$params.D},
            {$params.W},
            {$params.G_BASE},
            {$params.G_LEN},
            {$params.NOISE_WIDTH_MILLIONTHS},
        >
    }
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const M: usize,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncodingScheme
    for GSWEncoding<N, N_PLUS_ONE, M, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = Z_N_CycloRaw<D, Q>;
    type Ciphertext = Matrix<N_PLUS_ONE, M, Z_N_CycloNTT<D, Q, W>>;
    type SecretKey = Matrix<N, 1, Z_N_CycloNTT<D, Q, W>>;

    fn keygen() -> Self::SecretKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, Z_N_CycloNTT<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn encode(s: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_T: Matrix<1, M, Z_N_CycloNTT<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        let E: Matrix<N, M, Z_N_CycloNTT<D, Q, W>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let C: Matrix<N_PLUS_ONE, M, Z_N_CycloNTT<D, Q, W>> =
            &Matrix::stack(&a_T, &(&(s * &a_T) + &E))
                + &(&build_gadget::<Z_N_CycloNTT<D, Q, W>, N_PLUS_ONE, M, Q, G_BASE, G_LEN>() * &Z_N_CycloNTT::from(mu));
        C
    }
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const M: usize,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > GSWEncoding<N, N_PLUS_ONE, M, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    pub fn complement(
        ct: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        &build_gadget::<Z_N_CycloNTT<D, Q, W>, N_PLUS_ONE, M, Q, G_BASE, G_LEN>() - &ct
    }
}
