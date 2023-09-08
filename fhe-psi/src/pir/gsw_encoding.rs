use crate::math::gadget::build_gadget;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::utils::floor_log;
use crate::pir::encoding::EncodingScheme;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

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

#[allow(non_snake_case)]
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
        let t_mat = floor_log(self.G_BASE, self.Q) + 1;
        GSWEncodingParams {
            N: self.N,
            N_PLUS_ONE: self.N + 1,
            M: (self.N + 1) * t_mat,
            Q: self.Q,
            D: self.D,
            W: self.W,
            G_BASE: self.G_BASE,
            G_LEN: t_mat,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
        }
    }
}

#[allow(non_snake_case)]
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
    type Plaintext = IntModCyclo<D, Q>;
    type Ciphertext = Matrix<N_PLUS_ONE, M, IntModCycloEval<D, Q, W>>;
    type SecretKey = Matrix<N, 1, IntModCycloEval<D, Q, W>>;

    fn keygen() -> Self::SecretKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, IntModCycloEval<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn encode(s: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M, IntModCycloEval<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, M, IntModCycloEval<D, Q, W>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<N_PLUS_ONE, M, IntModCycloEval<D, Q, W>> =
            &Matrix::stack(&a_t, &(&(s * &a_t) + &e_mat))
                + &(&build_gadget::<IntModCycloEval<D, Q, W>, N_PLUS_ONE, M, G_BASE, G_LEN>()
                    * &IntModCycloEval::from(mu));
        c_mat
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
        &build_gadget::<IntModCycloEval<D, Q, W>, N_PLUS_ONE, M, G_BASE, G_LEN>() - &ct
    }
}
