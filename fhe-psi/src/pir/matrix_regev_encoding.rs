use crate::math::gadget::gadget_inverse;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::pir::encoding::EncodingScheme;
use crate::pir::gsw_encoding::{GSWEncoding, GSWEncodingParams, GSWEncodingParamsRaw};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub struct MatrixRegevEncoding<
    const N: usize,
    const N_PLUS_ONE: usize,
    const Q: u64,
    const D: usize,
    const W: u64,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

#[allow(non_snake_case)]
pub struct MatrixRegevEncodingParamsRaw {
    pub N: usize,
    pub Q: u64,
    pub D: usize,
    pub W: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

impl MatrixRegevEncodingParamsRaw {
    pub const fn expand(&self) -> MatrixRegevEncodingParams {
        MatrixRegevEncodingParams {
            N: self.N,
            N_PLUS_ONE: self.N + 1,
            Q: self.Q,
            D: self.D,
            W: self.W,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
        }
    }
}

#[allow(non_snake_case)]
pub struct MatrixRegevEncodingParams {
    pub N: usize,
    pub N_PLUS_ONE: usize,
    pub Q: u64,
    pub D: usize,
    pub W: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

#[macro_export]
macro_rules! matrix_regev_encoding {
    ($params: expr) => {
        MatrixRegevEncoding<
            {$params.N},
            {$params.N_PLUS_ONE},
            {$params.Q},
            {$params.D},
            {$params.W},
            {$params.NOISE_WIDTH_MILLIONTHS},
        >
    }
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const Q: u64,
        const D: usize,
        const W: u64,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncodingScheme for MatrixRegevEncoding<N, N_PLUS_ONE, Q, D, W, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = Matrix<N, N, IntModCyclo<D, Q>>;
    type Ciphertext = Matrix<N_PLUS_ONE, N, IntModCycloEval<D, Q, W>>;
    type SecretKey = Matrix<N, 1, IntModCycloEval<D, Q, W>>;

    fn keygen() -> Self::SecretKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, IntModCycloEval<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn encode(s: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, N, IntModCycloEval<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, N, IntModCycloEval<D, Q, W>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<N_PLUS_ONE, N, IntModCycloEval<D, Q, W>> = Matrix::stack(
            &a_t,
            &(&(&(s * &a_t) + &e_mat) + &mu.into_ring(|x| IntModCycloEval::from(x))),
        );
        c_mat
    }
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const Q: u64,
        const D: usize,
        const W: u64,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MatrixRegevEncoding<N, N_PLUS_ONE, Q, D, W, NOISE_WIDTH_MILLIONTHS>
{
    pub fn add_hom(
        lhs: &<Self as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs + rhs
    }

    pub fn sub_hom(
        lhs: &<Self as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs - rhs
    }

    pub fn mul_scalar(
        lhs: &<Self as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Plaintext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs * &rhs.into_ring(|x| IntModCycloEval::from(x))
    }

    pub fn mul_hom_gsw<const M: usize, const G_BASE: u64, const G_LEN: usize>(
        lhs: &<GSWEncoding<N, N_PLUS_ONE, M, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS> as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs * &gadget_inverse::<IntModCycloEval<D, Q, W>, N_PLUS_ONE, M, N, G_BASE, G_LEN>(rhs)
    }

    pub fn decode(
        s: &<Self as EncodingScheme>::SecretKey,
        c: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Plaintext {
        (&Matrix::append(&-s, &Matrix::<N, N, _>::identity()) * c)
            .into_ring(|x| IntModCyclo::from(x))
    }
}

#[allow(non_snake_case)]
pub struct HybridEncodingParamsRaw {
    pub N: usize,
    pub Q: u64,
    pub D: usize,
    pub W: u64,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

pub struct HybridEncodingParams {
    pub matrix_regev: MatrixRegevEncodingParams,
    pub gsw: GSWEncodingParams,
}

impl HybridEncodingParamsRaw {
    pub const fn expand(&self) -> HybridEncodingParams {
        HybridEncodingParams {
            matrix_regev: MatrixRegevEncodingParamsRaw {
                N: self.N,
                Q: self.Q,
                D: self.D,
                W: self.W,
                NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
            }
            .expand(),
            gsw: GSWEncodingParamsRaw {
                N: self.N,
                Q: self.Q,
                D: self.D,
                W: self.W,
                G_BASE: self.G_BASE,
                NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
            }
            .expand(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{gsw_encoding, matrix_regev_encoding};

    const TEST_HYBRID_PARAMS: HybridEncodingParams = HybridEncodingParamsRaw {
        N: 2,
        Q: 268369921,
        D: 4,
        W: 185593570,
        G_BASE: 2,
        NOISE_WIDTH_MILLIONTHS: 1_000_000,
    }
    .expand();

    type TestRegev = matrix_regev_encoding!(TEST_HYBRID_PARAMS.matrix_regev);
    type TestGSW = gsw_encoding!(TEST_HYBRID_PARAMS.gsw);

    #[test]
    fn test_encode_decode() {
        let sk = TestRegev::keygen();
        let mut msg = Matrix::zero();
        msg[(0, 0)] = IntModCyclo::from(10000_u64);
        msg[(0, 1)] = IntModCyclo::from(80000_u64);
        msg[(1, 0)] = IntModCyclo::from(77000_u64);
        msg[(1, 1)] = IntModCyclo::from(0_u64);
        let ct = TestRegev::encode(&sk, &msg);
        let diff = &TestRegev::decode(&sk, &ct) - &msg;
        assert!(diff.norm() <= 8); // |E| = at most 8 widths = 8
    }

    #[test]
    fn test_hom() {
        let sk = TestRegev::keygen();
        let mut msg1 = Matrix::zero();
        msg1[(0, 0)] = IntModCyclo::from(33000_i64);
        msg1[(0, 1)] = IntModCyclo::from(10000_i64);
        msg1[(1, 0)] = IntModCyclo::from(0_i64);
        msg1[(1, 1)] = IntModCyclo::from(65000_i64);

        let mut msg2 = Matrix::zero();
        msg2[(0, 0)] = IntModCyclo::from(1_i64);
        msg2[(0, 1)] = IntModCyclo::from(1_i64);
        msg2[(1, 0)] = IntModCyclo::from(-1_i64);
        msg2[(1, 1)] = IntModCyclo::from(1_i64);

        let ct1 = TestRegev::encode(&sk, &msg1);
        let ct2 = TestRegev::encode(&sk, &msg2);
        let ct1_plus_ct2 = TestRegev::add_hom(&ct1, &ct2);
        let ct1_times_msg2 = TestRegev::mul_scalar(&ct1, &msg2);
        let diff1 = &TestRegev::decode(&sk, &ct1_plus_ct2) - &(&msg1 + &msg2);
        assert!(diff1.norm() <= 16, "{:?} too big", &diff1); // |E_msg1| + |E_msg2| = 8 + 8
        let diff2 = &TestRegev::decode(&sk, &ct1_times_msg2) - &(&msg1 * &msg2);
        assert!(diff2.norm() <= 64, "{:?} too big", &diff2); // n * |msg2| * |E_msg1| = 2 * 1 * 8
    }

    #[test]
    fn test_hom_gsw() {
        let sk = TestRegev::keygen();
        let mut msg1 = Matrix::zero();
        msg1[(0, 0)] = IntModCyclo::from(33000_i64);
        msg1[(0, 1)] = IntModCyclo::from(10000_i64);
        msg1[(1, 0)] = IntModCyclo::from(0_i64);
        msg1[(1, 1)] = IntModCyclo::from(65000_i64);

        let msg2 = IntModCyclo::from(vec![-1_i64, 0, 1, 0]);
        let ct1 = TestRegev::encode(&sk, &msg1);
        let ct2 = TestGSW::encode(&sk, &msg2);
        let ct1_mul_ct2 = TestRegev::mul_hom_gsw::<
            { TEST_HYBRID_PARAMS.gsw.M },
            { TEST_HYBRID_PARAMS.gsw.G_BASE },
            { TEST_HYBRID_PARAMS.gsw.G_LEN },
        >(&ct2, &ct1);
        let diff = &TestRegev::decode(&sk, &ct1_mul_ct2) - &(&msg1 * &msg2);
        assert!(diff.norm() <= 368, "{:?} too big", &diff); // d * |msg2| * |E_msg1| + m * d * |E_msg2| * z / 2 = 4 * 1 * 8 + 84 * 4 * 1
    }
}