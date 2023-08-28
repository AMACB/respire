use crate::fhe::gadget::gadget_inverse;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use crate::pir::encoding::EncodingScheme;
use crate::pir::gsw_encoding::GSWEncoding;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub struct MatrixRegevEncoding<
    const N: usize,
    const N_PLUS_ONE: usize,
    const Q: u64,
    const D: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const Q: u64,
        const D: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncodingScheme for MatrixRegevEncoding<N, N_PLUS_ONE, Q, D, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = Matrix<N, N, Z_N_CycloRaw<D, Q>>;
    type Ciphertext = Matrix<N_PLUS_ONE, N, Z_N_CycloRaw<D, Q>>;
    type SecretKey = Matrix<N, 1, Z_N_CycloRaw<D, Q>>;

    fn keygen() -> Self::SecretKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, Z_N_CycloRaw<D, Q>> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn encode(s: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_T: Matrix<1, N, Z_N_CycloRaw<D, Q>> = Matrix::rand_uniform(&mut rng);
        let E: Matrix<N, N, Z_N_CycloRaw<D, Q>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let C: Matrix<N_PLUS_ONE, N, Z_N_CycloRaw<D, Q>> =
            Matrix::stack(&a_T, &(&(&(s * &a_T) + &E) + mu));
        C
    }
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const Q: u64,
        const D: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MatrixRegevEncoding<N, N_PLUS_ONE, Q, D, NOISE_WIDTH_MILLIONTHS>
{
    pub fn add_hom(
        lhs: &<Self as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs + rhs
    }

    pub fn mul_scalar(
        lhs: &<Self as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Plaintext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs * rhs
    }

    pub fn mul_hom_gsw<const M: usize, const G_BASE: u64, const G_LEN: usize>(
        lhs: &<GSWEncoding<N, N_PLUS_ONE, M, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS> as EncodingScheme>::Ciphertext,
        rhs: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Ciphertext {
        lhs * &gadget_inverse::<Z_N_CycloRaw<D, Q>, N_PLUS_ONE, M, N, G_BASE, G_LEN>(rhs)
    }

    pub fn decode(
        s: &<Self as EncodingScheme>::SecretKey,
        c: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Plaintext {
        &Matrix::append(&-s, &Matrix::<N, N, _>::identity()) * c
    }
}

pub trait HybridEncodingParamsTypes {
    const N: usize;
    const N_PLUS_ONE: usize;
    const M: usize;
    const Q: u64;
    const D: usize;
    const G_LEN: usize;
    const G_BASE: u64;
    const NOISE_WIDTH_MILLIONTHS: u64;
    type Regev;
    type GSW;
}

pub struct HybridEncodingParams<
    const N: usize,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

struct HybridEncodingCreate {
    N: usize,
    Q: u64,
    D: usize,
    G_BASE: u64,
    NOISE_WIDTH_MILLIONTHS: u64,
}

#[macro_export]
macro_rules! hybrid_encoding_create {
    ($name: ident, $params: expr) => {
        type $name = HybridEncodingParams<
            { $params.N },
            { $params.Q },
            { $params.D },
            { $params.G_BASE },
            { $params.NOISE_WIDTH_MILLIONTHS },
        >;
        use crate::math::utils::floor_log;
        impl HybridEncodingParamsTypes for $name {
            const N: usize = $params.N;
            const N_PLUS_ONE: usize = Self::N + 1;
            const M: usize = (Self::N + 1) * Self::G_LEN;
            const Q: u64 = $params.Q;
            const D: usize = $params.D;
            const G_LEN: usize = floor_log(Self::G_BASE, Self::Q) + 1;
            const G_BASE: u64 = $params.G_BASE;
            const NOISE_WIDTH_MILLIONTHS: u64 = $params.NOISE_WIDTH_MILLIONTHS;
            type Regev = MatrixRegevEncoding<
                { Self::N },
                { Self::N_PLUS_ONE },
                { Self::Q },
                { Self::D },
                { Self::NOISE_WIDTH_MILLIONTHS },
            >;
            type GSW = GSWEncoding<
                { Self::N },
                { Self::N_PLUS_ONE },
                { Self::M },
                { Self::Q },
                { Self::D },
                { Self::G_BASE },
                { Self::G_LEN },
                { Self::NOISE_WIDTH_MILLIONTHS },
            >;
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;

    hybrid_encoding_create!(
        TestParams,
        HybridEncodingCreate {
            N: 2,
            Q: 268369921,
            D: 4,
            G_BASE: 2,
            NOISE_WIDTH_MILLIONTHS: 1_000_000
        }
    );
    type TestRegev = <TestParams as HybridEncodingParamsTypes>::Regev;
    type TestGSW = <TestParams as HybridEncodingParamsTypes>::GSW;

    #[test]
    fn test_encode_decode() {
        let sk = TestRegev::keygen();
        let mut msg = Matrix::zero();
        msg[(0, 0)] = Z_N_CycloRaw::from(10000_u64);
        msg[(0, 1)] = Z_N_CycloRaw::from(80000_u64);
        msg[(1, 0)] = Z_N_CycloRaw::from(77000_u64);
        msg[(1, 1)] = Z_N_CycloRaw::from(0_u64);
        let ct = TestRegev::encode(&sk, &msg);
        let diff = &TestRegev::decode(&sk, &ct) - &msg;
        assert!(diff.norm() <= 8); // |E| = at most 8 widths = 8
    }

    #[test]
    fn test_hom() {
        let sk = TestRegev::keygen();
        let mut msg1 = Matrix::zero();
        msg1[(0, 0)] = Z_N_CycloRaw::from(33000_i64);
        msg1[(0, 1)] = Z_N_CycloRaw::from(10000_i64);
        msg1[(1, 0)] = Z_N_CycloRaw::from(0_i64);
        msg1[(1, 1)] = Z_N_CycloRaw::from(65000_i64);

        let mut msg2 = Matrix::zero();
        msg2[(0, 0)] = Z_N_CycloRaw::from(1_i64);
        msg2[(0, 1)] = Z_N_CycloRaw::from(1_i64);
        msg2[(1, 0)] = Z_N_CycloRaw::from(-1_i64);
        msg2[(1, 1)] = Z_N_CycloRaw::from(1_i64);

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
        msg1[(0, 0)] = Z_N_CycloRaw::from(33000_i64);
        msg1[(0, 1)] = Z_N_CycloRaw::from(10000_i64);
        msg1[(1, 0)] = Z_N_CycloRaw::from(0_i64);
        msg1[(1, 1)] = Z_N_CycloRaw::from(65000_i64);

        let msg2 = Z_N_CycloRaw::from(vec![-1_i64, 0, 1, 0]);
        let ct1 = TestRegev::encode(&sk, &msg1);
        let ct2 = TestGSW::encode(&sk, &msg2);
        let ct1_mul_ct2 = TestRegev::mul_hom_gsw::<
            { TestParams::M },
            { TestParams::G_BASE },
            { TestParams::G_LEN },
        >(&ct2, &ct1);
        let diff = &TestRegev::decode(&sk, &ct1_mul_ct2) - &(&msg1 * &msg2);
        assert!(diff.norm() <= 368, "{:?} too big", &diff); // d * |msg2| * |E_msg1| + m * d * |E_msg2| * z / 2 = 4 * 1 * 8 + 84 * 4 * 1
    }
}
