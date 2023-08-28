use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub trait EncodingScheme {
    type Plaintext: Clone;
    type Ciphertext: Clone;
    type SecretKey: Clone;

    fn keygen() -> Self::SecretKey;
    fn encode(sk: &Self::SecretKey, M: &Self::Plaintext) -> Self::Ciphertext;
}

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

    pub fn decode(
        s: &<Self as EncodingScheme>::SecretKey,
        c: &<Self as EncodingScheme>::Ciphertext,
    ) -> <Self as EncodingScheme>::Plaintext {
        &Matrix::append(&-s, &Matrix::<N, N, _>::identity()) * c
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type TestScheme = MatrixRegevEncoding<2, 3, 268369921, 4, 1_000_000>;

    #[test]
    fn test_encode_decode() {
        let sk = TestScheme::keygen();
        let mut msg = Matrix::zero();
        msg[(0, 0)] = Z_N_CycloRaw::from(10000_u64);
        msg[(0, 1)] = Z_N_CycloRaw::from(80000_u64);
        msg[(1, 0)] = Z_N_CycloRaw::from(77000_u64);
        msg[(1, 1)] = Z_N_CycloRaw::from(0_u64);
        let ct = TestScheme::encode(&sk, &msg);
        let diff = &TestScheme::decode(&sk, &ct) - &msg;
        assert!(diff.norm() <= 8); // 8 Gaussian widths
    }

    #[test]
    fn test_hom() {
        let sk = TestScheme::keygen();
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

        let ct1 = TestScheme::encode(&sk, &msg1);
        let ct2 = TestScheme::encode(&sk, &msg2);
        let ct1_plus_ct2 = TestScheme::add_hom(&ct1, &ct2);
        let ct1_times_msg2 = TestScheme::mul_scalar(&ct1, &msg2);
        let diff1 = &TestScheme::decode(&sk, &ct1_plus_ct2) - &(&msg1 + &msg2);
        assert!(diff1.norm() <= 16, "{:?} too big", &diff1); // 8 + 8
        let diff2 = &TestScheme::decode(&sk, &ct1_times_msg2) - &(&msg1 * &msg2);
        assert!(diff2.norm() <= 64, "{:?} too big", &diff2); // 8 * 2 * 4
    }
}
