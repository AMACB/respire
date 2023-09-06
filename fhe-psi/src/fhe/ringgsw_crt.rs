//! RingGSW, but with CRT.

use crate::fhe::fhe::*;
use crate::fhe::gsw_utils::*;
use crate::math::matrix::Matrix;
use crate::math::utils::{ceil_log, mod_inverse};
use crate::math::z_n::Z_N;
use crate::math::z_n_crt::Z_N_CRT;
use crate::math::z_n_cyclo_crt::Z_N_CycloRaw_CRT;

pub struct RingGSWCRT<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const Q1_INV: u64,
    const Q2_INV: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

#[derive(Clone, Debug)]
pub struct RingGSWCRTCiphertext<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const Q1_INV: u64,
    const Q2_INV: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, Z_N_CycloRaw_CRT<D, Q1, Q2, Q1_INV, Q2_INV>>,
}

#[derive(Clone, Debug)]
pub struct RingGSWCRTPublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const Q1_INV: u64,
    const Q2_INV: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, Z_N_CycloRaw_CRT<D, Q1, Q2, Q1_INV, Q2_INV>>,
}

#[derive(Clone, Debug)]
pub struct RingGSWCRTSecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const Q1_INV: u64,
    const Q2_INV: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, Z_N_CycloRaw_CRT<D, Q1, Q2, Q1_INV, Q2_INV>>,
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const Q1_INV: u64,
        const Q2_INV: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > FHEScheme
    for RingGSWCRT<
        N_MINUS_1,
        N,
        M,
        P,
        Q,
        Q1,
        Q2,
        Q1_INV,
        Q2_INV,
        D,
        G_BASE,
        G_LEN,
        NOISE_WIDTH_MILLIONTHS,
    >
{
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const Q1_INV: u64,
        const Q2_INV: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncryptionScheme
    for RingGSWCRT<
        N_MINUS_1,
        N,
        M,
        P,
        Q,
        Q1,
        Q2,
        Q1_INV,
        Q2_INV,
        D,
        G_BASE,
        G_LEN,
        NOISE_WIDTH_MILLIONTHS,
    >
{
    type Plaintext = Z_N<P>;
    type Ciphertext = RingGSWCRTCiphertext<N, M, P, Q, Q1, Q2, Q1_INV, Q2_INV, D, G_BASE, G_LEN>;
    type PublicKey = RingGSWCRTPublicKey<N, M, P, Q, Q1, Q2, Q1_INV, Q2_INV, D, G_BASE, G_LEN>;
    type SecretKey = RingGSWCRTSecretKey<N, M, P, Q, Q1, Q2, Q1_INV, Q2_INV, D, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let (A, s_T) = gsw_keygen::<N_MINUS_1, N, M, _, NOISE_WIDTH_MILLIONTHS>();
        (Self::PublicKey { A }, Self::SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mu = Z_N_CycloRaw_CRT::<D, Q1, Q2, Q1_INV, Q2_INV>::from(u64::from(*mu));
        let ct = gsw_encrypt_pk::<N, M, G_BASE, G_LEN, _>(&pk.A, mu);
        Self::Ciphertext { ct }
    }

    fn encrypt_sk(sk: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mu = Z_N_CycloRaw_CRT::<D, Q1, Q2, Q1_INV, Q2_INV>::from(u64::from(*mu));
        let ct = gsw_encrypt_sk::<N_MINUS_1, N, M, G_BASE, G_LEN, _, NOISE_WIDTH_MILLIONTHS>(
            &sk.s_T, mu,
        );
        Self::Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let pt = gsw_half_decrypt::<N, M, P, Q, G_BASE, G_LEN, _>(s_T, ct);
        gsw_round::<P, Q, Z_N_CRT<Q1, Q2, Q1_INV, Q2_INV>>((&pt).into())
    }
}

/*
 * RingGSWCRT homomorphic addition / multiplication
 */

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const Q1_INV: u64,
        const Q2_INV: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddHomEncryptionScheme
    for RingGSWCRT<
        N_MINUS_1,
        N,
        M,
        P,
        Q,
        Q1,
        Q2,
        Q1_INV,
        Q2_INV,
        D,
        G_BASE,
        G_LEN,
        NOISE_WIDTH_MILLIONTHS,
    >
{
    fn add_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        Self::Ciphertext {
            ct: ciphertext_add::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs.ct),
        }
    }
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const Q1_INV: u64,
        const Q2_INV: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulHomEncryptionScheme
    for RingGSWCRT<
        N_MINUS_1,
        N,
        M,
        P,
        Q,
        Q1,
        Q2,
        Q1_INV,
        Q2_INV,
        D,
        G_BASE,
        G_LEN,
        NOISE_WIDTH_MILLIONTHS,
    >
{
    fn mul_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        Self::Ciphertext {
            ct: ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs.ct),
        }
    }
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const Q1_INV: u64,
        const Q2_INV: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddScalarEncryptionScheme<Z_N<P>>
    for RingGSWCRT<
        N_MINUS_1,
        N,
        M,
        P,
        Q,
        Q1,
        Q2,
        Q1_INV,
        Q2_INV,
        D,
        G_BASE,
        G_LEN,
        NOISE_WIDTH_MILLIONTHS,
    >
{
    fn add_scalar(lhs: &Self::Ciphertext, rhs: &Z_N<P>) -> Self::Ciphertext {
        let rhs_q = Z_N_CycloRaw_CRT::<D, Q1, Q2, Q1_INV, Q2_INV>::from(u64::from(*rhs));
        Self::Ciphertext {
            ct: scalar_ciphertext_add::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs_q),
        }
    }
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const Q1_INV: u64,
        const Q2_INV: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulScalarEncryptionScheme<Z_N<P>>
    for RingGSWCRT<
        N_MINUS_1,
        N,
        M,
        P,
        Q,
        Q1,
        Q2,
        Q1_INV,
        Q2_INV,
        D,
        G_BASE,
        G_LEN,
        NOISE_WIDTH_MILLIONTHS,
    >
{
    fn mul_scalar(lhs: &Self::Ciphertext, rhs: &Z_N<P>) -> Self::Ciphertext {
        let rhs_q = Z_N_CycloRaw_CRT::<D, Q1, Q2, Q1_INV, Q2_INV>::from(u64::from(*rhs));
        Self::Ciphertext {
            ct: scalar_ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs_q),
        }
    }
}
/*
 * RingGSWCRT params
 */

pub struct Params {
    pub N: usize,
    pub M: usize,
    pub P: u64,
    pub Q1: u64,
    pub Q2: u64,
    pub D: usize,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

macro_rules! gsw_from_params {
    ($params:expr) => {
        RingGSWCRT<
            { $params.N - 1 },
            { $params.N },
            { $params.M },
            { $params.P },
            { $params.Q1 * $params.Q2 },
            { $params.Q1 },
            { $params.Q2 },
            { mod_inverse($params.Q1, $params.Q2) },
            { mod_inverse($params.Q2, $params.Q1) },
            { $params.D },
            { $params.G_BASE },
            { ceil_log($params.G_BASE, $params.Q1 * $params.Q2) },
            { $params.NOISE_WIDTH_MILLIONTHS },
        >
    }
}

/*
 * Pre-defined sets of parameters
 */

pub const RingGSWCRT_TEST_PARAMS: Params = Params {
    N: 2,
    M: 112,
    P: 31,
    Q1: 268369921,
    Q2: 249561089,
    D: 4,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
};

pub type RingGSWCRTTest = gsw_from_params!(RingGSWCRT_TEST_PARAMS);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn keygen_is_correct() {
        let threshold =
            4f64 * (RingGSWCRT_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let (A, s_T) = RingGSWCRTTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..RingGSWCRT_TEST_PARAMS.M {
            assert!(
                (e[(0, i)].norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = RingGSWCRTTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = RingGSWCRTTest::encrypt(&A, &mu);
            let pt = RingGSWCRTTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = RingGSWCRTTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = Z_N::from(i);
                let mu2 = Z_N::from(j);
                let ct1 = RingGSWCRTTest::encrypt(&A, &mu1);
                let ct2 = RingGSWCRTTest::encrypt(&A, &mu2);

                let pt_add_ct =
                    RingGSWCRTTest::decrypt(&s_T, &(RingGSWCRTTest::add_hom(&ct1, &ct2)));
                let pt_mul_ct =
                    RingGSWCRTTest::decrypt(&s_T, &(RingGSWCRTTest::mul_hom(&ct1, &ct2)));
                let pt_mul_scalar =
                    RingGSWCRTTest::decrypt(&s_T, &(RingGSWCRTTest::mul_scalar(&ct1, &mu2)));

                assert_eq!(pt_add_ct, mu1 + mu2, "ciphertext addition failed");
                assert_eq!(pt_mul_ct, mu1 * mu2, "ciphertext multiplication failed");
                assert_eq!(pt_mul_scalar, mu1 * mu2, "multiplication by scalar failed");
            }
        }
    }

    #[test]
    fn homomorphism_mul_multiple_correct() {
        let (A, s_T) = RingGSWCRTTest::keygen();
        let mu1 = Z_N::from(5_u64);
        let mu2 = Z_N::from(12_u64);
        let mu3 = Z_N::from(6_u64);
        let mu4 = Z_N::from(18_u64);

        let ct1 = RingGSWCRTTest::encrypt(&A, &mu1);
        let ct2 = RingGSWCRTTest::encrypt(&A, &mu2);
        let ct3 = RingGSWCRTTest::encrypt(&A, &mu3);
        let ct4 = RingGSWCRTTest::encrypt(&A, &mu4);

        let ct12 = RingGSWCRTTest::mul_hom(&ct1, &ct2);
        let ct34 = RingGSWCRTTest::mul_hom(&ct3, &ct4);
        let ct1234 = RingGSWCRTTest::mul_hom(&ct12, &ct34);
        // let ct31234 = &ct3 * &ct1234;

        let pt12 = RingGSWCRTTest::decrypt(&s_T, &ct12);
        let pt34 = RingGSWCRTTest::decrypt(&s_T, &ct34);
        let pt1234 = RingGSWCRTTest::decrypt(&s_T, &ct1234);
        // let pt31234 = gsw::decrypt(&s_T, &ct31234);

        assert_eq!(pt12, &mu1 * &mu2);
        assert_eq!(pt34, &mu3 * &mu4);
        assert_eq!(pt1234, &(&(&mu1 * &mu2) * &mu3) * &mu4);
        // assert_eq!(pt31234, &(&(&(&mu1 * &mu2) * &mu3) * &mu4) * &mu3);
    }
}
