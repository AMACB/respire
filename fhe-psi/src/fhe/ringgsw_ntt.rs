//! Ring GSW, but polynomials are represented via their DFT. See `Z_N_CycloNTT`.

use crate::fhe::fhe::*;
use crate::fhe::gsw_utils::*;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::matrix::Matrix;
use crate::math::utils::ceil_log;

pub struct RingGSWNTT<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

#[derive(Clone, Debug)]
pub struct RingGSWNTTCiphertext<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, IntModCycloEval<D, Q, W>>,
}

#[derive(Clone, Debug)]
pub struct RingGSWNTTPublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, IntModCycloEval<D, Q, W>>,
}

#[derive(Clone, Debug)]
pub struct RingGSWNTTSecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, IntModCycloEval<D, Q, W>>,
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > FHEScheme for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncryptionScheme
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = IntModCyclo<D, P>;
    type Ciphertext = RingGSWNTTCiphertext<N, M, P, Q, D, W, G_BASE, G_LEN>;
    type PublicKey = RingGSWNTTPublicKey<N, M, P, Q, D, W, G_BASE, G_LEN>;
    type SecretKey = RingGSWNTTSecretKey<N, M, P, Q, D, W, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let (A, s_T) = gsw_keygen::<N_MINUS_1, N, M, _, NOISE_WIDTH_MILLIONTHS>();
        (Self::PublicKey { A }, Self::SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let ct = gsw_encrypt_pk::<N, M, G_BASE, G_LEN, _>(&pk.A, mu.include_into().into());
        Self::Ciphertext { ct }
    }

    fn encrypt_sk(sk: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let ct = gsw_encrypt_sk::<N_MINUS_1, N, M, G_BASE, G_LEN, _, NOISE_WIDTH_MILLIONTHS>(
            &sk.s_T,
            mu.include_into().into(),
        );
        Self::Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Self::Plaintext {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let pt_eval = gsw_half_decrypt::<N, M, P, Q, G_BASE, G_LEN, _>(s_T, ct);
        let pt: IntModCyclo<D, Q> = pt_eval.into();
        pt.round_down_into()
    }
}

/*
 * homomorphic addition / multiplication
 */
impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddHomEncryptionScheme
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulHomEncryptionScheme
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddScalarEncryptionScheme<IntModCyclo<D, P>>
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn add_scalar(lhs: &Self::Ciphertext, rhs: &Self::Plaintext) -> Self::Ciphertext {
        let rhs_q = rhs.include_into().into();
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
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulScalarEncryptionScheme<IntModCyclo<D, P>>
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn mul_scalar(lhs: &Self::Ciphertext, rhs: &Self::Plaintext) -> Self::Ciphertext {
        let rhs_q = rhs.include_into().into();
        Self::Ciphertext {
            ct: scalar_ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs_q),
        }
    }
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > NegEncryptionScheme
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn negate(ct: &Self::Ciphertext) -> Self::Ciphertext {
        Self::Ciphertext { ct: -&ct.ct }
    }
}

pub struct Params {
    pub N: usize,
    pub M: usize,
    pub P: u64,
    pub Q: u64,
    pub D: usize,
    pub W: u64,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

macro_rules! ring_gsw_ntt_from_params {
    ($params:expr) => {
        RingGSWNTT<
            { $params.N - 1 },
            { $params.N },
            { $params.M },
            { $params.P },
            { $params.Q },
            { $params.D },
            { $params.W },
            { $params.G_BASE },
            { ceil_log($params.G_BASE, $params.Q) },
            { $params.NOISE_WIDTH_MILLIONTHS },
        >
    }
}

pub const RING_GSW_NTT_TEST_PARAMS: Params = Params {
    N: 2,
    M: 56,
    P: 31,
    Q: 268369921,
    D: 4,
    W: 185593570,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
};

pub const RING_GSW_NTT_TEST_MEDIUM_PARAMS: Params = Params {
    N: 2,
    M: 56,
    P: 31,
    Q: 268369921,
    // D: 256,
    // W: 63703579,
    D: 2048,
    W: 66294444,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 1,
};

// For parameter selection: fix Q, then choose smallest N*D for appropriate security.

pub type RingGSWNTTTest = ring_gsw_ntt_from_params!(RING_GSW_NTT_TEST_PARAMS);
pub type RingGSWNTTTestMedium = ring_gsw_ntt_from_params!(RING_GSW_NTT_TEST_MEDIUM_PARAMS);

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::ring_elem::NormedRingElement;

    #[test]
    fn keygen_is_correct() {
        let threshold =
            4f64 * (RING_GSW_NTT_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let (A, s_T) = RingGSWNTTTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..RING_GSW_NTT_TEST_PARAMS.M {
            assert!(
                (IntModCyclo::from(e[(0, i)].clone()).norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = RingGSWNTTTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = i;
            let ct = RingGSWNTTTest::encrypt(&A, &mu.into());
            let pt = RingGSWNTTTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu.into(), "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = RingGSWNTTTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = i.into();
                let mu2 = j.into();
                let ct1 = RingGSWNTTTest::encrypt(&A, &mu1);
                let ct2 = RingGSWNTTTest::encrypt(&A, &mu2);
                let pt_add_ct = RingGSWNTTTest::decrypt(&s_T, &RingGSWNTTTest::add_hom(&ct1, &ct2));
                let pt_mul_ct = RingGSWNTTTest::decrypt(&s_T, &RingGSWNTTTest::mul_hom(&ct1, &ct2));
                let pt_mul_scalar =
                    RingGSWNTTTest::decrypt(&s_T, &RingGSWNTTTest::mul_scalar(&ct1, &j.into()));
                assert_eq!(pt_add_ct, &mu1 + &mu2, "ciphertext addition failed");
                assert_eq!(pt_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
                assert_eq!(
                    pt_mul_scalar,
                    &mu1 * &mu2,
                    "multiplication by scalar failed"
                );
            }
        }
    }

    #[test]
    fn homomorphism_mul_multiple_correct() {
        let (A, s_T) = RingGSWNTTTest::keygen();
        let mu1 = IntModCyclo::from(5_u64);
        let mu2 = IntModCyclo::from(12_u64);
        let mu3 = IntModCyclo::from(6_u64);
        let mu4 = IntModCyclo::from(18_u64);

        let ct1 = RingGSWNTTTest::encrypt(&A, &mu1);
        let ct2 = RingGSWNTTTest::encrypt(&A, &mu2);
        let ct3 = RingGSWNTTTest::encrypt(&A, &mu3);
        let ct4 = RingGSWNTTTest::encrypt(&A, &mu4);

        let ct12 = RingGSWNTTTest::mul_hom(&ct1, &ct2);
        let ct34 = RingGSWNTTTest::mul_hom(&ct3, &ct4);
        let ct1234 = RingGSWNTTTest::mul_hom(&ct12, &ct34);
        // let ct31234 = &ct3 * &ct1234;

        let pt12 = RingGSWNTTTest::decrypt(&s_T, &ct12);
        let pt34 = RingGSWNTTTest::decrypt(&s_T, &ct34);
        let pt1234 = RingGSWNTTTest::decrypt(&s_T, &ct1234);
        // let pt31234 = gsw::decrypt(&s_T, &ct31234);

        assert_eq!(pt12, &mu1 * &mu2);
        assert_eq!(pt34, &mu3 * &mu4);
        assert_eq!(pt1234, &(&(&mu1 * &mu2) * &mu3) * &mu4);
        // assert_eq!(pt31234, &(&(&(&mu1 * &mu2) * &mu3) * &mu4) * &mu3);
    }
}
