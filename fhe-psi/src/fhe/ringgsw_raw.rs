//! Standard GSW over a ring of polynomials.

use crate::fhe::fhe::*;
use crate::fhe::gsw_utils::*;
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::matrix::Matrix;
use crate::math::utils::ceil_log;

/*
 * A Ring GSW implementation
 *
 * Parameters:
 *   - N_MINUS_1: N-1, since generics cannot be used in const expressions yet. Used only in key generation.
 *   - N,M: matrix dimensions. It is assumed that `M = N log_{G_BASE} Q`.
 *   - P: plaintext modulus.
 *   - Q: ciphertext modulus.
 *   - D: degree of polynomials. See `Z_N_CycloRaw`.
 *   - G_BASE: base used for the gadget matrix.
 *   - G_LEN: length of the `g` gadget vector, or alternatively `log_{G_BASE} Q`.
 *   - NOISE_WIDTH_MILLIONTHS: noise width, expressed in millionths to allow for precision past the decimal point (since f64 is not a valid generic parameter).
 *
 */

pub struct RingGSWRaw<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

#[derive(Clone, Debug)]
pub struct RingGSWRawCiphertext<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, IntModCyclo<D, Q>>,
}

#[derive(Clone, Debug)]
pub struct RingGSWRawPublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, IntModCyclo<D, Q>>,
}

#[derive(Clone, Debug)]
pub struct RingGSWRawSecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, IntModCyclo<D, Q>>,
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > FHEScheme for RingGSWRaw<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > EncryptionScheme
    for RingGSWRaw<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = IntMod<P>;
    type Ciphertext = RingGSWRawCiphertext<N, M, P, Q, D, G_BASE, G_LEN>;
    type PublicKey = RingGSWRawPublicKey<N, M, P, Q, D, G_BASE, G_LEN>;
    type SecretKey = RingGSWRawSecretKey<N, M, P, Q, D, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let (A, s_T) = gsw_keygen::<N_MINUS_1, N, M, _, NOISE_WIDTH_MILLIONTHS>();
        (Self::PublicKey { A }, Self::SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mu = IntModCyclo::from(u64::from(mu.clone()));
        let ct = gsw_encrypt_pk::<N, M, G_BASE, G_LEN, _>(&pk.A, mu);
        Self::Ciphertext { ct }
    }

    fn encrypt_sk(sk: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let mu = IntModCyclo::from(u64::from(mu.clone()));
        let ct = gsw_encrypt_sk::<N_MINUS_1, N, M, G_BASE, G_LEN, _, NOISE_WIDTH_MILLIONTHS>(
            &sk.s_T, mu,
        );
        Self::Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> IntMod<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let pt = gsw_half_decrypt::<N, M, P, Q, G_BASE, G_LEN, _>(s_T, ct);
        // TODO support arbitrary messages, not just constants
        pt[0].round_down_into()
    }
}

/*
 * GSW homomorphic addition / multiplication
 */

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddHomEncryptionScheme
    for RingGSWRaw<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulHomEncryptionScheme
    for RingGSWRaw<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddScalarEncryptionScheme<IntMod<P>>
    for RingGSWRaw<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn add_scalar(lhs: &Self::Ciphertext, rhs: &IntMod<P>) -> Self::Ciphertext {
        let rhs_q = IntModCyclo::<D, Q>::from(u64::from(*rhs));
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulScalarEncryptionScheme<IntMod<P>>
    for RingGSWRaw<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn mul_scalar(lhs: &Self::Ciphertext, rhs: &IntMod<P>) -> Self::Ciphertext {
        let rhs_q = IntModCyclo::<D, Q>::from(u64::from(*rhs));
        Self::Ciphertext {
            ct: scalar_ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs_q),
        }
    }
}

pub struct Params {
    pub N: usize,
    pub M: usize,
    pub P: u64,
    pub Q: u64,
    pub D: usize,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

macro_rules! ring_gsw_raw_from_params {
    ($params:expr) => {
        RingGSWRaw<
            { $params.N - 1 },
            { $params.N },
            { $params.M },
            { $params.P },
            { $params.Q },
            { $params.D },
            { $params.G_BASE },
            { ceil_log($params.G_BASE, $params.Q) },
            { $params.NOISE_WIDTH_MILLIONTHS },
        >
    }
}

pub const RING_GSW_RAW_TEST_PARAMS: Params = Params {
    N: 2,
    M: 56,
    P: 31,
    Q: 268369921,
    D: 4,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
};

pub const RING_GSW_RAW_TEST_MEDIUM_PARAMS: Params = Params {
    N: 2,
    M: 56,
    P: 31,
    Q: 268369921,
    D: 2048,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 1,
};

pub type RingGSWRawTest = ring_gsw_raw_from_params!(RING_GSW_RAW_TEST_PARAMS);
pub type RingGSWRawTestMedium = ring_gsw_raw_from_params!(RING_GSW_RAW_TEST_MEDIUM_PARAMS);

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::ring_elem::NormedRingElement;

    #[test]
    fn keygen_is_correct() {
        let threshold =
            4f64 * (RING_GSW_RAW_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let (A, s_T) = RingGSWRawTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..RING_GSW_RAW_TEST_PARAMS.M {
            assert!(
                (e[(0, i)].norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = RingGSWRawTest::keygen();
        for mu in 0_u64..10_u64 {
            let ct = RingGSWRawTest::encrypt(&A, &mu.into());
            let pt = RingGSWRawTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu.into(), "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = RingGSWRawTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = i.into();
                let mu2 = j.into();
                let ct1 = RingGSWRawTest::encrypt(&A, &mu1);
                let ct2 = RingGSWRawTest::encrypt(&A, &mu2);
                let pt_add_ct = RingGSWRawTest::decrypt(&s_T, &RingGSWRawTest::add_hom(&ct1, &ct2));
                let pt_mul_ct = RingGSWRawTest::decrypt(&s_T, &RingGSWRawTest::mul_hom(&ct1, &ct2));
                let pt_mul_scalar =
                    RingGSWRawTest::decrypt(&s_T, &RingGSWRawTest::mul_scalar(&ct1, &j.into()));
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
        let (A, s_T) = RingGSWRawTest::keygen();
        let mu1 = IntMod::from(5_u64);
        let mu2 = IntMod::from(12_u64);
        let mu3 = IntMod::from(6_u64);
        let mu4 = IntMod::from(18_u64);

        let ct1 = RingGSWRawTest::encrypt(&A, &mu1);
        let ct2 = RingGSWRawTest::encrypt(&A, &mu2);
        let ct3 = RingGSWRawTest::encrypt(&A, &mu3);
        let ct4 = RingGSWRawTest::encrypt(&A, &mu4);

        let ct12 = RingGSWRawTest::mul_hom(&ct1, &ct2);
        let ct34 = RingGSWRawTest::mul_hom(&ct3, &ct4);
        let ct1234 = RingGSWRawTest::mul_hom(&ct12, &ct34);
        // let ct31234 = &ct3 * &ct1234;

        let pt12 = RingGSWRawTest::decrypt(&s_T, &ct12);
        let pt34 = RingGSWRawTest::decrypt(&s_T, &ct34);
        let pt1234 = RingGSWRawTest::decrypt(&s_T, &ct1234);
        // let pt31234 = gsw::decrypt(&s_T, &ct31234);

        assert_eq!(pt12, &mu1 * &mu2);
        assert_eq!(pt34, &mu3 * &mu4);
        assert_eq!(pt1234, &(&(&mu1 * &mu2) * &mu3) * &mu4);
        // assert_eq!(pt31234, &(&(&(&mu1 * &mu2) * &mu3) * &mu4) * &mu3);
    }
}
