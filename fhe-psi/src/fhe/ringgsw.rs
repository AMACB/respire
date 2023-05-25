//! Standard GSW over a ring of polynomials.

use crate::fhe::fhe::*;
use crate::fhe::gsw_utils::*;
use crate::math::matrix::Matrix;
use crate::math::utils::ceil_log;
use crate::math::z_n::Z_N;
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use std::ops::{Add, Mul};

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

pub struct RingGSW<
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
pub struct CiphertextRaw<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, Z_N_CycloRaw<D, Q>>,
}

#[derive(Clone, Debug)]
pub struct PublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, Z_N_CycloRaw<D, Q>>,
}

#[derive(Clone, Debug)]
pub struct SecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, Z_N_CycloRaw<D, Q>>,
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
    > FHEScheme<P> for RingGSW<N_MINUS_1, N, M, P, Q, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Ciphertext = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;
    type PublicKey = PublicKey<N, M, P, Q, D, G_BASE, G_LEN>;
    type SecretKey = SecretKey<N, M, P, Q, D, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let (A, s_T) = gsw_keygen::<N_MINUS_1, N, M, _, NOISE_WIDTH_MILLIONTHS>();
        (PublicKey { A }, SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        let mu = Z_N_CycloRaw::<D, Q>::from(u64::from(mu));
        let ct = gsw_encrypt_pk::<N, M, G_BASE, G_LEN, _>(&pk.A, mu);
        CiphertextRaw { ct }
    }
    fn encrypt_sk(sk: &Self::SecretKey, mu: Z_N<P>) -> Self::Ciphertext {
        let mu = Z_N_CycloRaw::<D, Q>::from(u64::from(mu));
        let ct = gsw_encrypt_sk::<N_MINUS_1, N, M, G_BASE, G_LEN, _, NOISE_WIDTH_MILLIONTHS>(
            &sk.s_T, mu,
        );
        CiphertextRaw { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let pt = gsw_half_decrypt::<N, M, P, Q, G_BASE, G_LEN, _>(s_T, ct);
        // TODO support arbitrary messages, not just constants
        gsw_round::<P, Q, _>(pt[0])
    }
}

/*
 * GSW homomorphic addition / multiplication
 */

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > CiphertextRef<P, CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>>
    for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
}

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Add<&Z_N<P>> for &CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
    type Output = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;
    fn add(self, rhs: &Z_N<P>) -> Self::Output {
        let rhs_q = &Z_N_CycloRaw::<D, Q>::from(u64::from(*rhs));
        CiphertextRaw {
            ct: scalar_ciphertext_add::<N, M, G_BASE, G_LEN, _>(&self.ct, &rhs_q),
        }
    }
}

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul<Z_N<P>> for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
    type Output = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;

    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        let rhs_q = &Z_N_CycloRaw::<D, Q>::from(u64::from(rhs));
        CiphertextRaw {
            ct: scalar_ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&self.ct, &rhs_q),
        }
    }
}

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Add for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
    type Output = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;

    fn add(self, rhs: Self) -> Self::Output {
        CiphertextRaw {
            ct: ciphertext_add::<N, M, G_BASE, G_LEN, _>(&self.ct, &rhs.ct),
        }
    }
}

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
    type Output = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;

    fn mul(self, rhs: Self) -> Self::Output {
        CiphertextRaw {
            ct: ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&self.ct, &rhs.ct),
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

macro_rules! ring_gsw_from_params {
    ($params:expr) => {
        RingGSW<
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

pub const RING_GSW_TEST_PARAMS: Params = Params {
    N: 2,
    M: 56,
    P: 31,
    Q: 268369921,
    D: 4,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
};

pub const RING_GSW_TEST_MEDIUM_PARAMS: Params = Params {
    N: 2,
    M: 56,
    P: 31,
    Q: 268369921,
    D: 2048,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 1,
};

pub type RingGSWTest = ring_gsw_from_params!(RING_GSW_TEST_PARAMS);
pub type RingGSWTestMedium = ring_gsw_from_params!(RING_GSW_TEST_MEDIUM_PARAMS);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn keygen_is_correct() {
        let threshold = 4f64 * (RING_GSW_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let (A, s_T) = RingGSWTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..RING_GSW_TEST_PARAMS.M {
            assert!(
                (e[(0, i)].norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = RingGSWTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = RingGSWTest::encrypt(&A, mu);
            let pt = RingGSWTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = RingGSWTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = Z_N::from(i);
                let mu2 = Z_N::from(j);
                let ct1 = RingGSWTest::encrypt(&A, mu1);
                let ct2 = RingGSWTest::encrypt(&A, mu2);
                // let pt_add = GSWTest::decrypt(&s_T, &(&ct1 + mu2));
                let pt_mul = RingGSWTest::decrypt(&s_T, &(&ct1 * mu2));
                let pt_add_ct = RingGSWTest::decrypt(&s_T, &(&ct1 + &ct2));
                let pt_mul_ct = RingGSWTest::decrypt(&s_T, &(&ct1 * &ct2));
                // assert_eq!(pt_add, &mu1 + &mu2, "addition by scalar failed");
                assert_eq!(pt_add_ct, &mu1 + &mu2, "ciphertext addition failed");

                assert_eq!(pt_mul, &mu1 * &mu2, "multiplication by scalar failed");
                assert_eq!(pt_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
            }
        }
    }

    #[test]
    fn homomorphism_mul_multiple_correct() {
        let (A, s_T) = RingGSWTest::keygen();
        let mu1 = Z_N::from(5_u64);
        let mu2 = Z_N::from(12_u64);
        let mu3 = Z_N::from(6_u64);
        let mu4 = Z_N::from(18_u64);

        let ct1 = RingGSWTest::encrypt(&A, mu1);
        let ct2 = RingGSWTest::encrypt(&A, mu2);
        let ct3 = RingGSWTest::encrypt(&A, mu3);
        let ct4 = RingGSWTest::encrypt(&A, mu4);

        let ct12 = &ct1 * &ct2;
        let ct34 = &ct3 * &ct4;
        let ct1234 = &ct12 * &ct34;
        // let ct31234 = &ct3 * &ct1234;

        let pt12 = RingGSWTest::decrypt(&s_T, &ct12);
        let pt34 = RingGSWTest::decrypt(&s_T, &ct34);
        let pt1234 = RingGSWTest::decrypt(&s_T, &ct1234);
        // let pt31234 = gsw::decrypt(&s_T, &ct31234);

        assert_eq!(pt12, &mu1 * &mu2);
        assert_eq!(pt34, &mu3 * &mu4);
        assert_eq!(pt1234, &(&(&mu1 * &mu2) * &mu3) * &mu4);
        // assert_eq!(pt31234, &(&(&(&mu1 * &mu2) * &mu3) * &mu4) * &mu3);
    }
}
