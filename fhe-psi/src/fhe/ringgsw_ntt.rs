use crate::fhe::fhe::*;
use crate::fhe::gadget::*;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::RingElement;
use crate::math::utils::ceil_log;
use crate::math::z_n::Z_N;
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use crate::math::z_n_cyclo_ntt::Z_N_CycloNTT;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::ops::{Add, Mul};

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
pub struct CiphertextNTT<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, Z_N_CycloNTT<D, Q, W>>,
}

#[derive(Clone, Debug)]
pub struct PublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, Z_N_CycloNTT<D, Q, W>>,
}

#[derive(Clone, Debug)]
pub struct SecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const D: usize,
    const W: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, Z_N_CycloNTT<D, Q, W>>,
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
    > FHEScheme<P>
    for RingGSWNTT<N_MINUS_1, N, M, P, Q, D, W, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Ciphertext = CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>;
    type PublicKey = PublicKey<N, M, P, Q, D, W, G_BASE, G_LEN>;
    type SecretKey = SecretKey<N, M, P, Q, D, W, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let mut rng = ChaCha20Rng::from_entropy();

        let a_bar: Matrix<N_MINUS_1, M, Z_N_CycloNTT<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        let s_bar_T: Matrix<1, N_MINUS_1, Z_N_CycloNTT<D, Q, W>> = Matrix::rand_uniform(&mut rng);
        let e: Matrix<1, M, Z_N_CycloNTT<D, Q, W>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);

        let A: Matrix<N, M, Z_N_CycloNTT<D, Q, W>> =
            Matrix::stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
        let mut s_T: Matrix<1, N, Z_N_CycloNTT<D, Q, W>> = Matrix::zero();
        s_T.copy_into(&(-&s_bar_T), 0, 0);
        s_T[(0, N - 1)] = Z_N_CycloNTT::one();
        (PublicKey { A }, SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        let A = &pk.A;

        let mut rng = ChaCha20Rng::from_entropy();
        let R: Matrix<M, M, Z_N_CycloNTT<D, Q, W>> = Matrix::rand_zero_one(&mut rng);

        let G = build_gadget::<Z_N_CycloNTT<D, Q, W>, N, M, Q, G_BASE, G_LEN>();

        let mu = Z_N_CycloNTT::<D, Q, W>::from(u64::from(mu));
        let ct = &(A * &R) + &(&G * &mu);
        CiphertextNTT { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let q_over_p = Z_N_CycloNTT::from(Q / P);
        let g_inv = &gadget_inverse::<Z_N_CycloNTT<D, Q, W>, N, M, N, G_BASE, G_LEN>(
            &(&Matrix::<N, N, Z_N_CycloNTT<D, Q, W>>::identity() * &q_over_p),
        );

        let pt = &(&(s_T * ct) * g_inv)[(0, N - 1)];
        // TODO support arbitrary messages, not just constants
        let pt = Z_N_CycloRaw::from(pt.clone())[0];
        let floored = u64::from(pt) * P * 2 / Q;
        Z_N::from((floored + 1) / 2)
    }
}

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const D: usize,
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
    > Add for &'a CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>
{
    type Output = CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>;

    fn add(self, rhs: Self) -> Self::Output {
        CiphertextNTT {
            ct: &self.ct + &rhs.ct,
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
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul for &'a CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>
{
    type Output = CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>;

    fn mul(self, rhs: Self) -> Self::Output {
        CiphertextNTT {
            ct: &self.ct
                * &gadget_inverse::<Z_N_CycloNTT<D, Q, W>, N, M, M, G_BASE, G_LEN>(&rhs.ct),
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
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul<Z_N<P>> for &'a CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>
{
    type Output = CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>;

    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        let rhs_q = Z_N::from(u64::from(rhs));
        let mut G_rhs: Matrix<N, M, Z_N_CycloRaw<D, Q>> = build_gadget::<Z_N_CycloRaw<D, Q>, N, M, Q, G_BASE, G_LEN>();
        for i in 0..N {
            for j in 0..M {
                G_rhs[(i, j)] *= rhs_q;
            }
        }

        let G_inv_G_rhs_raw: Matrix<M, M, Z_N_CycloRaw<D, Q>> = gadget_inverse::<Z_N_CycloRaw<D, Q>, N, M, M, G_BASE, G_LEN>(&G_rhs);

        let mut G_inv_G_rhs_ntt: Matrix<M, M, Z_N_CycloNTT<D, Q, W>> = Matrix::zero();
        for i in 0..M {
            for j in 0..M {
                G_inv_G_rhs_ntt[(i, j)] = (&G_inv_G_rhs_raw[(i, j)]).into();
            }
        }

        CiphertextNTT {
            ct: &self.ct * &G_inv_G_rhs_ntt,
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
        const W: u64,
        const G_BASE: u64,
        const G_LEN: usize,
    > CiphertextRef<P, CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>>
    for &'a CiphertextNTT<N, M, P, Q, D, W, G_BASE, G_LEN>
{
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
    D: 2048,
    W: 63703579,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 1,
};

// For parameter selection: fix Q, then choose smallest N*D for appropriate security.

pub type RingGSWNTTTest = ring_gsw_ntt_from_params!(RING_GSW_NTT_TEST_PARAMS);
pub type RingGSWNTTTestMedium = ring_gsw_ntt_from_params!(RING_GSW_NTT_TEST_MEDIUM_PARAMS);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn keygen_is_correct() {
        let threshold =
            4f64 * (RING_GSW_NTT_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let (A, s_T) = RingGSWNTTTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..RING_GSW_NTT_TEST_PARAMS.M {
            assert!(
                (Z_N_CycloRaw::from(e[(0, i)].clone()).norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = RingGSWNTTTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = RingGSWNTTTest::encrypt(&A, mu);
            let pt = RingGSWNTTTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = RingGSWNTTTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = Z_N::from(i);
                let mu2 = Z_N::from(j);
                let ct1 = RingGSWNTTTest::encrypt(&A, mu1);
                let ct2 = RingGSWNTTTest::encrypt(&A, mu2);
                let ct_add_ct = RingGSWNTTTest::decrypt(&s_T, &(&ct1 + &ct2));
                let ct_mul_ct = RingGSWNTTTest::decrypt(&s_T, &(&ct1 * &ct2));
                let ct_mul_scalar = RingGSWNTTTest::decrypt(&s_T, &(&ct1 * mu2));
                assert_eq!(ct_add_ct, &mu1 + &mu2, "ciphertext addition failed");
                assert_eq!(ct_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
                assert_eq!(
                    ct_mul_scalar,
                    &mu1 * &mu2,
                    "multiplication by scalar failed"
                );
            }
        }
    }

    #[test]
    fn homomorphism_mul_multiple_correct() {
        let (A, s_T) = RingGSWNTTTest::keygen();
        let mu1 = Z_N::from(5_u64);
        let mu2 = Z_N::from(12_u64);
        let mu3 = Z_N::from(6_u64);
        let mu4 = Z_N::from(18_u64);

        let ct1 = RingGSWNTTTest::encrypt(&A, mu1);
        let ct2 = RingGSWNTTTest::encrypt(&A, mu2);
        let ct3 = RingGSWNTTTest::encrypt(&A, mu3);
        let ct4 = RingGSWNTTTest::encrypt(&A, mu4);

        let ct12 = &ct1 * &ct2;
        let ct34 = &ct3 * &ct4;
        let ct1234 = &ct12 * &ct34;
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
