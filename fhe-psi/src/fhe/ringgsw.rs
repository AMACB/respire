use crate::fhe::fhe::*;
use crate::fhe::gadget::*;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::RingElement;
use crate::math::utils::ceil_log;
use crate::math::z_n::Z_N;
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::ops::{Add, Mul};

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
        let mut rng = ChaCha20Rng::from_entropy();

        let a_bar: Matrix<N_MINUS_1, M, Z_N_CycloRaw<D, Q>> = Matrix::rand_uniform(&mut rng);
        let s_bar_T: Matrix<1, N_MINUS_1, Z_N_CycloRaw<D, Q>> = Matrix::rand_uniform(&mut rng);
        let e: Matrix<1, M, Z_N_CycloRaw<D, Q>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);

        let A: Matrix<N, M, Z_N_CycloRaw<D, Q>> =
            Matrix::stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
        let mut s_T: Matrix<1, N, Z_N_CycloRaw<D, Q>> = Matrix::zero();
        s_T.copy_into(&(-&s_bar_T), 0, 0);
        s_T[(0, N - 1)] = Z_N_CycloRaw::one();
        (PublicKey { A }, SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        let A = &pk.A;

        let mut rng = ChaCha20Rng::from_entropy();
        let R: Matrix<M, M, Z_N_CycloRaw<D, Q>> = Matrix::rand_zero_one(&mut rng);

        let G = build_gadget::<Z_N_CycloRaw<D, Q>, N, M, Q, G_BASE, G_LEN>();

        let mu = Z_N_CycloRaw::<D, Q>::from(u64::from(mu));
        let ct = &(A * &R) + &(&G * &mu);
        CiphertextRaw { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let q_over_p = Z_N_CycloRaw::from(Q / P);
        let g_inv = &gadget_inverse::<Z_N_CycloRaw<D, Q>, N, M, N, G_BASE, G_LEN>(
            &(&Matrix::<N, N, Z_N_CycloRaw<D, Q>>::identity() * &q_over_p),
        );

        let pt = &(&(s_T * ct) * g_inv)[(0, N - 1)];
        // TODO support arbitrary messages, not just constants
        let pt = pt[0];
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
        const G_BASE: u64,
        const G_LEN: usize,
    > Add for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
    type Output = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;

    fn add(self, rhs: Self) -> Self::Output {
        CiphertextRaw {
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
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
    type Output = CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>;

    fn mul(self, rhs: Self) -> Self::Output {
        CiphertextRaw {
            ct: &self.ct * &gadget_inverse::<Z_N_CycloRaw<D, Q>, N, M, M, G_BASE, G_LEN>(&rhs.ct),
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
            ct: &self.ct
                * &gadget_inverse::<Z_N_CycloRaw<D, Q>, N, M, M, G_BASE, G_LEN>(
                    &(&build_gadget::<Z_N_CycloRaw<D, Q>, N, M, Q, G_BASE, G_LEN>() * rhs_q),
                ),
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
    > CiphertextRef<P, CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>>
    for &'a CiphertextRaw<N, M, P, Q, D, G_BASE, G_LEN>
{
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
    P: 3571,
    Q: 268369921,
    D: 256,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
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
