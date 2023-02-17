use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::ops::{Add, Mul};

use crate::discrete_gaussian::DiscreteGaussian;
use crate::fhe::{CiphertextRef, FHEScheme};
use crate::ring_elem::RingElement;
use crate::{gadget::*, matrix::*, z_n::*};

pub struct GSW<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONS: u64,
> {}

#[derive(Debug)]
pub struct Ciphertext<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, Z_N<Q>>,
}

#[derive(Debug)]
pub struct PublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, Z_N<Q>>,
}

#[derive(Debug)]
pub struct SecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, Z_N<Q>>,
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONS: u64,
    > GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONS>
{
}

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONS: u64,
    > FHEScheme<P> for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONS>
{
    type Ciphertext = Ciphertext<N, M, P, Q, G_BASE, G_LEN>;
    type PublicKey = PublicKey<N, M, P, Q, G_BASE, G_LEN>;
    type SecretKey = SecretKey<N, M, P, Q, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let dg = DiscreteGaussian::init(NOISE_WIDTH_MILLIONS as f64 / 1_000_000_f64);
        let mut rng = ChaCha20Rng::from_entropy();

        let a_bar: Matrix<N_MINUS_1, M, Z_N<Q>> = Matrix::random_rng(&mut rng);
        let s_bar_T: Matrix<1, N_MINUS_1, Z_N<Q>> = Matrix::random_rng(&mut rng);
        let e: Matrix<1, M, Z_N<Q>> = dg.sample_int_matrix(&mut rng);

        let A: Matrix<N, M, Z_N<Q>> = stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
        let mut s_T: Matrix<1, N, Z_N<Q>> = Matrix::zero();
        s_T.copy_into(&(-&s_bar_T), 0, 0);
        s_T[(0, N - 1)] = Z_N::one();
        (PublicKey { A }, SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        let A = &pk.A;

        let mut rng = ChaCha20Rng::from_entropy();

        let mut R: Matrix<M, M, Z_N<Q>> = Matrix::zero();
        for i in 0..M {
            for j in 0..M {
                if rng.gen_bool(0.5) {
                    R[(i, j)] = Z_N::one();
                }
            }
        }

        let G = build_gadget::<N, M, Q, G_BASE, G_LEN>();

        let mu = Z_N::<Q>::from(u64::from(mu));
        let ct = &(A * &R) + &(&G * &mu);
        Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let q_over_p = Z_N::from(Q / P);
        let g_inv =
            &gadget_inverse::<N, M, N, Q, G_BASE, G_LEN>(&(&identity::<N, Z_N<Q>>() * &q_over_p));

        let pt = &(&(s_T * ct) * g_inv)[(0, N - 1)];
        let floored = u64::from(*pt) * P * 2 / Q;
        Z_N::from((floored + 1) / 2)
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
        const G_BASE: u64,
        const G_LEN: usize,
    > CiphertextRef<P, Ciphertext<N, M, P, Q, G_BASE, G_LEN>>
    for &'a Ciphertext<N, M, P, Q, G_BASE, G_LEN>
{
}

// impl<
//         const N: usize,
//         const M: usize,
//         const P: u64,
//         const Q: u64,
//         const G_BASE: u64,
//         const G_LEN: usize,
//     > Add<&Z_N<P>> for &Ciphertext<N, M, P, Q, G_BASE, G_LEN>
// {
//     type Output = Ciphertext<N, M, P, Q, G_BASE, G_LEN>;
//     fn add(self, rhs: &Z_N<P>) -> Self::Output {
//         let rhs_q = &Z_N::<Q>::from(u64::from(*rhs));
//         Ciphertext {
//             ct: &self.ct + &(&build_gadget::<N, M, Q, G_BASE, G_LEN>() * rhs_q),
//         }
//     }
// }

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul<Z_N<P>> for &'a Ciphertext<N, M, P, Q, G_BASE, G_LEN>
{
    type Output = Ciphertext<N, M, P, Q, G_BASE, G_LEN>;
    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        let rhs_q = &Z_N::<Q>::from(u64::from(rhs));
        Ciphertext {
            ct: &self.ct
                * &gadget_inverse::<N, M, M, Q, G_BASE, G_LEN>(
                    &(&build_gadget::<N, M, Q, G_BASE, G_LEN>() * rhs_q),
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
        const G_BASE: u64,
        const G_LEN: usize,
    > Add for &'a Ciphertext<N, M, P, Q, G_BASE, G_LEN>
{
    type Output = Ciphertext<N, M, P, Q, G_BASE, G_LEN>;
    fn add(self, rhs: Self) -> Self::Output {
        Ciphertext {
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
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul for &'a Ciphertext<N, M, P, Q, G_BASE, G_LEN>
{
    type Output = Ciphertext<N, M, P, Q, G_BASE, G_LEN>;
    fn mul(self, rhs: Self) -> Self::Output {
        Ciphertext {
            ct: &self.ct * &gadget_inverse::<N, M, M, Q, G_BASE, G_LEN>(&rhs.ct),
        }
    }
}

/*
 * GSW Params & compile-time verification
 */

pub const fn ceil_log(base: u64, x: u64) -> usize {
    let mut e = 0;
    let mut y = 1;

    while y < x {
        y *= base;
        e += 1;
    }

    e
}

pub struct Params {
    pub N: usize,
    pub M: usize,
    pub P: u64,
    pub Q: u64,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONS: u64,
}

macro_rules! gsw_from_params {
    ($params:expr) => {
        GSW<
            { $params.N - 1 },
            { $params.N },
            { $params.M },
            { $params.P },
            { $params.Q },
            { $params.G_BASE },
            { ceil_log($params.G_BASE, $params.Q) },
            { $params.NOISE_WIDTH_MILLIONS },
        >
    }
}

pub const GSW_TEST_PARAMS: Params = Params {
    N: 5,
    M: 140,
    P: 41,
    Q: 268369921,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONS: 6_400_000,
};

pub type GSWTest = gsw_from_params!(GSW_TEST_PARAMS);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn keygen_is_correct() {
        let threshold = 4f64 * (GSW_TEST_PARAMS.NOISE_WIDTH_MILLIONS as f64 / 1_000_000_f64);
        let (A, s_T) = GSWTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..GSW_TEST_PARAMS.M {
            // abs(e[i]) < threshold
            let ei_pos: u64 = e[(0, i)].into();
            let ei_neg: u64 = (-e[(0, i)]).into();
            assert!(
                (ei_pos as f64) < threshold || (ei_neg as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = GSWTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = GSWTest::encrypt(&A, mu);
            let pt = GSWTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = GSWTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = Z_N::from(i);
                let mu2 = Z_N::from(j);
                let ct1 = GSWTest::encrypt(&A, mu1);
                let ct2 = GSWTest::encrypt(&A, mu2);
                // let pt_add = GSWTest::decrypt(&s_T, &(&ct1 + mu2));
                let pt_mul = GSWTest::decrypt(&s_T, &(&ct1 * mu2));
                let pt_add_ct = GSWTest::decrypt(&s_T, &(&ct1 + &ct2));
                let pt_mul_ct = GSWTest::decrypt(&s_T, &(&ct1 * &ct2));
                // assert_eq!(pt_add, &mu1 + &mu2, "addition by scalar failed");
                assert_eq!(pt_add_ct, &mu1 + &mu2, "ciphertext addition failed");

                assert_eq!(pt_mul, &mu1 * &mu2, "multiplication by scalar failed");
                assert_eq!(pt_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
            }
        }
    }

    #[test]
    fn homomorphism_mul_multiple_correct() {
        let (A, s_T) = GSWTest::keygen();
        let mu1 = Z_N::from(5_u64);
        let mu2 = Z_N::from(12_u64);
        let mu3 = Z_N::from(6_u64);
        let mu4 = Z_N::from(18_u64);

        let ct1 = GSWTest::encrypt(&A, mu1);
        let ct2 = GSWTest::encrypt(&A, mu2);
        let ct3 = GSWTest::encrypt(&A, mu3);
        let ct4 = GSWTest::encrypt(&A, mu4);

        let ct12 = &ct1 * &ct2;
        let ct34 = &ct3 * &ct4;
        let ct1234 = &ct12 * &ct34;
        // let ct31234 = &ct3 * &ct1234;

        let pt12 = GSWTest::decrypt(&s_T, &ct12);
        let pt34 = GSWTest::decrypt(&s_T, &ct34);
        let pt1234 = GSWTest::decrypt(&s_T, &ct1234);
        // let pt31234 = gsw::decrypt(&s_T, &ct31234);

        assert_eq!(pt12, &mu1 * &mu2);
        assert_eq!(pt34, &mu3 * &mu4);
        assert_eq!(pt1234, &(&(&mu1 * &mu2) * &mu3) * &mu4);
        // assert_eq!(pt31234, &(&(&(&mu1 * &mu2) * &mu3) * &mu4) * &mu3);
    }
}
