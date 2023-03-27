use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::ops::{Add, Mul};

use crate::fhe::fhe::{CiphertextRef, FHEScheme};
use crate::fhe::gadget::{build_gadget, gadget_inverse};
use crate::math::matrix::{identity, stack, Matrix};
use crate::math::rand_sampled::{
    RandDiscreteGaussianSampled, RandUniformSampled, RandZeroOneSampled,
};
use crate::math::ring_elem::RingElement;
use crate::math::utils::ceil_log;
use crate::math::z_n::Z_N;

/*
 * A naive GSW implementation
 *
 * Parameters:
 *   - N_MINUS_1: N-1, since generics cannot be used in const expressions yet. Used only in key generation.
 *   - N,M: matrix dimensions.
 *   - P: plaintext modulus.
 *   - Q: ciphertext modulus.
 *   - G_BASE: base used for the gadget matrix.
 *   - G_LEN: length of the "g" gadget vector, or alternatively log q.
 *   - NOISE_WIDTH_MILLIONTHS: noise width, expressed in millionths to allow for precision past the decimal point (since f64 is not a valid generic parameter).
 *   - NG_LEN: N * G_LEN, since generics cannot be used in const expressions yet. Used only in ciphertext multiplication.
 */

pub struct GSW<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
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

// TODO: Find a way to validate these params at compile time (static_assert / const_guards crate?)

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > FHEScheme<P> for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Ciphertext = Ciphertext<N, M, P, Q, G_BASE, G_LEN>;
    type PublicKey = PublicKey<N, M, P, Q, G_BASE, G_LEN>;
    type SecretKey = SecretKey<N, M, P, Q, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let mut rng = ChaCha20Rng::from_entropy();

        let a_bar: Matrix<N_MINUS_1, M, Z_N<Q>> = Matrix::rand_uniform(&mut rng);
        let s_bar_T: Matrix<1, N_MINUS_1, Z_N<Q>> = Matrix::rand_uniform(&mut rng);
        let e: Matrix<1, M, Z_N<Q>> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);

        let A: Matrix<N, M, Z_N<Q>> = stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
        let mut s_T: Matrix<1, N, Z_N<Q>> = Matrix::zero();
        s_T.copy_into(&(-&s_bar_T), 0, 0);
        s_T[(0, N - 1)] = Z_N::one();
        (PublicKey { A }, SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        let A = &pk.A;

        let mut rng = ChaCha20Rng::from_entropy();
        let R: Matrix<M, M, Z_N<Q>> = Matrix::rand_zero_one(&mut rng);

        let G = build_gadget::<Z_N<Q>, N, M, Q, G_BASE, G_LEN>();

        let mu = Z_N::<Q>::from(u64::from(mu));
        let ct = &(A * &R) + &(&G * &mu);
        Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let q_over_p = Z_N::from(Q / P);
        let g_inv = &gadget_inverse::<Z_N<Q>, N, M, N, G_BASE, G_LEN>(
            &(&identity::<N, Z_N<Q>>() * &q_over_p),
        );

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
                * &gadget_inverse::<Z_N<Q>, N, M, M, G_BASE, G_LEN>(
                    &(&build_gadget::<Z_N<Q>, N, M, Q, G_BASE, G_LEN>() * rhs_q),
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
            ct: &self.ct * &gadget_inverse::<Z_N<Q>, N, M, M, G_BASE, G_LEN>(&rhs.ct),
        }
    }
}

/*
 * GSW params
 */

pub struct Params {
    pub N: usize,
    pub M: usize,
    pub P: u64,
    pub Q: u64,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
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
            { $params.NOISE_WIDTH_MILLIONTHS },
        >
    }
}

/*
 * Pre-defined sets of parameters
 */

pub const GSW_TEST_PARAMS: Params = Params {
    N: 5,
    M: 140,
    P: 31,
    Q: 268369921,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
};

pub type GSWTest = gsw_from_params!(GSW_TEST_PARAMS);

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn keygen_is_correct() {
        let threshold = 4f64 * (GSW_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
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

// Old testing for valid parameters -- for future reference if we want to reimplement these as compile time checks

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     fn verify_int_params<
//         const N: usize,
//         const M: usize,
//         const P: u64,
//         const Q: u64,
//         const G_BASE: u64,
//         const G_LEN: usize,
//         const N_MINUS_1: usize,
//     >(
//         _params: IntParams<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN>,
//     ) {
//         assert_eq!(N_MINUS_1 + 1, N, "N_MINUS_1 not correct");
//         assert!(P <= Q, "plaintext modulus bigger than ciphertext modulus");
//         let mut x = Q;
//         for _ in 0..G_LEN {
//             x /= G_BASE;
//         }
//         assert_eq!(
//             x, 0,
//             "gadget matrix not long enough (expected: G_LEN >= log Q)"
//         );
//         assert!(G_LEN * N <= M, "M >= N log Q not satisfied");
//     }
//
//     #[test]
//     fn test_params_is_correct() {
//         verify_int_params(TEST_PARAMS);
//     }
// }

// pub struct Params {
//     pub poly_len: usize,
//     pub poly_len_log2: usize,
//     pub ntt_tables: Vec<Vec<Vec<u64>>>,
//     pub scratch: Vec<u64>,

//     pub crt_count: usize,
//     pub barrett_cr_0: [u64; MAX_MODULI],
//     pub barrett_cr_1: [u64; MAX_MODULI],
//     pub barrett_cr_0_modulus: u64,
//     pub barrett_cr_1_modulus: u64,
//     pub mod0_inv_mod1: u64,
//     pub mod1_inv_mod0: u64,
//     pub moduli: [u64; MAX_MODULI],
//     pub modulus: u64,
//     pub modulus_log2: u64,
//     pub noise_width: f64,

//     pub n: usize,
//     pub pt_modulus: u64,
//     pub q2_bits: u64,
//     pub t_conv: usize,
//     pub t_exp_left: usize,
//     pub t_exp_right: usize,
//     pub t_gsw: usize,

//     pub expand_queries: bool,
//     pub db_dim_1: usize,
//     pub db_dim_2: usize,
//     pub instances: usize,
//     pub db_item_size: usize,
// }
