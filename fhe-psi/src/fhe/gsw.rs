//! Plain GSW.

use crate::fhe::fhe::*;
use crate::fhe::gsw_utils::*;
use crate::math::matrix::Matrix;
use crate::math::utils::ceil_log;
use crate::math::z_n::Z_N;

/*
 * A naive GSW implementation
 *
 * Parameters:
 *   - N_MINUS_1: N-1, since generics cannot be used in const expressions yet. Used only in key generation.
 *   - N,M: matrix dimensions. It is assumed that `M = N log_{G_BASE} Q`.
 *   - P: plaintext modulus.
 *   - Q: ciphertext modulus.
 *   - G_BASE: base used for the gadget matrix.
 *   - G_LEN: length of the `g` gadget vector, or alternatively `log_{G_BASE} Q`.
 *   - NOISE_WIDTH_MILLIONTHS: noise width, expressed in millionths to allow for precision past the decimal point (since f64 is not a valid generic parameter).
 *
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

#[derive(Clone, Debug)]
pub struct GSWCiphertext<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, Z_N<Q>>,
}

#[derive(Clone, Debug)]
pub struct GSWPublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, Z_N<Q>>,
}

#[derive(Clone, Debug)]
pub struct GSWSecretKey<
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
    > FHEScheme for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
    const NOISE_WIDTH_MILLIONTHS: u64,
    > EncryptionScheme for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Plaintext = Z_N<P>;
    type Ciphertext = GSWCiphertext<N, M, P, Q, G_BASE, G_LEN>;
    type PublicKey = GSWPublicKey<N, M, P, Q, G_BASE, G_LEN>;
    type SecretKey = GSWSecretKey<N, M, P, Q, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let (A, s_T) = gsw_keygen::<N_MINUS_1, N, M, _, NOISE_WIDTH_MILLIONTHS>();
        (Self::PublicKey { A }, Self::SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let ct = gsw_encrypt_pk::<N, M, G_BASE, G_LEN, _>(&pk.A, mu.include_into());
        Self::Ciphertext { ct }
    }

    fn encrypt_sk(sk: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        let ct = gsw_encrypt_sk::<N_MINUS_1, N, M, G_BASE, G_LEN, _, NOISE_WIDTH_MILLIONTHS>(
            &sk.s_T, mu.include_into(),
        );
        Self::Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let pt = gsw_half_decrypt::<N, M, P, Q, G_BASE, G_LEN, _>(s_T, ct);
        pt.round_down_into()
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddHomEncryptionScheme for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulHomEncryptionScheme for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > AddScalarEncryptionScheme<Z_N<P>>
    for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn add_scalar(lhs: &Self::Ciphertext, rhs: &Z_N<P>) -> Self::Ciphertext {
        let rhs_q: Z_N<Q> = rhs.include_into();
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
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > MulScalarEncryptionScheme<Z_N<P>>
    for GSW<N_MINUS_1, N, M, P, Q, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    fn mul_scalar(lhs: &Self::Ciphertext, rhs: &Z_N<P>) -> Self::Ciphertext {
        let rhs_q: Z_N<Q> = rhs.include_into();
        Self::Ciphertext {
            ct: scalar_ciphertext_mul::<N, M, G_BASE, G_LEN, _>(&lhs.ct, &rhs_q),
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
            assert!(
                (e[(0, i)].norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = GSWTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = GSWTest::encrypt(&A, &mu);
            let pt = GSWTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn encryption_sk_is_correct() {
        let (A, s_T) = GSWTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = GSWTest::encrypt_sk(&s_T, &mu);
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
                let ct1 = GSWTest::encrypt(&A, &mu1);
                let ct2 = GSWTest::encrypt(&A, &mu2);

                let pt_add_ct = GSWTest::decrypt(&s_T, &(GSWTest::add_hom(&ct1, &ct2)));
                let pt_mul_ct = GSWTest::decrypt(&s_T, &(GSWTest::mul_hom(&ct1, &ct2)));
                let pt_mul_scalar = GSWTest::decrypt(&s_T, &(GSWTest::mul_scalar(&ct1, &mu2)));

                assert_eq!(pt_add_ct, mu1 + mu2, "ciphertext addition failed");
                assert_eq!(pt_mul_ct, mu1 * mu2, "ciphertext multiplication failed");
                assert_eq!(pt_mul_scalar, mu1 * mu2, "multiplication by scalar failed");
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

        let ct1 = GSWTest::encrypt(&A, &mu1);
        let ct2 = GSWTest::encrypt(&A, &mu2);
        let ct3 = GSWTest::encrypt(&A, &mu3);
        let ct4 = GSWTest::encrypt(&A, &mu4);

        let ct12 = GSWTest::mul_hom(&ct1, &ct2);
        let ct34 = GSWTest::mul_hom(&ct3, &ct4);
        let ct1234 = GSWTest::mul_hom(&ct12, &ct34);
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

// Params struct from spiral

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
