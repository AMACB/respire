use crate::fhe::fhe::*;
use crate::fhe::gadget::*;
use crate::fhe::gsw_utils::*;
use crate::math::matrix::Matrix;
use crate::math::utils::ceil_log;
use crate::math::z_n::Z_N;
use crate::math::z_n_crt::Z_N_CRT;
use crate::math::z_n_cyclo_crt_ntt::Z_N_CycloNTT_CRT;
use std::ops::{Add, Mul};

pub struct RingGSW_NTT_CRT<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const W1: u64,
    const W2: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
> {}

#[derive(Clone, Debug)]
pub struct Ciphertext<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const W1: u64,
    const W2: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    ct: Matrix<N, M, Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>>,
}

#[derive(Clone, Debug)]
pub struct PublicKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const W1: u64,
    const W2: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    A: Matrix<N, M, Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>>,
}

#[derive(Clone, Debug)]
pub struct SecretKey<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const Q1: u64,
    const Q2: u64,
    const W1: u64,
    const W2: u64,
    const D: usize,
    const G_BASE: u64,
    const G_LEN: usize,
> {
    s_T: Matrix<1, N, Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>>,
}

// TODO: Find a way to validate these params at compile time (static_assert / const_guards crate?)

impl<
        const N_MINUS_1: usize,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const W1: u64,
        const W2: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
    > FHEScheme<P>
    for RingGSW_NTT_CRT<N_MINUS_1, N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN, NOISE_WIDTH_MILLIONTHS>
{
    type Ciphertext = Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;
    type PublicKey = PublicKey<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;
    type SecretKey = SecretKey<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        let (A, s_T) = gsw_keygen::<N_MINUS_1, N, M, Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>, NOISE_WIDTH_MILLIONTHS>();
        (PublicKey { A }, SecretKey { s_T })
    }

    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        let mu = Z_N_CycloNTT_CRT::<D, Q1, Q2, W1, W2>::from(u64::from(mu));
        let ct = gsw_encrypt_pk::<N, M, G_BASE, G_LEN, Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>>(&pk.A, mu);
        Ciphertext { ct }
    }

    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let pt = (&gsw_half_decrypt::<N, M, P, Q, G_BASE, G_LEN, Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>>(s_T, ct)).into();
        gsw_round::<P, Q, Z_N_CRT<Q1, Q2>>(pt)
    }
}

/*
 * RingGSW_NTT_CRT homomorphic addition / multiplication
 */

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const W1: u64,
        const W2: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > CiphertextRef<P, Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>>
    for &'a Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>
{
}

// impl<
//         const N: usize,
//         const M: usize,
//         const P: u64,
//         const Q: u64,
//         const Q1: u64,
//         const Q2: u64,
//         const W1: u64,
//         const W2: u64,
//         const D: usize,
//         const G_BASE: u64,
//         const G_LEN: usize,
//     > Add<&Z_N<P>> for &Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>
// {
//     type Output = Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;
//     fn add(self, rhs: &Z_N<P>) -> Self::Output {
//         let rhs_q = &Z_N::<Q>::from(u64::from(*rhs));
//         Ciphertext {
//             ct: &self.ct + &(&build_gadget::<N, M, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>() * rhs_q),
//         }
//     }
// }

impl<
        'a,
        const N: usize,
        const M: usize,
        const P: u64,
        const Q: u64,
        const Q1: u64,
        const Q2: u64,
        const W1: u64,
        const W2: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul<Z_N<P>> for &'a Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>
{
    type Output = Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;
    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        let rhs_q = &Z_N_CycloNTT_CRT::<D, Q1, Q2, W1, W2>::from(u64::from(rhs));
        Ciphertext {
            ct: &self.ct
                * &gadget_inverse::<Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>, N, M, M, G_BASE, G_LEN>(
                    &(&build_gadget::<Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>, N, M, G_BASE, G_LEN>() * rhs_q),
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
        const Q1: u64,
        const Q2: u64,
        const W1: u64,
        const W2: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Add for &'a Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>
{
    type Output = Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;
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
        const Q1: u64,
        const Q2: u64,
        const W1: u64,
        const W2: u64,
        const D: usize,
        const G_BASE: u64,
        const G_LEN: usize,
    > Mul for &'a Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>
{
    type Output = Ciphertext<N, M, P, Q, Q1, Q2, W1, W2, D, G_BASE, G_LEN>;
    fn mul(self, rhs: Self) -> Self::Output {
        Ciphertext {
            ct: &self.ct * &gadget_inverse::<Z_N_CycloNTT_CRT<D, Q1, Q2, W1, W2>, N, M, M, G_BASE, G_LEN>(&rhs.ct),
        }
    }
}

/*
 * RingGSW_NTT_CRT params
 */

pub struct Params {
    pub N: usize,
    pub M: usize,
    pub P: u64,
    pub Q1: u64,
    pub Q2: u64,
    pub W1: u64,
    pub W2: u64,
    pub D: usize,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
}

macro_rules! gsw_from_params {
    ($params:expr) => {
        RingGSW_NTT_CRT<
            { $params.N - 1 },
            { $params.N },
            { $params.M },
            { $params.P },
            { $params.Q1 * $params.Q2 },
            { $params.Q1 },
            { $params.Q2 },
            { $params.W1 },
            { $params.W2 },
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

// TODO? Tests passed when I set W1, W2 = 1, but maybe this is only because of constants...
pub const RingGSW_NTT_CRT_TEST_PARAMS: Params = Params {
    N: 2,
    M: 112,
    P: 31,
    Q1: 268369921,
    Q2: 249561089,
    W1: 185593570,
    W2: 188090060,
    D: 4,
    G_BASE: 2,
    NOISE_WIDTH_MILLIONTHS: 6_400_000,
};

pub type RingGSW_NTT_CRTTest = gsw_from_params!(RingGSW_NTT_CRT_TEST_PARAMS);

#[cfg(test)]
mod test {
    use crate::math::z_n_cyclo_crt::Z_N_CycloRaw_CRT;

    use super::*;

    #[test]
    fn keygen_is_correct() {
        let threshold = 4f64 * (RingGSW_NTT_CRT_TEST_PARAMS.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let (A, s_T) = RingGSW_NTT_CRTTest::keygen();
        let e = &s_T.s_T * &A.A;

        for i in 0..RingGSW_NTT_CRT_TEST_PARAMS.M {
            assert!(
                (e[(0, i)].norm() as f64) < threshold,
                "e^T = s_T * A was too big"
            );
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = RingGSW_NTT_CRTTest::keygen();
        for i in 0_u64..10_u64 {
            let mu = Z_N::from(i);
            let ct = RingGSW_NTT_CRTTest::encrypt(&A, mu);
            let pt = RingGSW_NTT_CRTTest::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = RingGSW_NTT_CRTTest::keygen();
        for i in 0_u64..10_u64 {
            for j in 0_u64..10_u64 {
                let mu1 = Z_N::from(i);
                let mu2 = Z_N::from(j);
                let ct1 = RingGSW_NTT_CRTTest::encrypt(&A, mu1);
                let ct2 = RingGSW_NTT_CRTTest::encrypt(&A, mu2);
                // let pt_add = RingGSW_NTT_CRTTest::decrypt(&s_T, &(&ct1 + mu2));
                let pt_mul = RingGSW_NTT_CRTTest::decrypt(&s_T, &(&ct1 * mu2));
                let pt_add_ct = RingGSW_NTT_CRTTest::decrypt(&s_T, &(&ct1 + &ct2));
                let pt_mul_ct = RingGSW_NTT_CRTTest::decrypt(&s_T, &(&ct1 * &ct2));
                // assert_eq!(pt_add, &mu1 + &mu2, "addition by scalar failed");
                assert_eq!(pt_add_ct, &mu1 + &mu2, "ciphertext addition failed");

                assert_eq!(pt_mul, &mu1 * &mu2, "multiplication by scalar failed");
                assert_eq!(pt_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
            }
        }
    }

    #[test]
    fn homomorphism_mul_multiple_correct() {
        let (A, s_T) = RingGSW_NTT_CRTTest::keygen();
        let mu1 = Z_N::from(5_u64);
        let mu2 = Z_N::from(12_u64);
        let mu3 = Z_N::from(6_u64);
        let mu4 = Z_N::from(18_u64);

        let ct1 = RingGSW_NTT_CRTTest::encrypt(&A, mu1);
        let ct2 = RingGSW_NTT_CRTTest::encrypt(&A, mu2);
        let ct3 = RingGSW_NTT_CRTTest::encrypt(&A, mu3);
        let ct4 = RingGSW_NTT_CRTTest::encrypt(&A, mu4);

        let ct12 = &ct1 * &ct2;
        let ct34 = &ct3 * &ct4;
        let ct1234 = &ct12 * &ct34;
        // let ct31234 = &ct3 * &ct1234;

        let pt12 = RingGSW_NTT_CRTTest::decrypt(&s_T, &ct12);
        let pt34 = RingGSW_NTT_CRTTest::decrypt(&s_T, &ct34);
        let pt1234 = RingGSW_NTT_CRTTest::decrypt(&s_T, &ct1234);
        // let pt31234 = gsw::decrypt(&s_T, &ct31234);

        assert_eq!(pt12, &mu1 * &mu2);
        assert_eq!(pt34, &mu3 * &mu4);
        assert_eq!(pt1234, &(&(&mu1 * &mu2) * &mu3) * &mu4);
        // assert_eq!(pt31234, &(&(&(&mu1 * &mu2) * &mu3) * &mu4) * &mu3);
    }
}
