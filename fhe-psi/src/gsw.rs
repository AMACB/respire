use crate::{z_n::*, matrix::*, gadget::*};
use std::ops::{Add, Mul};

#[derive(Debug)]
pub struct PublicKey<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> {
    A : Matrix<N,M,Z_N<Q>>
}

#[derive(Debug)]
pub struct PrivateKey<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> {
    s_T : Matrix<1,N,Z_N<Q>>
}

#[derive(Debug)]
pub struct Ciphertext<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> {
    ct : Matrix<N,M,Z_N<Q>>
}

impl<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> Add<&Z_N<P>> for &Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
    type Output = Ciphertext<N,M,P,Q,G_BASE,G_LEN>;
    fn add(self, rhs: &Z_N<P>) -> Self::Output {
        Ciphertext { ct: &self.ct + &(&build_gadget::<N,M,Q,G_BASE,G_LEN>() * &Z_N::new_u(rhs.to_u())) }
    }
}

impl<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> Mul<&Z_N<P>> for &Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
    type Output = Ciphertext<N,M,P,Q,G_BASE,G_LEN>;
    fn mul(self, rhs: &Z_N<P>) -> Self::Output {
        Ciphertext { ct: &self.ct * &gadget_inverse::<N,M,M,Q,G_BASE,G_LEN>(&(&build_gadget::<N,M,Q,G_BASE,G_LEN>() * &Z_N::new_u(rhs.to_u()))) }
    }
}

impl<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> Add for &Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
    type Output = Ciphertext<N,M,P,Q,G_BASE,G_LEN>;
    fn add(self, rhs: Self) -> Self::Output {
        Ciphertext { ct: &self.ct + &rhs.ct }
    }
}

impl<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> Mul for &Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
    type Output = Ciphertext<N,M,P,Q,G_BASE,G_LEN>;
    fn mul(self, rhs: Self) -> Self::Output {
        Ciphertext { ct: &self.ct * &gadget_inverse::<N,M,M,Q,G_BASE,G_LEN>(&rhs.ct) }
    }
}

pub mod gsw {
    use super::*;
    use crate::{ring_elem::*, discrete_gaussian::*, params::*};
    use rand::{SeedableRng, Rng};
    use rand_chacha::ChaCha20Rng;
    pub fn keygen<const N_MINUS_1: usize, const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(params : IntParams<N,M,P,Q,G_BASE,G_LEN,N_MINUS_1>) -> ( PublicKey<N,M,P,Q,G_BASE,G_LEN>, PrivateKey<N,M,P,Q,G_BASE,G_LEN> ) {
        assert!(N_MINUS_1 + 1 == N);

        let dg = DiscreteGaussian::init(params.noise_width);
        let mut rng = ChaCha20Rng::from_entropy();

        let a_bar : Matrix<N_MINUS_1, M, Z_N<Q>> = Matrix::random_rng(&mut rng);
        let s_bar_T : Matrix<1, N_MINUS_1, Z_N<Q>>= Matrix::random_rng(&mut rng);
        let e : Matrix<1, M, Z_N<Q>> = dg.sample_int_matrix(&mut rng);

        let A : Matrix<N, M, Z_N<Q>> = stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
        let mut s_T : Matrix<1, N, Z_N<Q>> = Matrix::zero();
        s_T.copy_into(&(-&s_bar_T), 0, 0);
        s_T[(0,N-1)] = Z_N::one();
        (PublicKey {A}, PrivateKey {s_T})
    }

    pub fn encrypt<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(pk : &PublicKey<N,M,P,Q,G_BASE,G_LEN>, mu: &Z_N<P>) -> Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
        let A = &pk.A;

        let mut rng = ChaCha20Rng::from_entropy();

        let mut R : Matrix<M, M, Z_N<Q>> = Matrix::zero();
        for i in 0..M {
            for j in 0..M {
                if rng.gen_bool(0.5) {
                    R[(i,j)] = Z_N::one();
                }
            }
        }

        let G = build_gadget::<N,M,Q,G_BASE,G_LEN>();

        let mu = Z_N::new_u(mu.to_u());
        let ct = &(A * &R) + &(&G * &mu);
        Ciphertext {ct}
    }

    pub fn decrypt<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(sk: &PrivateKey<N,M,P,Q,G_BASE,G_LEN>, ct: &Ciphertext<N,M,P,Q,G_BASE,G_LEN>) -> Z_N<P> {
        let s_T = &sk.s_T;
        let ct = &ct.ct;
        let g_inv = &gadget_inverse::<N,M,N,Q,G_BASE,G_LEN>(&(&identity::<N, Z_N<Q>>() * &Z_N::new_u(Q / P)));

        let pt = &(&(s_T * ct) * g_inv)[(0, N-1)];

        let floored = pt.to_u() * P * 2 / Q;

        Z_N::new_u((floored+1) / 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;

    #[test]
    fn keygen_is_correct() {
        let threshold = 4f64 * TEST_PARAMS.noise_width;
        let (A, s_T) = gsw::keygen(TEST_PARAMS);
        let e = &s_T.s_T * &A.A;

        for i in 0..TEST_PARAMS_RAW.M {
            // abs(e[i]) < threshold
            assert!((e[(0,i)].to_u() as f64) < threshold || ((-&e[(0,i)]).to_u() as f64) < threshold, "e^T = s_T * A was too big");
        }
    }

    #[test]
    fn encryption_is_correct() {
        let (A, s_T) = gsw::keygen(TEST_PARAMS);
        for i in 0..10 {
            let mu = Z_N::new_u(i);
            let ct = gsw::encrypt(&A, &mu);
            let pt = gsw::decrypt(&s_T, &ct);
            assert_eq!(pt, mu, "decryption failed");
        }
    }

    #[test]
    fn homomorphism_is_correct() {
        let (A, s_T) = gsw::keygen(TEST_PARAMS);
        for i in 0..10 {
            for j in 0..10 {
                let mu1 = Z_N::new_u(i);
                let mu2 = Z_N::new_u(j);
                let ct1 = gsw::encrypt(&A, &mu1);
                let ct2 = gsw::encrypt(&A, &mu2);
                let pt_add = gsw::decrypt(&s_T, &(&ct1 + &mu2));
                let pt_mul = gsw::decrypt(&s_T, &(&ct1 * &mu2));
                let pt_add_ct = gsw::decrypt(&s_T, &(&ct1 + &ct2));
                let pt_mul_ct = gsw::decrypt(&s_T, &(&ct1 * &ct2));
                assert_eq!(pt_add, &mu1 + &mu2, "addition by scalar failed");
                assert_eq!(pt_add_ct, &mu1 + &mu2, "ciphertext addition failed");

                assert_eq!(pt_mul, &mu1 * &mu2, "multiplication by scalar failed");
                assert_eq!(pt_mul_ct, &mu1 * &mu2, "ciphertext multiplication failed");
            }
        }
    }
}
