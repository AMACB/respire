use crate::{z_n::*, ring_elem::*, matrix::*, discrete_gaussian::*, params::*, gadget::*};
use std::ops::{Add, Mul};
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;

#[derive(Debug)]
pub struct PublicKey<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> {
    A : Matrix<N,M,Z_N<Q>>
}

#[derive(Debug)]
pub struct PrivateKey<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> {
    s_T : Matrix<1,N,Z_N<Q>>
}

pub fn keygen<const N_MINUS_1: usize, const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(params : IntParams<N,M,P,Q,G_BASE,G_LEN,N_MINUS_1>) -> ( PublicKey<N,M,P,Q,G_BASE,G_LEN>, PrivateKey<N,M,P,Q,G_BASE,G_LEN> ) {
    assert!(N_MINUS_1 + 1 == N);

    let dg = DiscreteGaussian::init(params.noise_width);
    let mut rng = ChaCha20Rng::from_entropy();

    let a_bar : Matrix<N_MINUS_1, M, Z_N<Q>> = Matrix::random_rng(&mut rng);
    let s_bar_T : Matrix<1, N_MINUS_1, Z_N<Q>>= Matrix::random_rng(&mut rng);
    let e : Matrix<1, M, Z_N<Q>> = dg.sample_int_matrix(&mut rng);

    let A : Matrix<N, M, Z_N<Q>> = stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
    let mut s_T : Matrix<1, N, Z_N<Q>> = Matrix::new_uninitialized();
    s_T.copy_into(&(-&s_bar_T), 0, 0);
    s_T[(0,N-1)] = Z_N::one();
    (PublicKey {A}, PrivateKey {s_T})
}

#[derive(Debug)]
pub struct Ciphertext<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> {
    ct : Matrix<N,M,Z_N<Q>>
}

pub fn encrypt<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(pk : &PublicKey<N,M,P,Q,G_BASE,G_LEN>, mu: Z_N<P>) -> Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
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

    let G = gen_G::<N,M,Q,G_BASE,G_LEN>();

    let mu = Z_N::new_u(mu.to_u());
    let ct = &(A * &R) + &(&G * mu);
    Ciphertext {ct}
}

pub fn decrypt<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(sk: &PrivateKey<N,M,P,Q,G_BASE,G_LEN>, ct: &Ciphertext<N,M,P,Q,G_BASE,G_LEN>) -> Z_N<P> {
    let s_T = &sk.s_T;
    let ct = &ct.ct;
    let g_inv = &G_inv::<N,M,N,Q,G_BASE,G_LEN>(&(&identity::<N, Z_N<Q>>() * Z_N::new_u(Q / P)));

    let pt = (&(s_T * ct) * g_inv)[(0, N-1)];

    let floored = pt.to_u() * P * 2 / Q;

    Z_N::new_u((floored+1) / 2)
}

impl<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> Add<Z_N<P>> for &Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
    type Output = Ciphertext<N,M,P,Q,G_BASE,G_LEN>;
    fn add(self, rhs: Z_N<P>) -> Self::Output {
        Ciphertext { ct: &self.ct + &(&gen_G::<N,M,Q,G_BASE,G_LEN>() * Z_N::new_u(rhs.to_u())) }
    }
}

impl<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize> Mul<Z_N<P>> for &Ciphertext<N,M,P,Q,G_BASE,G_LEN> {
    type Output = Ciphertext<N,M,P,Q,G_BASE,G_LEN>;
    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        Ciphertext { ct: &self.ct * &G_inv::<N,M,M,Q,G_BASE,G_LEN>(&(&gen_G::<N,M,Q,G_BASE,G_LEN>() * Z_N::new_u(rhs.to_u()))) }
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
        Ciphertext { ct: &self.ct * &G_inv::<N,M,M,Q,G_BASE,G_LEN>(&rhs.ct) }
    }
}
