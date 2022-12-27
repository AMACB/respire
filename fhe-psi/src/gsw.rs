use crate::{z_n::*, ring_elem::*, matrix::*, discrete_gaussian::*, params::*, gadget::gen_G};
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;

#[derive(Debug)]
pub struct PublicKey<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64> {
    A : Matrix<N,M,Z_N<Q>>
}

#[derive(Debug)]
pub struct PrivateKey<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64> {
    s_T : Matrix<1,N,Z_N<Q>>
}

#[derive(Debug)]
pub struct Ciphertext<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64> {
    ct : Matrix<N,M,Z_N<Q>>
}

pub fn keygen<const N_MINUS_1: usize, const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64>(params : IntParams<N,M,P,Q,G_BASE, N_MINUS_1>) -> ( PublicKey<N,M,P,Q,G_BASE>, PrivateKey<N,M,P,Q,G_BASE> ) {
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

pub fn encrypt<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64>(pk : &PublicKey<N,M,P,Q,G_BASE>, mu: u64) -> Ciphertext<N,M,P,Q,G_BASE> {
    assert!(mu < P);
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

    let G = gen_G::<N,M,Q,G_BASE>();

    let mu = Z_N::new_u(mu * Q / P);
    let ct = &(A * &R) + &(&G * mu);
    Ciphertext {ct}
}

pub fn decrypt<const N: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64>(sk: &PrivateKey<N,M,P,Q,G_BASE>, ct: &Ciphertext<N,M,P,Q,G_BASE>) -> u64 {
    let s_T = &sk.s_T;
    let ct = &ct.ct;

    // TODO: encode this as a constant in Params, probably
    let mut x = 1;
    let mut g_len = 0;
    while x < Q {
        x *= G_BASE;
        g_len += 1;
    }
    let pt = (s_T * ct)[(0, (N-1)*g_len)];


    let floored = pt.to_u() * P * 2 / Q;

    ((floored+1) / 2) % P
}
