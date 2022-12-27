use crate::{z_n::*, ring_elem::*, matrix::*, discrete_gaussian::*, params::*};
use rand::{SeedableRng, Rng};
use rand_chacha::ChaCha20Rng;

// TODO: fill out these types
pub struct PublicKey {
    
}

pub struct PrivateKey {
    
}

pub struct Ciphertext {
    
}

// TODO: The clunk is real :(
pub fn keygen<const N_MINUS_1: usize, const N: usize, const M: usize, const Q: u64>(width: f64) -> (Matrix<N,M,Z_N<Q>>, Matrix<1,N,Z_N<Q>>) {
    assert!(N_MINUS_1 + 1 == N);

    let dg = DiscreteGaussian::init(width);
    let mut rng = ChaCha20Rng::from_entropy();

    let a_bar : Matrix<N_MINUS_1, M, Z_N<Q>> = Matrix::random_rng(&mut rng);
    let s_bar_T : Matrix<1, N_MINUS_1, Z_N<Q>>= Matrix::random_rng(&mut rng);
    let e : Matrix<1, M, Z_N<Q>> = dg.sample_int_matrix(&mut rng);

    let a : Matrix<N, M, Z_N<Q>> = stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
    let mut s_T : Matrix<1, N, Z_N<Q>> = Matrix::new_uninitialized();
    s_T.copy_into(&(-&s_bar_T), 0, 0);
    s_T[(0,N-1)] = Z_N::one();
    (a, s_T)
}

pub fn test_keygen() -> (Matrix<{DumbParams.N},{DumbParams.M},Z_N<{DumbParams.Q}>>, Matrix<1,{DumbParams.N},Z_N<{DumbParams.Q}>>) {

    keygen::<{DumbParams.N-1}, {DumbParams.N}, {DumbParams.M}, {DumbParams.Q}>(DumbParams.noise_width)
}

pub fn encrypt<const N: usize, const M: usize, const Q: u64>(pk : &Matrix<N, M, Z_N<Q>>, mu: u64) -> Matrix<N, M, Z_N<Q>> {
    let mut rng = ChaCha20Rng::from_entropy();
    let mut R : Matrix<M, M, Z_N<Q>> = Matrix::zero();
    for i in 0..M {
        for j in 0..M {
            if rng.gen_bool(0.5) {
                R[(i,j)] = Z_N::one();
            }
        }
    }

    // TODO: real gadget matrix
    let mut G : Matrix<N, M, Z_N<Q>> = Matrix::zero();
    for i in 0..N {
        G[(i,i)] = Z_N::one();
    }

    // TODO: pt size != 2
    let mu = Z_N::new_u(mu * Q / 2);
    let ct = &(pk * &R) + &(&G * mu);
    ct
}

pub fn decrypt<const N: usize, const M: usize, const Q: u64>(sk: &Matrix<1, N, Z_N<Q>>, ct: &Matrix<N, M, Z_N<Q>>) -> u64 {

    // TODO: fix after using real gadget
    let pt = (sk * ct)[(0, N-1)];

    // TODO: pt size != 2
    let floored = pt.to_u() / (Q / 4);
    if floored >= 3 {
        return 0
    }

    return (floored+1) / 2
}
