//! Generic GSW implementations over ring elements.

use crate::fhe::gadget::*;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use crate::math::z_n::Z_N;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub fn gsw_keygen<
    const N_MINUS_1: usize,
    const N: usize,
    const M: usize,
    R: RingElement,
    const NOISE_WIDTH_MILLIONTHS: u64,
>() -> (Matrix<N, M, R>, Matrix<1, N, R>)
where
    R: RandUniformSampled,
    R: RandDiscreteGaussianSampled,
    for<'a> &'a R: RingElementRef<R>,
{
    let mut rng = ChaCha20Rng::from_entropy();

    let a_bar: Matrix<N_MINUS_1, M, R> = Matrix::rand_uniform(&mut rng);
    let s_bar_T: Matrix<1, N_MINUS_1, R> = Matrix::rand_uniform(&mut rng);
    let e: Matrix<1, M, R> = Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);

    let A: Matrix<N, M, R> = Matrix::stack(&a_bar, &(&(&s_bar_T * &a_bar) + &e));
    let mut s_T: Matrix<1, N, R> = Matrix::zero();
    s_T.copy_into(&(-&s_bar_T), 0, 0);
    s_T[(0, N - 1)] = R::one();
    (A, s_T)
}

pub fn gsw_encrypt_pk<
    const N: usize,
    const M: usize,
    const G_BASE: u64,
    const G_LEN: usize,
    R: RingElement,
>(
    A: &Matrix<N, M, R>,
    mu: R,
) -> Matrix<N, M, R>
where
    R: RandZeroOneSampled,
    R: RingElementDecomposable<G_BASE, G_LEN>,
    for<'a> &'a R: RingElementRef<R>,
{
    let mut rng = ChaCha20Rng::from_entropy();
    let R_mat: Matrix<M, M, R> = Matrix::rand_zero_one(&mut rng);
    let G = build_gadget::<R, N, M, G_BASE, G_LEN>();
    &(A * &R_mat) + &(&G * &mu)
}

// TODO: Secret key encryption
// pub fn gsw_encrypt_sk<
//         const N: usize,
//     const M: usize,
//     const P: u64,
//     const Q: u64,
//     const G_BASE: u64,
//     const G_LEN: usize,
//     R: RingElement,
//     >(A: &Matrix<N, M, R>, mu: R) -> Matrix<N, M, R>
// where
//     R: RandZeroOneSampled,
//     R: RingElementDecomposable<G_BASE, G_LEN>,
// for<'a> &'a R: RingElementRef<R>
// {
//     let mut rng = ChaCha20Rng::from_entropy();
//     let R_mat: Matrix<M, M, R> = Matrix::rand_zero_one(&mut rng);

//     let G = build_gadget::<R, N, M, Q, G_BASE, G_LEN>();

//     let ct = &(A * &R_mat) + &(&G * &mu);
//     ct
// }

// This does *not* do any rounding, hence half decrypt.
pub fn gsw_half_decrypt<
    const N: usize,
    const M: usize,
    const P: u64,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    R: RingElement,
>(
    s_T: &Matrix<1, N, R>,
    ct: &Matrix<N, M, R>,
) -> R
where
    R: RandZeroOneSampled,
    R: RingElementDecomposable<G_BASE, G_LEN>,
    for<'a> &'a R: RingElementRef<R>,
{
    let q_over_p = R::from(Q / P);
    let g_inv =
        &gadget_inverse::<R, N, M, N, G_BASE, G_LEN>(&(&Matrix::<N, N, R>::identity() * &q_over_p));

    // is clone bad?
    (&(s_T * ct) * g_inv)[(0, N - 1)].clone()
}

pub fn gsw_round<const P: u64, const Q: u64, T: Into<u64>>(x: T) -> Z_N<P> {
    let floored = x.into() * P * 2 / Q;
    Z_N::from((floored + 1) / 2)
}
