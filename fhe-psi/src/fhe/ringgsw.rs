use crate::fhe::fhe::{CiphertextRef, FHEScheme};
use crate::fhe::gadget::{build_gadget, gadget_inverse};
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::RingElement;
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

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
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
        let pt = Z_N::<Q>::try_from(pt).unwrap();
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
        todo!()
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
        todo!()
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
        todo!()
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test123() {}
}
