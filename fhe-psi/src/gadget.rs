use crate::matrix::*;
use crate::z_n::*;

pub fn gen_G<const N: usize, const M: usize, const Q: u64, const G_BASE: u64, const G_LEN: usize>() -> Matrix<N,M,Z_N<Q>> {
    let mut G = Matrix::zero();

    let mut x = 1;
    let mut i = 0;

    for j in 0..M {
        G[(i,j)] = Z_N::new_u(x);
        x *= G_BASE;
        if x > Q {
            i += 1;
            x = 1;
        }
    }

    G
}

pub fn G_inv<const N: usize, const M: usize, const K: usize, const Q: u64, const G_BASE: u64, const G_LEN: usize>(m: &Matrix<N,K,Z_N<Q>>) -> Matrix<M,K,Z_N<Q>> {
    let mut m_expanded = Matrix::zero();

    for i in 0..N {
        for j in 0..K {
            let mut a = m[(i,j)].to_u();
            for k in 0..G_LEN {
                m_expanded[(i*G_LEN + k, j)] = Z_N::new_u(a % G_BASE);
                a /= G_BASE;
            }
        }
    }
    m_expanded
}
