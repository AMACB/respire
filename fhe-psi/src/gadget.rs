use crate::matrix::*;
use crate::z_n::*;

pub fn gen_G<const N: usize, const M: usize, const Q: u64, const G_BASE: u64>() -> Matrix<N,M,Z_N<Q>> {
    let mut G = Matrix::zero();

    let mut x = 1;
    let mut i = 0;

    for j in 0..M {
        G[(i,j)] = Z_N::new_u(x);
        x *= G_BASE;
        if(x > Q) {
            i += 1;
            x = 1;
        }
    }

    G
}

