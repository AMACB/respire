use crate::matrix::*;
use crate::z_n::*;

pub fn build_gadget<const N: usize, const M: usize, const Q: u64, const G_BASE: u64, const G_LEN: usize>() -> Matrix<N,M,Z_N<Q>> {
    let mut gadget = Matrix::zero();

    let mut x = 1;
    let mut i = 0;

    for j in 0..M {
        gadget[(i,j)] = Z_N::new_u(x);
        x *= G_BASE;
        if x > Q {
            i += 1;
            x = 1;
        }
    }

    gadget
}

pub fn gadget_inverse<const N: usize, const M: usize, const K: usize, const Q: u64, const G_BASE: u64, const G_LEN: usize>(m: &Matrix<N,K,Z_N<Q>>) -> Matrix<M,K,Z_N<Q>> {
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::*;

    const N : usize     = 2;
    const M : usize     = 8;
    const Q : u64       = 11;
    const G_BASE : u64  = 2;
    const G_LEN : usize = ceil_log(G_BASE, Q);

    #[test]
    fn gadget_is_correct() {
        let G = build_gadget::<N,M,Q,G_BASE,G_LEN>();

        let mut expected_G : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        expected_G[(0,0)] = Z_N::new_u(1);
        expected_G[(0,1)] = Z_N::new_u(2);
        expected_G[(0,2)] = Z_N::new_u(4);
        expected_G[(0,3)] = Z_N::new_u(8);

        expected_G[(1,4)] = Z_N::new_u(1);
        expected_G[(1,5)] = Z_N::new_u(2);
        expected_G[(1,6)] = Z_N::new_u(4);
        expected_G[(1,7)] = Z_N::new_u(8);

        assert_eq!(G, expected_G, "gadget constructed incorrectly");
    }

    #[test]
    fn gadget_inverse_is_correct() {
        let mut R : Matrix<N,M,Z_N<Q>> = Matrix::zero();

        for i in 0..N {
            for j in 0..M {
                R[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }

        let G = build_gadget::<N,M,Q,G_BASE,G_LEN>();
        let R_inv = gadget_inverse::<N,M,M,Q,G_BASE,G_LEN>(&R);
        let R_hopefully = &G * &R_inv;
        assert_eq!(R, R_hopefully, "gadget inverse was not correct");
    }
}
