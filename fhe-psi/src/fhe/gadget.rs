//! Gadget matrix and gadget inverse (n-ary decomposition).

use crate::math::matrix::*;
use crate::math::ring_elem::*;

// TODO
// Write tests for Z_N_Cyclo

pub fn build_gadget<
    R: RingElementDivModdable,
    const N: usize,
    const M: usize,
    const Q: u64,
    const G_BASE: u64,
    const G_LEN: usize,
>() -> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    let mut gadget = Matrix::zero();

    let mut x = 1;
    let mut i = 0;

    for j in 0..M {
        gadget[(i, j)] = x.into();
        x *= G_BASE;
        if x > Q {
            i += 1;
            x = 1;
        }
    }

    gadget
}

pub fn gadget_inverse<
    R: RingElementDivModdable,
    const N: usize,
    const M: usize,
    const K: usize,
    const G_BASE: u64,
    const G_LEN: usize,
>(
    m: &Matrix<N, K, R>,
) -> Matrix<M, K, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    let mut m_expanded: Matrix<M, K, R> = Matrix::zero();

    for i in 0..N {
        for j in 0..K {
            let mut a: R = m[(i, j)].clone();
            let mut r;
            for k in 0..G_LEN {
                (a, r) = a.div_mod(G_BASE);
                m_expanded[(i * G_LEN + k, j)] = r;
            }
        }
    }
    m_expanded
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::utils::ceil_log;
    use crate::math::z_n::Z_N;

    const N: usize = 2;
    const M: usize = 8;
    const Q: u64 = 11;
    const G_BASE: u64 = 2;
    const G_LEN: usize = ceil_log(G_BASE, Q);

    #[test]
    fn gadget_is_correct() {
        let G = build_gadget::<Z_N<Q>, N, M, Q, G_BASE, G_LEN>();

        let mut expected_G: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        expected_G[(0, 0)] = 1_u64.into();
        expected_G[(0, 1)] = 2_u64.into();
        expected_G[(0, 2)] = 4_u64.into();
        expected_G[(0, 3)] = 8_u64.into();

        expected_G[(1, 4)] = 1_u64.into();
        expected_G[(1, 5)] = 2_u64.into();
        expected_G[(1, 6)] = 4_u64.into();
        expected_G[(1, 7)] = 8_u64.into();

        assert_eq!(G, expected_G, "gadget constructed incorrectly");
    }

    #[test]
    fn gadget_inverse_is_correct() {
        let mut R: Matrix<N, M, Z_N<Q>> = Matrix::zero();

        for i in 0..N {
            for j in 0..M {
                R[(i, j)] = ((i * M + j) as u64).into();
            }
        }

        let G = build_gadget::<Z_N<Q>, N, M, Q, G_BASE, G_LEN>();
        let R_inv = gadget_inverse::<Z_N<Q>, N, M, M, G_BASE, G_LEN>(&R);
        let R_hopefully = &G * &R_inv;
        assert_eq!(R, R_hopefully, "gadget inverse was not correct");
    }
}
