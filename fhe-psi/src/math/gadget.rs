//! Gadget matrix and gadget inverse (n-ary decomposition).

use crate::math::matrix::*;
use crate::math::ring_elem::*;

// TODO
// Write tests for Z_N_Cyclo

pub fn build_gadget<
    R: RingElementDecomposable<G_BASE, G_LEN>,
    const N: usize,
    const M: usize,
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
        if j % G_LEN == G_LEN - 1 {
            i += 1;
            x = 1;
        }
    }

    gadget
}

pub trait RingElementDecomposable<const BASE: u64, const LEN: usize>: RingElement
where
    for<'a> &'a Self: RingElementRef<Self>,
{
    /// Computes the `BASE`-ary decomposition as a `1 x LEN` column vector, and writes it into `mat`
    /// starting at the index `(i,j)`.
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    );
}

pub fn gadget_inverse<
    R: RingElementDecomposable<G_BASE, G_LEN>,
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
            m[(i, j)].decompose_into_mat(&mut m_expanded, i * G_LEN, j);
        }
    }
    m_expanded
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod::IntMod;
    use crate::math::utils::ceil_log;

    const N: usize = 2;
    const M: usize = 8;
    const Q: u64 = 11;
    const G_BASE: u64 = 2;
    const G_LEN: usize = ceil_log(G_BASE, Q);

    #[test]
    fn gadget_is_correct() {
        let g_mat = build_gadget::<IntMod<Q>, N, M, G_BASE, G_LEN>();

        let mut expected_g_mat: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        expected_g_mat[(0, 0)] = 1_u64.into();
        expected_g_mat[(0, 1)] = 2_u64.into();
        expected_g_mat[(0, 2)] = 4_u64.into();
        expected_g_mat[(0, 3)] = 8_u64.into();

        expected_g_mat[(1, 4)] = 1_u64.into();
        expected_g_mat[(1, 5)] = 2_u64.into();
        expected_g_mat[(1, 6)] = 4_u64.into();
        expected_g_mat[(1, 7)] = 8_u64.into();

        assert_eq!(g_mat, expected_g_mat, "gadget constructed incorrectly");
    }

    #[test]
    fn gadget_inverse_is_correct() {
        let mut m: Matrix<N, M, IntMod<Q>> = Matrix::zero();

        for i in 0..N {
            for j in 0..M {
                m[(i, j)] = ((i * M + j) as u64).into();
            }
        }

        let g_mat = build_gadget::<IntMod<Q>, N, M, G_BASE, G_LEN>();
        let g_inv_m = gadget_inverse::<IntMod<Q>, N, M, M, G_BASE, G_LEN>(&m);
        let m_hopefully = &g_mat * &g_inv_m;
        assert_eq!(m, m_hopefully, "gadget inverse was not correct");
    }
}
