//! Gadget matrix and gadget inverse (n-ary decomposition).

use crate::math::matrix::*;
use crate::math::ring_elem::*;

// TODO
// Write tests for Z_N_Cyclo

pub const fn base_from_len(t: usize, q: u64) -> u64 {
    // z = floor(q^(1/t)) + 1
    let mut z_lo: u64 = 2;
    let mut z_hi: u64 = u64::MAX;
    while z_lo < z_hi - 1 {
        let z_mid = z_lo + (z_hi - z_lo) / 2;
        match (z_mid - 1).checked_pow(t as u32) {
            Some(x) if x <= q => z_lo = z_mid,
            _ => z_hi = z_mid,
        }
    }
    z_lo
}

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

pub struct IntModDecomposition<const BASE: u64, const LEN: usize> {
    a: u64,
    negate_all: bool,
    k: usize,
    n: u64,
}

impl<const BASE: u64, const LEN: usize> IntModDecomposition<BASE, LEN> {
    const fn max_positive(n: u64) -> u64 {
        if n > BASE.pow(LEN as u32) {
            panic!("RingElementDecomposable requires modulus <= base^len");
        }

        // Wrap if > threshold
        let threshold = BASE / 2;
        let mut i = 0;
        let mut sum = 0;
        while i < LEN {
            sum *= BASE;
            sum += threshold;
            i += 1;
        }
        sum
    }

    pub fn new(mut a: u64, n: u64) -> Self {
        let negate_all = a > Self::max_positive(n);
        if negate_all {
            a = n - a;
        }
        let k = 0;
        Self {
            a,
            negate_all,
            k,
            n,
        }
    }
}

impl<const BASE: u64, const LEN: usize> Iterator for IntModDecomposition<BASE, LEN> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.k == LEN {
            return None;
        }

        let mut reduced = self.a % BASE;
        self.a /= BASE;
        let subtract_base = reduced > BASE / 2;
        if subtract_base {
            self.a += 1;
            reduced = self.n - (BASE - reduced);
        }
        if self.negate_all {
            reduced = self.n - reduced;
        }
        self.k += 1;
        Some(reduced)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (LEN - self.k, Some(LEN - self.k))
    }
}

/// Computes G^(-1) of an `N x K` matrix, producing an `M x K` matrix.
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

pub fn gadget_inverse_scalar<
    R: RingElementDecomposable<G_BASE, G_LEN>,
    const G_BASE: u64,
    const G_LEN: usize,
>(
    a: &R,
) -> Matrix<G_LEN, 1, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    let mut m_expanded: Matrix<G_LEN, 1, R> = Matrix::zero();
    a.decompose_into_mat(&mut m_expanded, 0, 0);
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

    #[test]
    fn gadget_factors() {
        let q = 268369921 * 249561089;
        for t in 2..56 {
            let z = base_from_len(t, q);
            eprintln!(
                "t = {}, z = {}, gadget factor = {}",
                t,
                z,
                t as u64 * (z / 2).pow(2)
            );
        }
    }
}
