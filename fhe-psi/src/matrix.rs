use crate::ring_elem::*;
use std::ops::{Mul, Index, IndexMut};
use rand::Rng;

#[derive(Debug)]
pub struct Matrix<const N: usize, const M: usize, R: RingElement> {
    pub data: [[R ; M] ; N],
}

impl<const N: usize, const M: usize, R: RingElement> Matrix<N,M,R> {

    fn new_uninitialized() -> Self {
        // TODO: do something unsafe to not call R::zero()
        Matrix {
            data: [[R::zero() ; M] ; N],
        }
    }


    pub fn zero() -> Self {
        let mut out = Matrix::new_uninitialized();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = R::zero();
            }
        }
        out
    }

    pub fn random_rng<T: Rng>(rng: &mut T) -> Self {
        let mut out = Matrix::new_uninitialized();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = R::random(rng);
            }
        }
        out
    }
}

// TODO: is it possible to make this good at compile time bounds checks?
impl<const N: usize, const M: usize, R: RingElement> Index<(usize, usize)> for Matrix<N, M, R> {
    type Output = R;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}
impl<const N: usize, const M: usize, R: RingElement> IndexMut<(usize, usize)> for Matrix<N, M, R> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<const N: usize, const M: usize, const K: usize, R: RingElement> Mul<&Matrix<M, K, R>> for &Matrix<N, M, R> {
    type Output = Matrix<N,K,R>;

    fn mul(self, other: &Matrix<M, K, R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..K {
                for i in 0..M {
                    out.data[r][c] += self.data[r][i] * other.data[i][c];
                }
            }
        }
        out
    }
}

// pub struct Polynomial<const D: usize, R: RingElement> {
//     pub coeffs: [R; D]
// }

// // TODO: align pls
// pub struct PolynomialNTT<const D: usize, R: RingElement> {
//     pub evals: [R; D]
// }
