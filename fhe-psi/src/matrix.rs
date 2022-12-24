use crate::ring_elem::*;
use std::ops::{Mul, Add, Neg, Index, IndexMut};
use rand::Rng;

#[derive(Debug)]
pub struct Matrix<const N: usize, const M: usize, R: RingElement> {
    pub data: [[R ; M] ; N],
}

impl<const N: usize, const M: usize, R: RingElement> Matrix<N,M,R> {

    pub fn new_uninitialized() -> Self {
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

    pub fn copy_into<const N2: usize, const M2: usize>(&mut self, m: &Matrix<N2,M2,R>, target_row: usize, target_col: usize) {
        assert!(target_row < N);
        assert!(target_col < M);
        assert!(target_row + N2 <= N);
        assert!(target_col + M2 <= M);
        for r in 0..N2 {
            for c in 0..M2 {
                self.data[target_row+r][target_col+c] = m.data[r][c];
            }
        }
    }
}

// TODO: lol oops cannot smartly do M1+M2 or N1+N2
pub fn append<const N: usize, const M1: usize, const M2: usize, const M3: usize, R: RingElement>(a: &Matrix<N, M1, R>, b : &Matrix<N, M2, R>) -> Matrix<N, M3, R> {
    assert!(M1+M2 == M3);
    let mut c = Matrix::new_uninitialized();
    c.copy_into(a, 0, 0);
    c.copy_into(b, 0, M1);
    c
}
pub fn stack<const N1: usize, const N2: usize, const N3: usize, const M: usize, R: RingElement>(a: &Matrix<N1, M, R>, b : &Matrix<N2, M, R>) -> Matrix<N3, M, R> {
    assert!(N1+N2 == N3);
    let mut c = Matrix::new_uninitialized();
    c.copy_into(a, 0, 0);
    c.copy_into(b, N1, 0);
    c
}

// TODO: is it possible to make this good at compile time bounds checks?

impl<const N: usize, const M: usize, R: RingElement> Index<(usize, usize)> for Matrix<N, M, R> {
    type Output = R;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < N);
        assert!(index.1 < M);
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

impl<const N: usize, const M: usize, R: RingElement> Add<&Matrix<N, M, R>> for &Matrix<N, M, R> {
    type Output = Matrix<N,M,R>;
    fn add(self, other: &Matrix<N,M,R>) -> Self::Output {
        let mut out = Matrix::new_uninitialized();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = self.data[r][c] + other.data[r][c]
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> Neg for &Matrix<N, M, R> {
    type Output = Matrix<N,M,R>;
    fn neg(self) -> Self::Output {
        let mut out = Matrix::new_uninitialized();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = -self.data[r][c];
            }
        }
        out
    }
}
