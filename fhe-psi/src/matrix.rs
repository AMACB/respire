use crate::ring_elem::*;
use std::ops::{Mul, Add, Neg, Index, IndexMut};
use rand::Rng;

#[derive(Debug, PartialEq, Eq)]
pub struct Matrix<const N: usize, const M: usize, R: RingElement> where for <'a> &'a R: RingElementRef<R> {
    data: [[R ; M] ; N],
}

impl<const N: usize, const M: usize, R: RingElement> Matrix<N,M,R> where for <'a> &'a R: RingElementRef<R> {

    pub fn zero() -> Self {
        let mat = [[(); M]; N].map(|row| row.map(|_| R::zero()));
        Matrix { data: mat }
    }

    pub fn random_rng<T: Rng>(rng: &mut T) -> Self {
        let mat = [[(); M]; N].map(|row| row.map(|_| R::random(rng)));
        Matrix { data: mat }
    }

    pub fn copy_into<const N2: usize, const M2: usize>(&mut self, m: &Matrix<N2,M2,R>, target_row: usize, target_col: usize) {
        assert!(target_row < N, "copy out of bounds");
        assert!(target_col < M, "copy out of bounds");
        assert!(target_row + N2 <= N, "copy out of bounds");
        assert!(target_col + M2 <= M, "copy out of bounds");
        for r in 0..N2 {
            for c in 0..M2 {
                self.data[target_row+r][target_col+c] = m.data[r][c].clone();
            }
        }
    }
}

pub fn identity<const N: usize, R: RingElement>() -> Matrix<N,N,R> where for <'a> &'a R: RingElementRef<R> {
    let mut out = Matrix::zero();
    for i in 0..N {
        out.data[i][i] = R::one();
    }
    out
}

// TODO: lol oops cannot smartly do M1+M2 or N1+N2
pub fn append<const N: usize, const M1: usize, const M2: usize, const M3: usize, R: RingElement>(a: &Matrix<N, M1, R>, b : &Matrix<N, M2, R>) -> Matrix<N, M3, R> where for <'a> &'a R: RingElementRef<R> {
    assert_eq!(M1+M2, M3, "dimensions do not add correctly");
    let mut c = Matrix::zero();
    c.copy_into(a, 0, 0);
    c.copy_into(b, 0, M1);
    c
}
pub fn stack<const N1: usize, const N2: usize, const N3: usize, const M: usize, R: RingElement>(a: &Matrix<N1, M, R>, b : &Matrix<N2, M, R>) -> Matrix<N3, M, R> where for <'a> &'a R: RingElementRef<R> {
    assert_eq!(N1+N2, N3, "dimensions do not add correctly");
    let mut c = Matrix::zero();
    c.copy_into(a, 0, 0);
    c.copy_into(b, N1, 0);
    c
}

// TODO: is it possible to make this good at compile time bounds checks?

impl<const N: usize, const M: usize, R: RingElement> Index<(usize, usize)> for Matrix<N, M, R> where for <'a> &'a R: RingElementRef<R> {
    type Output = R;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < N, "index out of bounds");
        assert!(index.1 < M, "index out of bounds");
        &self.data[index.0][index.1]
    }
}
impl<const N: usize, const M: usize, R: RingElement> IndexMut<(usize, usize)> for Matrix<N, M, R> where for <'a> &'a R: RingElementRef<R> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<const N: usize, const M: usize, R: RingElement> Mul<&R> for &Matrix<N, M, R> where for <'a> &'a R: RingElementRef<R> {
    type Output = Matrix<N,M,R>;

    fn mul(self, other: &R) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = &self.data[r][c] * &other;
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, const K: usize, R: RingElement> Mul<&Matrix<M, K, R>> for &Matrix<N, M, R> where for <'a> &'a R: RingElementRef<R> {
    type Output = Matrix<N,K,R>;

    fn mul(self, other: &Matrix<M, K, R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..K {
                for i in 0..M {
                    out.data[r][c] += &(&self.data[r][i] * &other.data[i][c]);
                }
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> Add<&Matrix<N, M, R>> for &Matrix<N, M, R> where for <'a> &'a R: RingElementRef<R> {
    type Output = Matrix<N,M,R>;
    fn add(self, other: &Matrix<N,M,R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = &self.data[r][c] + &other.data[r][c]
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> Neg for &Matrix<N, M, R> where for <'a> &'a R: RingElementRef<R> {
    type Output = Matrix<N,M,R>;
    fn neg(self) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out.data[r][c] = -&self.data[r][c];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::z_n::*;

    const N : usize = 2;
    const M : usize = 8;
    const Q : u64   = 11;

    #[test]
    fn zero_matrix_is_correct() {
        let zero : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                assert_eq!(zero[(i,j)], Z_N::zero());
            }
        }
    }

    #[test]
    fn identity_matrix_is_correct() {
        let I : Matrix<M,M,Z_N<Q>> = identity();
        for i in 0..M {
            for j in 0..M {
                if i == j {
                    assert_eq!(I[(i,j)], Z_N::one());
                } else {
                    assert_eq!(I[(i,j)], Z_N::zero());
                }
            }
        }
    }

    fn addition_test1() {
        let mut mat : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }
        let zero : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        assert_eq!(&mat + &zero, mat, "multiplication by identity failed");
    }

    fn addition_test2() {
        let mut mat1 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat1[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }

        let mut mat2 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i,j)] = Z_N::new_u((i + 2*j) as u64);
            }
        }

        let mut mat3 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat3[(i,j)] = Z_N::new_u((i*(M+1) + 3*j) as u64);
            }
        }

        assert_eq!(&mat1 + &mat2, mat3);
    }

    #[test]
    fn addition_is_correct() {
        addition_test1();
        addition_test2();
    }

    fn multiplication_test1() {
        let mut mat : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }
        let I = identity();
        assert_eq!(&mat * &I, mat, "multiplication by identity failed");
    }

    fn multiplication_test2() {
        let mut mat1 : Matrix<N,N,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..N {
                mat1[(i,j)] = Z_N::new_u((i*N + j) as u64);
            }
        }

        let mut mat2 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }

        let mut mat3 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for j in 0..M {
            mat3[(0,j)] = Z_N::new_u((M + j) as u64);
            mat3[(1,j)] = Z_N::new_u((3*M + 5*j) as u64);
        }

        assert_eq!(&mat1 * &mat2, mat3);
    }

    #[test]
    fn multiplication_is_correct() {
        multiplication_test1();
        multiplication_test2();
    }

    #[test]
    fn negation_is_correct() {
        let mut mat : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }
        let zero : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        assert_eq!(&mat + &(-&mat), zero, "addition by negation doesn't yield zero");
    }

    #[test]
    fn scalar_mult_is_correct() {
        let mut mat1 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat1[(i,j)] = Z_N::new_u((i*M + j) as u64);
            }
        }

        let mut mat2 : Matrix<N,M,Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i,j)] = Z_N::new_u((5*(i*M + j)) as u64);
            }
        }

        let zero : Matrix<N,M,Z_N<Q>> = Matrix::zero();

        assert_eq!(&mat1 * &Z_N::zero(), zero, "multiplication by scalar zero doesn't yield zero");
        assert_eq!(&mat1 * &Z_N::one(), mat1, "multiplication by scalar one doesn't yield itself");
        assert_eq!(&mat1 * &Z_N::new_u(5), mat2);
    }
}
