//! Matrices over generic rings.

use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::cmp::max;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

// TODO
// * Implement as an array instead of as a `Vec`. The main sticking point is that to move a matrix
// as an array to the heap, we are forced to copy (or use unsafe).
// * Enforce compile time checks for matrix dimensions

/// Representation of a matrix as a flattened row-major order vector. The operations in ring type
/// `R` are those used in the relevant matrix operations.
///
/// Technically, `Matrix` could in itself be `RingElement`. But so far there has not been a need
/// for this, so it is not implemented.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<const N: usize, const M: usize, R: RingElement>
where
    R: Sized,
    for<'a> &'a R: RingElementRef<R>,
{
    data: Vec<R>,
}

/// Matrix methods.

impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    /// Constructs the zero matrix.
    pub fn zero() -> Self {
        let mut vec = Vec::with_capacity(N * M);
        for _ in 0..N * M {
            vec.push(R::zero());
        }
        Matrix { data: vec }
    }

    /// Copies `m` into `self`, starting at `(target_row, target_col)`, by cloning each element.
    pub fn copy_into<const N2: usize, const M2: usize>(
        &mut self,
        m: &Matrix<N2, M2, R>,
        target_row: usize,
        target_col: usize,
    ) {
        debug_assert!(target_row < N, "copy out of bounds");
        debug_assert!(target_col < M, "copy out of bounds");
        debug_assert!(target_row + N2 <= N, "copy out of bounds");
        debug_assert!(target_col + M2 <= M, "copy out of bounds");
        for r in 0..N2 {
            for c in 0..M2 {
                self[(target_row + r, target_col + c)] = m[(r, c)].clone();
            }
        }
    }

    /// Appends `b` to `a` by augmentation, returning `[a | b]`.
    pub fn append<const M1: usize, const M2: usize>(
        a: &Matrix<N, M1, R>,
        b: &Matrix<N, M2, R>,
    ) -> Self
    where
        for<'a> &'a R: RingElementRef<R>,
    {
        debug_assert_eq!(M1 + M2, M, "dimensions do not add correctly");
        let mut c = Matrix::zero();
        c.copy_into(a, 0, 0);
        c.copy_into(b, 0, M1);
        c
    }

    /// Stacks `a` on top of `b` by "vertical" augmentation, returning `[a^T | b^T]^T`.
    pub fn stack<const N1: usize, const N2: usize>(
        a: &Matrix<N1, M, R>,
        b: &Matrix<N2, M, R>,
    ) -> Self
    where
        for<'a> &'a R: RingElementRef<R>,
    {
        debug_assert_eq!(N1 + N2, N, "dimensions do not add correctly");
        let mut c = Matrix::zero();
        c.copy_into(a, 0, 0);
        c.copy_into(b, N1, 0);
        c
    }
}

/// Square matrix specific methods.

impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    pub fn identity() -> Self {
        let mut out = Matrix::zero();
        for i in 0..N {
            out[(i, i)] = R::one();
        }
        out
    }
}

/// Indexing

impl<const N: usize, const M: usize, R: RingElement> Index<(usize, usize)> for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = R;

    /// Returns the `(row, col)` element of the matrix.
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        debug_assert!(index.0 < N, "index out of bounds");
        debug_assert!(index.1 < M, "index out of bounds");
        &self.data[index.0 * M + index.1]
    }
}

impl<const N: usize, const M: usize, R: RingElement> IndexMut<(usize, usize)> for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    /// Returns the `(row, col)` element of the matrix.
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * M + index.1]
    }
}

/// Conversions
impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    /// Converts a matrix over the ring `R` into a matrix over the ring `S`, given that `R` can be
    /// converted to `S`.
    pub fn into_ring<S: RingElement>(self) -> Matrix<N, M, S>
    where
        for<'a> &'a S: RingElementRef<S>,
        for<'a> S: From<&'a R>,
    {
        let mut result: Matrix<N, M, S> = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                result[(r, c)] = S::from(&self[(r, c)]);
            }
        }
        result
    }
}
/// Arithmetic operations

impl<const N: usize, const M: usize, R: RingElement> Mul<&R> for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;

    fn mul(self, other: &R) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = &self[(r, c)] * &other;
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, const K: usize, R: RingElement> Mul<&Matrix<M, K, R>>
    for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, K, R>;

    fn mul(self, other: &Matrix<M, K, R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..K {
                for i in 0..M {
                    out[(r, c)] += &(&self[(r, i)] * &other[(i, c)]);
                }
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> Add<&Matrix<N, M, R>> for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;
    fn add(self, other: &Matrix<N, M, R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = &self[(r, c)] + &other[(r, c)];
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> Sub<&Matrix<N, M, R>> for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;
    fn sub(self, other: &Matrix<N, M, R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = &self[(r, c)] - &other[(r, c)];
            }
        }
        out
    }
}
impl<const N: usize, const M: usize, R: RingElement> Neg for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;
    fn neg(self) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = -&self[(r, c)];
            }
        }
        out
    }
}

/// Norm
impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
    R: NormedRingElement,
{
    pub fn norm(&self) -> u64 {
        let mut worst: u64 = 0;
        for r in 0..N {
            for c in 0..M {
                worst = max(worst, self[(r, c)].norm());
            }
        }
        worst
    }
}

/// Random sampling implementations inherited from the base ring.

impl<const N: usize, const M: usize, R: RingElement> RandUniformSampled for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
    R: RandUniformSampled,
{
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in 0..M {
                result[(i, j)] = R::rand_uniform(rng);
            }
        }
        result
    }
}

impl<const N: usize, const M: usize, R: RingElement> RandZeroOneSampled for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
    R: RandZeroOneSampled,
{
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in 0..M {
                result[(i, j)] = R::rand_zero_one(rng);
            }
        }
        result
    }
}

impl<const N: usize, const M: usize, R: RingElement> RandDiscreteGaussianSampled for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
    R: RandDiscreteGaussianSampled,
{
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..N {
            for j in 0..M {
                result[(i, j)] = R::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(rng);
            }
        }
        result
    }
}

#[cfg(test)]
mod test {
    use crate::math::z_n::*;

    use super::*;

    const N: usize = 2;
    const M: usize = 8;
    const Q: u64 = 11;

    #[test]
    fn zero_matrix_is_correct() {
        let zero: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                assert_eq!(zero[(i, j)], Z_N::zero());
            }
        }
    }

    #[test]
    fn identity_matrix_is_correct() {
        let I: Matrix<M, M, Z_N<Q>> = Matrix::identity();
        for i in 0..M {
            for j in 0..M {
                if i == j {
                    assert_eq!(I[(i, j)], Z_N::one());
                } else {
                    assert_eq!(I[(i, j)], Z_N::zero());
                }
            }
        }
    }

    fn addition_test1() {
        let mut mat: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i, j)] = Z_N::from((i * M + j) as u64);
            }
        }
        let zero: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        assert_eq!(&mat + &zero, mat, "multiplication by identity failed");
    }

    fn addition_test2() {
        let mut mat1: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat1[(i, j)] = Z_N::from((i * M + j) as u64);
            }
        }

        let mut mat2: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i, j)] = Z_N::from((i + 2 * j) as u64);
            }
        }

        let mut mat3: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat3[(i, j)] = Z_N::from((i * (M + 1) + 3 * j) as u64);
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
        let mut mat: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i, j)] = Z_N::from((i * M + j) as u64);
            }
        }
        let I = Matrix::identity();
        assert_eq!(&mat * &I, mat, "multiplication by identity failed");
    }

    fn multiplication_test2() {
        let mut mat1: Matrix<N, N, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..N {
                mat1[(i, j)] = Z_N::from((i * N + j) as u64);
            }
        }

        let mut mat2: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i, j)] = Z_N::from((i * M + j) as u64);
            }
        }

        let mut mat3: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for j in 0..M {
            mat3[(0, j)] = Z_N::from((M + j) as u64);
            mat3[(1, j)] = Z_N::from((3 * M + 5 * j) as u64);
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
        let mut mat: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i, j)] = Z_N::from((i * M + j) as u64);
            }
        }
        let zero: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        assert_eq!(
            &mat + &(-&mat),
            zero,
            "addition by negation doesn't yield zero"
        );
    }

    #[test]
    fn scalar_mult_is_correct() {
        let mut mat1: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat1[(i, j)] = Z_N::from((i * M + j) as u64);
            }
        }

        let mut mat2: Matrix<N, M, Z_N<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i, j)] = Z_N::from((5 * (i * M + j)) as u64);
            }
        }

        let zero: Matrix<N, M, Z_N<Q>> = Matrix::zero();

        assert_eq!(
            &mat1 * &Z_N::zero(),
            zero,
            "multiplication by scalar zero doesn't yield zero"
        );
        assert_eq!(
            &mat1 * &Z_N::one(),
            mat1,
            "multiplication by scalar one doesn't yield itself"
        );
        assert_eq!(&mat1 * &Z_N::from(5_u64), mat2);
    }
}
