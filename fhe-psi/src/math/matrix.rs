//! Matrices over generic rings.

use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::cmp::max;
use std::mem::ManuallyDrop;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

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
#[repr(transparent)]
pub struct Matrix<const N: usize, const M: usize, R: RingElement>
where
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

    /// Copies all of `m` into `self`, starting at `(target_row, target_col)`, by cloning each element.
    pub fn copy_into<const N2: usize, const M2: usize>(
        &mut self,
        m: &Matrix<N2, M2, R>,
        target_row: usize,
        target_col: usize,
    ) {
        self.copy_into_with_len(m, target_row, target_col, N2, M2);
    }

    /// Copies the upper left `row_len` by `col_len` submatrix of `m` into `self`, starting at `(target_row, target_col)`, by cloning each element.
    pub fn copy_into_with_len<const N2: usize, const M2: usize>(
        &mut self,
        m: &Matrix<N2, M2, R>,
        target_row: usize,
        target_col: usize,
        row_len: usize,
        col_len: usize,
    ) {
        debug_assert!(target_row < N, "target_row out of bounds");
        debug_assert!(target_col < M, "target_col out of bounds");
        debug_assert!(
            target_row + row_len <= N,
            "target_row + row_len out of bounds"
        );
        debug_assert!(
            target_col + col_len <= M,
            "target_col + col_len out of bounds"
        );
        debug_assert!(row_len <= N2, "row_len exceeds source matrix dimension");
        debug_assert!(col_len <= M2, "col_len exceeds source matrix dimension");
        for r in 0..row_len {
            for c in 0..col_len {
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

/*
 * Square matrix specific methods.
 */

impl<const N: usize, R: RingElement> Matrix<N, N, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    /// Returns the identity matrix.
    pub fn identity() -> Self {
        let mut out = Matrix::zero();
        for i in 0..N {
            out[(i, i)] = R::one();
        }
        out
    }
}

/*
 * Indexing
 */

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
        debug_assert!(index.0 < N, "index out of bounds");
        debug_assert!(index.1 < M, "index out of bounds");
        &mut self.data[index.0 * M + index.1]
    }
}

impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    pub fn iter_do<F: Fn(&mut R)>(&mut self, f: F) {
        for r in 0..N {
            for c in 0..M {
                f(&mut self[(r, c)]);
            }
        }
    }
}

/// Conversions
impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    /// Converts a matrix over the ring `R` into a matrix over the ring `S`, given that `R` can be
    /// converted to `S`.
    pub fn into_ring<S: RingElement, F: Fn(&R) -> S>(&self, func: F) -> Matrix<N, M, S>
    where
        for<'a> &'a S: RingElementRef<S>,
    {
        let mut result: Matrix<N, M, S> = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                result[(r, c)] = func(&self[(r, c)]);
            }
        }
        result
    }

    ///
    /// Interprets a Matrix of `R` to a Matrix of `S`. This requires `R` and `S` to be compatible as
    /// defined with the `RingCompatible` trait.
    ///
    pub fn convert_ring<S: RingElement>(self) -> Matrix<N, M, S>
    where
        for<'a> &'a S: RingElementRef<S>,
        R: RingCompatible<S>,
    {
        let mut self_clone: ManuallyDrop<Matrix<N, M, R>> = ManuallyDrop::new(self);
        Matrix::<N, M, S> {
            data: unsafe {
                Vec::from_raw_parts(
                    self_clone.data.as_mut_ptr() as *mut S,
                    self_clone.data.len(),
                    self_clone.data.capacity(),
                )
            },
        }
    }
}
/// Arithmetic operations

impl<const N: usize, const M: usize, R: RingElement> Mul<&R> for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;

    /// Multiplies each element of the matrix by `other`.
    fn mul(self, other: &R) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = &self[(r, c)] * other;
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    pub fn mul_iter_do<const K: usize, F: FnMut((usize, usize), &R, &R)>(
        &self,
        other: &Matrix<M, K, R>,
        mut f: F,
    ) {
        for r in 0..N {
            for c in 0..K {
                for i in 0..M {
                    f((r, c), &self[(r, i)], &other[(i, c)]);
                }
            }
        }
    }
}

impl<const N: usize, const M: usize, const K: usize, R: RingElement> Mul<&Matrix<M, K, R>>
    for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, K, R>;

    /// Naive matrix multiplication.
    fn mul(self, other: &Matrix<M, K, R>) -> Self::Output {
        let mut out = Matrix::zero();
        self.mul_iter_do(other, |(r, c), lhs, rhs| {
            out[(r, c)].add_eq_mul(lhs, rhs);
        });
        out
    }
}

impl<const N: usize, const K: usize, R: RingElement> Matrix<N, K, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    pub fn add_eq_mul<const M: usize>(&mut self, a: &Matrix<N, M, R>, b: &Matrix<M, K, R>) {
        for r in 0..N {
            for c in 0..K {
                for i in 0..M {
                    self[(r, c)].add_eq_mul(&a[(r, i)], &b[(i, c)]);
                }
            }
        }
    }
}

impl<const N: usize, const M: usize, R: RingElement> Add<&Matrix<N, M, R>> for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;

    /// Element-wise addition.
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

impl<const N: usize, const M: usize, R: RingElement> AddAssign<&Matrix<N, M, R>> for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    fn add_assign(&mut self, rhs: &Matrix<N, M, R>) {
        for r in 0..N {
            for c in 0..M {
                self[(r, c)] += &rhs[(r, c)];
            }
        }
    }
}

impl<const N: usize, const M: usize, R: RingElement> Sub<&Matrix<N, M, R>> for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;

    /// Element-wise subtraction.
    fn sub(self, other: &Matrix<N, M, R>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = &self[(r, c)] - &other[(r, c)]
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement> SubAssign<&Matrix<N, M, R>> for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    fn sub_assign(&mut self, rhs: &Matrix<N, M, R>) {
        for r in 0..N {
            for c in 0..M {
                self[(r, c)] -= &rhs[(r, c)];
            }
        }
    }
}

impl<const N: usize, const M: usize, R: RingElement> Neg for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Output = Matrix<N, M, R>;

    /// Element-wise negation.
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
    /// Element-wise uniform random sampling.
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
    /// Element-wise random 0/1 sampling.
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
    /// Element-wise random discrete gaussian sampling.
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
    use crate::math::int_mod::*;

    use super::*;

    const N: usize = 2;
    const M: usize = 8;
    const Q: u64 = 11;

    #[test]
    fn zero_matrix_is_correct() {
        let zero: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                assert_eq!(zero[(i, j)], IntMod::zero());
            }
        }
    }

    #[test]
    fn identity_matrix_is_correct() {
        let ident: Matrix<M, M, IntMod<Q>> = Matrix::identity();
        for i in 0..M {
            for j in 0..M {
                if i == j {
                    assert_eq!(ident[(i, j)], IntMod::one());
                } else {
                    assert_eq!(ident[(i, j)], IntMod::zero());
                }
            }
        }
    }

    fn addition_test1() {
        let mut mat: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i, j)] = IntMod::from((i * M + j) as u64);
            }
        }
        let zero: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        assert_eq!(&mat + &zero, mat, "multiplication by identity failed");
    }

    fn addition_test2() {
        let mut mat1: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat1[(i, j)] = IntMod::from((i * M + j) as u64);
            }
        }

        let mut mat2: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i, j)] = IntMod::from((i + 2 * j) as u64);
            }
        }

        let mut mat3: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat3[(i, j)] = IntMod::from((i * (M + 1) + 3 * j) as u64);
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
        let mut mat: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i, j)] = IntMod::from((i * M + j) as u64);
            }
        }
        let ident = Matrix::identity();
        assert_eq!(&mat * &ident, mat, "multiplication by identity failed");
    }

    fn multiplication_test2() {
        let mut mat1: Matrix<N, N, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..N {
                mat1[(i, j)] = IntMod::from((i * N + j) as u64);
            }
        }

        let mut mat2: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i, j)] = IntMod::from((i * M + j) as u64);
            }
        }

        let mut mat3: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for j in 0..M {
            mat3[(0, j)] = IntMod::from((M + j) as u64);
            mat3[(1, j)] = IntMod::from((3 * M + 5 * j) as u64);
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
        let mut mat: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat[(i, j)] = IntMod::from((i * M + j) as u64);
            }
        }
        let zero: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        assert_eq!(
            &mat + &(-&mat),
            zero,
            "addition by negation doesn't yield zero"
        );
    }

    #[test]
    fn scalar_mult_is_correct() {
        let mut mat1: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat1[(i, j)] = IntMod::from((i * M + j) as u64);
            }
        }

        let mut mat2: Matrix<N, M, IntMod<Q>> = Matrix::zero();
        for i in 0..N {
            for j in 0..M {
                mat2[(i, j)] = IntMod::from((5 * (i * M + j)) as u64);
            }
        }

        let zero: Matrix<N, M, IntMod<Q>> = Matrix::zero();

        assert_eq!(
            &mat1 * &IntMod::zero(),
            zero,
            "multiplication by scalar zero doesn't yield zero"
        );
        assert_eq!(
            &mat1 * &IntMod::one(),
            mat1,
            "multiplication by scalar one doesn't yield itself"
        );
        assert_eq!(&mat1 * &IntMod::from(5_u64), mat2);
    }
}
