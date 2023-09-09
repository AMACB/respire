//! The ring `Z_n` of integers modulo `n`.

use crate::math::discrete_gaussian::DiscreteGaussian;
use crate::math::gadget::RingElementDecomposable;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::cmp::min;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Integers modulo `N` with overloaded modular arithmetic operation (`+`, `-`, `*`, unary `-`), and
/// several other utility methods.
///
/// Internally, elements of this type are represented as a u64 `a` in reduced form: `0 <= a < N`.
/// Thus `Z_N` is `Clone`. Furthermore the non-inplace operations are implemented in addition to the
/// inplace versions required by [`RingElement`].
///
/// The assumption as noted above is that `a` must be in reduced form at all times. The public
/// conversion methods accomplish this by doing % every time. However, there are times when it is
/// known that a certain value is already reduced, in which case the wrapper type NoReduce can
/// be used for the conversion. It is the caller's responsibility to ensure such a NoReduce is
/// indeed reduced already.
///
/// The behavior when `N < 2` is not defined.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct IntMod<const N: u64> {
    a: u64,
}

#[repr(transparent)]
pub struct NoReduce(pub u64);

/// Conversions

impl<const N: u64> From<IntMod<N>> for u64 {
    /// Extracts the reduced form modulo `N`.
    fn from(a: IntMod<N>) -> Self {
        a.a
    }
}

impl<const N: u64> From<u64> for IntMod<N> {
    /// Converts u64 to Z_N by modular reduction.
    fn from(a: u64) -> Self {
        // Optimized special case that is slower than just using %
        // if N == 268369921 {
        //     if a < N {
        //         return Z_N {
        //             a
        //         }
        //     }
        //     // 268369921 = 2^28 - 2^16 + 1
        //     const LSB28_MASK: u64 = (1 << 28) - 1;
        //     let low = a & LSB28_MASK;
        //     let mid = (a >> 28) & LSB28_MASK;
        //     let high = (a >> 56) & LSB28_MASK;
        //     // a = high * 2^56 + mid * 2^28 + low
        //     //   = high * (2^16-1)^2 + mid * (2^16-1) + low
        //     //   = high * (2^32 - 2^17 + 1) + mid * 2^16 - mid + low
        //     let a = (high << 32) - (high << 17) + high + (mid << 16) - mid + low;
        //     debug_assert!(a < (1 << 45));
        //
        //     let low = a & LSB28_MASK;
        //     let mid = (a >> 28) & LSB28_MASK;
        //     // a = mid * 2^28 + low
        //     //   = mid * (2^16-1) + low
        //     let a = (mid << 16) - mid + low;
        //     debug_assert!(a < (1 << 34));
        //
        //     let low = a & LSB28_MASK;
        //     let mid = (a >> 28) & LSB28_MASK;
        //     // Same thing, but now both terms are small
        //     return Z_N::<N> {
        //         a: (mid << 16) - mid
        //     } + Z_N::<N> {
        //         a: low
        //     };
        // }

        IntMod { a: a % N }
    }
}

impl<const N: u64> From<NoReduce> for IntMod<N> {
    fn from(nr: NoReduce) -> Self {
        IntMod { a: nr.0 }
    }
}

impl<const N: u64> From<i64> for IntMod<N> {
    /// Converts i64 to Z_N by modular reduction.
    fn from(a: i64) -> Self {
        if a < 0 {
            -IntMod::from(-a as u64)
        } else {
            IntMod::from(a as u64)
        }
    }
}

impl<const N: u64> From<IntMod<N>> for i64 {
    /// Converts from Z_N to the i64 of the smallest absolute value with the correct remainder. This
    /// is useful for operations that want to round towards zero.
    fn from(a: IntMod<N>) -> Self {
        if a.a <= (N - 1) / 2 {
            a.a as i64
        } else {
            -((N - a.a) as i64)
        }
    }
}

/// Math operations on owned `Z_N<N>`, including [`RingElement`] implementation.

impl<const N: u64> RingElement for IntMod<N> {
    fn zero() -> Self {
        0_u64.into()
    }
    fn one() -> Self {
        1_u64.into()
    }
}

impl<const N: u64> Add for IntMod<N> {
    type Output = IntMod<N>;
    fn add(self, rhs: Self) -> Self::Output {
        if N < (1 << 63) {
            let result = self.a + rhs.a;
            return if result >= N {
                NoReduce(result - N).into()
            } else {
                NoReduce(result).into()
            };
        }

        (((self.a as u128 + rhs.a as u128) % (N as u128)) as u64).into()
    }
}

impl<const N: u64> AddAssign for IntMod<N> {
    fn add_assign(&mut self, rhs: Self) {
        self.a = (self.clone() + rhs).a;
    }
}

impl<const N: u64> Mul for IntMod<N> {
    type Output = IntMod<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        if N < (1 << 32) {
            return (self.a * rhs.a).into();
        }

        (((self.a as u128 * rhs.a as u128) % (N as u128)) as u64).into()
    }
}

impl<const N: u64> MulAssign for IntMod<N> {
    fn mul_assign(&mut self, rhs: Self) {
        self.a = (self.clone() * rhs).a;
    }
}

impl<const N: u64> Sub for IntMod<N> {
    type Output = IntMod<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<const N: u64> SubAssign for IntMod<N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a = (self.clone() - rhs).a
    }
}

impl<const N: u64> Neg for IntMod<N> {
    type Output = IntMod<N>;
    fn neg(self) -> Self::Output {
        if self.a == 0 {
            self
        } else {
            NoReduce(N - self.a).into()
        }
    }
}

impl<const NN: u64, const BASE: u64, const LEN: usize> RingElementDecomposable<BASE, LEN>
    for IntMod<NN>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let mut a = self.a;
        for k in 0..LEN {
            mat[(i + k, j)] = (a % BASE).into();
            a /= BASE;
        }
    }
}

impl<const N: usize, const M: usize, R: RingElement, const P: u64> Mul<IntMod<P>>
    for &Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R> + Mul<IntMod<P>, Output = R>,
{
    type Output = Matrix<N, M, R>;

    /// Multiplies each element of the matrix by `other`.
    fn mul(self, other: IntMod<P>) -> Self::Output {
        let mut out = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                out[(r, c)] = &self[(r, c)] * other;
            }
        }
        out
    }
}

impl<const N: usize, const M: usize, R: RingElement, const P: u64> MulAssign<IntMod<P>>
    for Matrix<N, M, R>
where
    for<'a> &'a R: RingElementRef<R>,
    R: MulAssign<IntMod<P>>,
{
    /// Multiplies each element of the matrix by `other`.
    fn mul_assign(&mut self, other: IntMod<P>) {
        for r in 0..N {
            for c in 0..M {
                self[(r, c)] *= other;
            }
        }
    }
}
/// Misc

impl<const N: u64> IntMod<N> {
    /// Maps `Z_N` into `Z_M` by sending `0 <= a < N` to `a * floor(M / N)`. We require `N <= M`.
    pub fn scale_up_into<const M: u64>(self) -> IntMod<M> {
        assert!(N <= M);
        let ratio = M / N;
        (u64::from(self) * ratio).into()
    }

    /// Maps `Z_N` into `Z_M` by the inclusion map, i.e. `0 <= a < N` gets sent to `a`. We require
    /// `N <= M`.
    pub fn include_into<const M: u64>(self) -> IntMod<M> {
        assert!(N <= M);
        u64::from(self).into()
    }

    /// Maps `Z_N` into `Z_M` by projecting. This only makes sense if `M` divides `N`.
    pub fn project_into<const M: u64>(self) -> IntMod<M> {
        assert_eq!(N % M, 0);
        u64::from(self).into()
    }

    /// Maps `Z_N` into `Z_M` by rounding `0 <= a < N` to the nearest multiple of `N / M`, and
    /// dividing. This function acts like an inverse of `scale_up_into`, with tolerance to additive noise. We require `N >= M`.
    pub fn round_down_into<const M: u64>(self) -> IntMod<M> {
        assert!(N >= M);
        let ratio = N / M;
        ((u64::from(self) + ratio / 2) / ratio).into()
    }
}

/// Formatting

impl<const N: u64> fmt::Debug for IntMod<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a <= 3 * N / 4 {
            write!(f, "{}", self.a)
        } else {
            write!(f, "-{}", N - self.a)
        }
    }
}

/// Math operations on borrows `&Z_N<N>`, including [`RingElementRef`] implementation.

impl<const N: u64> RingElementRef<IntMod<N>> for &IntMod<N> {}

impl<const N: u64> Neg for &IntMod<N> {
    type Output = IntMod<N>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<const N: u64> Add for &IntMod<N> {
    type Output = IntMod<N>;
    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<const N: u64> AddAssign<&IntMod<N>> for IntMod<N> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a = (self.clone() + rhs.clone()).a
    }
}

impl<const N: u64> Sub for &IntMod<N> {
    type Output = IntMod<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<const N: u64> SubAssign<&IntMod<N>> for IntMod<N> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.a = (self.clone() - rhs.clone()).a
    }
}

impl<const N: u64> Mul for &IntMod<N> {
    type Output = IntMod<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<const N: u64> MulAssign<&IntMod<N>> for IntMod<N> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.a = (self.clone() * rhs.clone()).a
    }
}

/// Random sampling

impl<const N: u64> RandUniformSampled for IntMod<N> {
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        rng.gen_range(0..N).into()
    }
}

impl<const N: u64> RandZeroOneSampled for IntMod<N> {
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        rng.gen_range(0..2_u64).into()
    }
}

impl<const N: u64> RandDiscreteGaussianSampled for IntMod<N> {
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        DiscreteGaussian::sample::<_, NOISE_WIDTH_MILLIONTHS>(rng).into()
    }
}

/// Other methods
impl<const N: u64> IntMod<N> {
    pub fn norm(&self) -> u64 {
        let pos: u64 = u64::from(*self);
        let neg: u64 = u64::from(-*self);
        min(pos, neg)
    }

    pub fn pow(&self, mut e: u64) -> IntMod<N> {
        let mut val = self.clone();
        let mut res = IntMod::one();
        while e > 0 {
            if (e & 1) == 1 {
                res *= val;
            }
            e >>= 1;
            val *= val;
        }
        return res;
    }

    // TODO: this is not efficient, I think euclidean is faster
    // this also assumes N is prime
    pub fn inverse(&self) -> IntMod<N> {
        return self.pow(N - 2);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_into() {
        type Z31 = IntMod<31>;
        type ZBIG = IntMod<{ u64::MAX - 1 }>;

        let a: Z31 = 0_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z31 = 1_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z31 = 30_u64.into();
        assert_eq!(30_u64, a.into());

        let a: Z31 = 31_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z31 = 32_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z31 = ((31 * 439885 + 4) as u64).into();
        assert_eq!(4_u64, a.into());

        let a: ZBIG = (u64::MAX - 1).into();
        assert_eq!(0_u64, a.into());

        let a: ZBIG = u64::MAX.into();
        assert_eq!(1_u64, a.into());

        let a: i64 = Z31::from(0_u64).into();
        assert_eq!(0_i64, a);

        let a: i64 = Z31::from(15_u64).into();
        assert_eq!(15_i64, a);

        let a: i64 = Z31::from(16_u64).into();
        assert_eq!(-15_i64, a);

        let a: i64 = Z31::from(30_u64).into();
        assert_eq!(-1_i64, a);

        let half = (u64::MAX - 1) / 2;
        let a: i64 = ZBIG::from(half - 1).into();
        assert_eq!((half - 1) as i64, a);
        let a: i64 = ZBIG::from(half).into();
        assert_eq!(-(half as i64), a);
    }

    #[test]
    fn test_ops() {
        type Z31 = IntMod<31>;
        type ZBIG = IntMod<{ u64::MAX - 1 }>;

        let a: Z31 = 10_u64.into();
        let b: Z31 = -a;
        assert_eq!(21_u64, b.into());

        let a: Z31 = 0_u64.into();
        let b: Z31 = -a;
        assert_eq!(a, b);

        let mut a: Z31 = 23_u64.into();
        let b: Z31 = 24_u64.into();
        assert_eq!(16_u64, (a + b).into());
        a += Z31::from(24_u64);
        assert_eq!(16_u64, a.into());

        let mut a: Z31 = 23_u64.into();
        let b: Z31 = 24_u64.into();
        assert_eq!(30_u64, (a - b).into());
        a -= Z31::from(24_u64);
        assert_eq!(30_u64, a.into());

        let mut a: Z31 = 16_u64.into();
        let b: Z31 = 3_u64.into();
        assert_eq!(17_u64, (a * b).into());
        a *= Z31::from(3_u64);
        assert_eq!(17_u64, a.into());

        let a: ZBIG = 10_u64.into();
        let b: ZBIG = -a;
        assert_eq!(u64::MAX - 10 - 1, b.into());

        let mut a: ZBIG = (u64::MAX - 50005).into();
        let b: ZBIG = 60006_u64.into();
        assert_eq!(10002_u64, (a + b).into());
        a += ZBIG::from(60006_u64);
        assert_eq!(10002_u64, a.into());

        let mut a: ZBIG = 50005_u64.into();
        let b: ZBIG = 70007_u64.into();
        assert_eq!(u64::MAX - 20003, (a - b).into());
        a -= ZBIG::from(70007_u64);
        assert_eq!(u64::MAX - 20003, a.into());

        let mut a: ZBIG = (u64::MAX - 1 - 1984).into();
        let b: ZBIG = (u64::MAX - 1 - 3968).into();
        assert_eq!(7872512_u64, (a * b).into());
        a *= ZBIG::from(u64::MAX - 1 - 3968);
        assert_eq!(7872512_u64, a.into());

        let mut a: ZBIG = (u64::MAX - 1 - 1984).into();
        let b: ZBIG = 3968_u64.into();
        assert_eq!(u64::MAX - 1 - 7872512, (a * b).into());
        a *= ZBIG::from(3968_u64);
        assert_eq!(u64::MAX - 1 - 7872512, a.into());
    }

    #[test]
    fn test_scale_round() {
        type Z31 = IntMod<31>;
        type Z1000 = IntMod<1000>;

        const MAX_ERROR: i64 = ((1000 / 31) - 1) / 2;
        assert_eq!(MAX_ERROR, 15);
        for i in 0_u64..31 {
            for e in -MAX_ERROR..=MAX_ERROR {
                let orig: Z31 = Z31::from(i);
                let scaled_with_error: Z1000 = orig.scale_up_into() + e.into();
                let recovered: Z31 = scaled_with_error.round_down_into();
                assert_eq!(
                    orig,
                    recovered,
                    "orig = {}, error = {}, scaled = {}, recovered = {}",
                    i,
                    e,
                    u64::from(scaled_with_error),
                    u64::from(recovered)
                );
            }
        }
    }

    #[test]
    fn test_norm() {
        type Z31 = IntMod<31>;
        type ZBIG = IntMod<{ u64::MAX - 1 }>;

        let zero: Z31 = 0_u64.into();
        let one_pos: Z31 = 1_u64.into();
        let one_neg: Z31 = 30_u64.into();
        let two_pos: Z31 = 2_u64.into();
        let two_neg: Z31 = 29_u64.into();
        let fifteen_pos: Z31 = 15_u64.into();
        let fifteen_neg: Z31 = 16_u64.into();
        assert_eq!(zero.norm(), 0);
        assert_eq!(one_pos.norm(), 1);
        assert_eq!(one_neg.norm(), 1);
        assert_eq!(two_pos.norm(), 2);
        assert_eq!(two_neg.norm(), 2);
        assert_eq!(fifteen_pos.norm(), 15);
        assert_eq!(fifteen_neg.norm(), 15);

        let one_neg_big: ZBIG = (u64::MAX - 2).into();
        assert_eq!(one_neg_big.norm(), 1);
    }

    // #[test]
    // fn test_268369921() {
    //     type Z_Q = Z_N<268369921>;
    //     assert_eq!(Z_Q::from(6365999520220238746_u64), Z_Q::from(4475005_u64));
    //     assert_eq!(
    //         Z_Q::from(18446744073709551615_u64),
    //         Z_Q::from(234877183_u64)
    //     );
    //     assert_eq!(Z_Q::from(3615920400745237573_u64), Z_Q::from(64263131_u64));
    // }
}
