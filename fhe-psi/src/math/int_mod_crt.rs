//! The ring `Z_n` of integers modulo `n = n_1 * n_2`, internally represented by its residues modulo `n_1` and `n_2`.

use crate::math::discrete_gaussian::DiscreteGaussian;
use crate::math::gadget::RingElementDecomposable;
use crate::math::int_mod::IntMod;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::cmp::min;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO: documentation
// TODO: somewhat unsatisfactory -- can't generalize to N = N_1 * ... * N_k
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct IntModCRT<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> {
    a1: IntMod<N1>,
    a2: IntMod<N2>,
}

/// Conversions

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    From<IntModCRT<N1, N2, N1_INV, N2_INV>> for u64
{
    /// Reconstructs the reduced form modulo `N`.
    fn from(a: IntModCRT<N1, N2, N1_INV, N2_INV>) -> Self {
        let a1: u128 = u64::from(a.a1) as u128;
        let a2: u128 = u64::from(a.a2) as u128;
        let n1: u128 = N1.into();
        let n2: u128 = N2.into();
        let n1_inv: u128 = N1_INV.into();
        let n2_inv: u128 = N2_INV.into();
        ((n2_inv * n2 * a1 + n1_inv * n1 * a2) % (n1 * n2)) as u64
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    From<(IntMod<N1>, IntMod<N2>)> for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn from(a: (IntMod<N1>, IntMod<N2>)) -> Self {
        IntModCRT { a1: a.0, a2: a.1 }
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> From<u64>
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    /// Converts u64 to IntModCRT by modular reductions.
    fn from(a: u64) -> Self {
        IntModCRT {
            a1: a.into(),
            a2: a.into(),
        }
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> From<i64>
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    /// Converts i64 to IntModCRT by modular reductions.
    fn from(a: i64) -> Self {
        if a < 0 {
            -IntModCRT::from(-a as u64)
        } else {
            IntModCRT::from(a as u64)
        }
    }
}

/// Math operations on owned `IntModCRT<N1, N2, N1_INV, N2_INV>`, including [`RingElement`] implementation.

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> RingElement
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn zero() -> Self {
        0_u64.into()
    }
    fn one() -> Self {
        1_u64.into()
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Add
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn add(self, rhs: Self) -> Self::Output {
        IntModCRT {
            a1: self.a1 + rhs.a1,
            a2: self.a2 + rhs.a2,
        }
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> AddAssign
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn add_assign(&mut self, rhs: Self) {
        self.a1 += rhs.a1;
        self.a2 += rhs.a2;
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Mul
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn mul(self, rhs: Self) -> Self::Output {
        IntModCRT {
            a1: self.a1 * rhs.a1,
            a2: self.a2 * rhs.a2,
        }
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> MulAssign
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn mul_assign(&mut self, rhs: Self) {
        self.a1 *= rhs.a1;
        self.a2 *= rhs.a2;
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Sub
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> SubAssign
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn sub_assign(&mut self, rhs: Self) {
        self.a1 -= rhs.a1;
        self.a2 -= rhs.a2;
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Neg
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn neg(mut self) -> Self::Output {
        self.a1 = -self.a1;
        self.a2 = -self.a2;
        self
    }
}

impl<
        const N1: u64,
        const N2: u64,
        const N1_INV: u64,
        const N2_INV: u64,
        const BASE: u64,
        const LEN: usize,
    > RingElementDecomposable<BASE, LEN> for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let mut a: u64 = self.clone().into();
        for k in 0..LEN {
            mat[(i + k, j)] = (a % BASE).into();
            a /= BASE;
        }
    }
}

/// Misc

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    IntModCRT<N1, N2, N1_INV, N2_INV>
{
    /// Maps `Z_N` into `Z_M` by rounding `0 <= a < N` to the nearest multiple of `N / M`, and
    /// dividing. This function acts like an inverse of `scale_up_into`, with tolerance to additive noise. We require `N >= M`.
    pub fn round_down_into<const M: u64>(self) -> IntMod<M> {
        assert!(N1 * N2 >= M);
        let ratio = N1 * N2 / M;
        ((u64::from(self) + ratio / 2) / ratio).into()
    }
}

/// Formatting

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> fmt::Debug
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.a1, self.a2)
    }
}

/// Math operations on borrows `&IntModCRT<N1, N2, N1_INV, N2_INV>`, including [`RingElementRef`] implementation.

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    RingElementRef<IntModCRT<N1, N2, N1_INV, N2_INV>> for &IntModCRT<N1, N2, N1_INV, N2_INV>
{
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Neg
    for &IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Add
    for &IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    AddAssign<&IntModCRT<N1, N2, N1_INV, N2_INV>> for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn add_assign(&mut self, rhs: &Self) {
        self.a1 += rhs.a1;
        self.a2 += rhs.a2;
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Sub
    for &IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    SubAssign<&IntModCRT<N1, N2, N1_INV, N2_INV>> for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn sub_assign(&mut self, rhs: &Self) {
        self.a1 -= rhs.a1;
        self.a2 -= rhs.a2;
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Mul
    for &IntModCRT<N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCRT<N1, N2, N1_INV, N2_INV>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    MulAssign<&IntModCRT<N1, N2, N1_INV, N2_INV>> for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn mul_assign(&mut self, rhs: &Self) {
        self.a1 *= rhs.a1;
        self.a2 *= rhs.a2;
    }
}

/// Random sampling

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> RandUniformSampled
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        IntModCRT {
            a1: rng.gen_range(0..N1).into(),
            a2: rng.gen_range(0..N2).into(),
        }
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> RandZeroOneSampled
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        rng.gen_range(0..2_u64).into()
    }
}

impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> RandDiscreteGaussianSampled
    for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        DiscreteGaussian::sample::<_, NOISE_WIDTH_MILLIONTHS>(rng).into()
    }
}

/// Other methods
impl<const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    IntModCRT<N1, N2, N1_INV, N2_INV>
{
    pub fn norm(&self) -> u64 {
        let pos: u64 = u64::from(*self);
        let neg: u64 = u64::from(-*self);
        min(pos, neg)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_into() {
        type Z55 = IntModCRT<5, 11, 9, 1>;

        let a: Z55 = 0_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z55 = 1_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z55 = 54_u64.into();
        assert_eq!(54_u64, a.into());

        let a: Z55 = 55_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z55 = 56_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z55 = ((55 * 439885 + 16) as u64).into();
        assert_eq!(16_u64, a.into());
    }

    #[test]
    fn test_ops() {
        type Z55 = IntModCRT<5, 11, 9, 1>;

        let a: Z55 = 21_u64.into();
        let b: Z55 = -a;
        assert_eq!(34_u64, b.into());

        let a: Z55 = 0_u64.into();
        let b: Z55 = -a;
        assert_eq!(a, b);

        let mut a: Z55 = 23_u64.into();
        let b: Z55 = 45_u64.into();
        assert_eq!(13_u64, (a + b).into());
        a += Z55::from(45_u64);
        assert_eq!(13_u64, a.into());

        let mut a: Z55 = 23_u64.into();
        let b: Z55 = 45_u64.into();
        assert_eq!(33_u64, (a - b).into());
        a -= Z55::from(45_u64);
        assert_eq!(33_u64, a.into());

        let mut a: Z55 = 16_u64.into();
        let b: Z55 = 4_u64.into();
        assert_eq!(9_u64, (a * b).into());
        a *= Z55::from(4_u64);
        assert_eq!(9_u64, a.into());
    }

    #[test]
    fn test_norm() {
        type Z55 = IntModCRT<5, 11, 9, 1>;

        let zero: Z55 = 0_u64.into();
        let one_pos: Z55 = 1_u64.into();
        let one_neg: Z55 = 54_u64.into();
        let two_pos: Z55 = 2_u64.into();
        let two_neg: Z55 = 53_u64.into();
        let twentyseven_pos: Z55 = 27_u64.into();
        let twentyseven_neg: Z55 = 28_u64.into();
        assert_eq!(zero.norm(), 0);
        assert_eq!(one_pos.norm(), 1);
        assert_eq!(one_neg.norm(), 1);
        assert_eq!(two_pos.norm(), 2);
        assert_eq!(two_neg.norm(), 2);
        assert_eq!(twentyseven_pos.norm(), 27);
        assert_eq!(twentyseven_neg.norm(), 27);
    }
}
