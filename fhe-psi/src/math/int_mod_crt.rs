//! The ring `Z_n` of integers modulo `n = n_1 * n_2`, internally represented by its residues modulo `n_1` and `n_2`.

use crate::math::discrete_gaussian::DiscreteGaussian;
use crate::math::gadget::{IntModDecomposition, RingElementDecomposable};
use crate::math::int_mod::IntMod;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use crate::math::utils::mod_inverse;
use rand::Rng;
use std::cmp::min;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO: documentation
// TODO: somewhat unsatisfactory -- can't generalize to N = N_1 * ... * N_k
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct IntModCRT<const N1: u64, const N2: u64> {
    pub proj1: IntMod<N1>,
    pub proj2: IntMod<N2>,
}

impl<const N1: u64, const N2: u64> IntModCRT<N1, N2> {
    pub const N1_INV: u64 = mod_inverse(N1, N2);
    pub const N2_INV: u64 = mod_inverse(N2, N1);
}

/// Conversions

impl<const N1: u64, const N2: u64> From<IntModCRT<N1, N2>> for u64 {
    /// Reconstructs the reduced form modulo `N`.
    fn from(a: IntModCRT<N1, N2>) -> Self {
        // u64 arithmetic only when N1, N2 are 32 bit
        if N1 < (1u64 << 32) && N2 < (1u64 << 32) {
            let a1 = u64::from(a.proj1);
            let a2 = u64::from(a.proj2);
            (((IntModCRT::<N1, N2>::N2_INV * a1) % N1) * N2
                + ((IntModCRT::<N1, N2>::N1_INV * a2) % N2) * N1)
                % (N1 * N2)
        } else {
            let a1: u128 = u64::from(a.proj1) as u128;
            let a2: u128 = u64::from(a.proj2) as u128;
            let n1: u128 = N1.into();
            let n2: u128 = N2.into();
            let n1_inv: u128 = IntModCRT::<N1, N2>::N1_INV.into();
            let n2_inv: u128 = IntModCRT::<N1, N2>::N2_INV.into();
            ((n2_inv * n2 * a1 + n1_inv * n1 * a2) % (n1 * n2)) as u64
        }
    }
}

impl<const N1: u64, const N2: u64> From<(IntMod<N1>, IntMod<N2>)> for IntModCRT<N1, N2> {
    fn from(a: (IntMod<N1>, IntMod<N2>)) -> Self {
        IntModCRT {
            proj1: a.0,
            proj2: a.1,
        }
    }
}

impl<const N1: u64, const N2: u64> From<u64> for IntModCRT<N1, N2> {
    /// Converts u64 to IntModCRT by modular reductions.
    fn from(a: u64) -> Self {
        IntModCRT {
            proj1: a.into(),
            proj2: a.into(),
        }
    }
}

impl<const N1: u64, const N2: u64> From<i64> for IntModCRT<N1, N2> {
    /// Converts i64 to IntModCRT by modular reductions.
    fn from(a: i64) -> Self {
        if a < 0 {
            -IntModCRT::from(-a as u64)
        } else {
            IntModCRT::from(a as u64)
        }
    }
}

/// Math operations on owned `IntModCRT<N1, N2>`, including [`RingElement`] implementation.

impl<const N1: u64, const N2: u64> RingElement for IntModCRT<N1, N2> {
    fn zero() -> Self {
        0_u64.into()
    }
    fn one() -> Self {
        1_u64.into()
    }
}

impl<const N1: u64, const N2: u64> Add for IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn add(self, rhs: Self) -> Self::Output {
        IntModCRT {
            proj1: self.proj1 + rhs.proj1,
            proj2: self.proj2 + rhs.proj2,
        }
    }
}

impl<const N1: u64, const N2: u64> AddAssign for IntModCRT<N1, N2> {
    fn add_assign(&mut self, rhs: Self) {
        self.proj1 += rhs.proj1;
        self.proj2 += rhs.proj2;
    }
}

impl<const N1: u64, const N2: u64> Mul for IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn mul(self, rhs: Self) -> Self::Output {
        IntModCRT {
            proj1: self.proj1 * rhs.proj1,
            proj2: self.proj2 * rhs.proj2,
        }
    }
}

impl<const N1: u64, const N2: u64> MulAssign for IntModCRT<N1, N2> {
    fn mul_assign(&mut self, rhs: Self) {
        self.proj1 *= rhs.proj1;
        self.proj2 *= rhs.proj2;
    }
}

impl<const N1: u64, const N2: u64> Sub for IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<const N1: u64, const N2: u64> SubAssign for IntModCRT<N1, N2> {
    fn sub_assign(&mut self, rhs: Self) {
        self.proj1 -= rhs.proj1;
        self.proj2 -= rhs.proj2;
    }
}

impl<const N1: u64, const N2: u64> Neg for IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn neg(mut self) -> Self::Output {
        self.proj1 = -self.proj1;
        self.proj2 = -self.proj2;
        self
    }
}

impl<const N1: u64, const N2: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for IntModCRT<N1, N2>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let decomp = IntModDecomposition::<BASE, LEN>::new(u64::from(*self), N1 * N2);
        for (k, u) in decomp.enumerate() {
            mat[(i + k, j)] = Self::from(u);
        }
    }
}

/// Misc

impl<const N1: u64, const N2: u64> IntModCRT<N1, N2> {
    /// Maps `Z_N` into `Z_M` by rounding `0 <= a < N` to the nearest multiple of `N / M`, and
    /// dividing. This function acts like an inverse of `scale_up_into`, with tolerance to additive noise. We require `N >= M`.
    pub fn round_down_into<const M: u64>(self) -> IntMod<M> {
        assert!(N1 * N2 >= M);
        let ratio = N1 * N2 / M;
        ((u64::from(self) + ratio / 2) / ratio).into()
    }
}

/// Formatting

impl<const N1: u64, const N2: u64> fmt::Debug for IntModCRT<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.proj1, self.proj2)
    }
}

/// Math operations on borrows `&IntModCRT<N1, N2>`, including [`RingElementRef`] implementation.

impl<const N1: u64, const N2: u64> RingElementRef<IntModCRT<N1, N2>> for &IntModCRT<N1, N2> {}

impl<const N1: u64, const N2: u64> Neg for &IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn neg(self) -> Self::Output {
        -*self
    }
}

impl<const N1: u64, const N2: u64> Add for &IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<const N1: u64, const N2: u64> AddAssign<&IntModCRT<N1, N2>> for IntModCRT<N1, N2> {
    fn add_assign(&mut self, rhs: &Self) {
        self.proj1 += rhs.proj1;
        self.proj2 += rhs.proj2;
    }
}

impl<const N1: u64, const N2: u64> Sub for &IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<const N1: u64, const N2: u64> SubAssign<&IntModCRT<N1, N2>> for IntModCRT<N1, N2> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.proj1 -= rhs.proj1;
        self.proj2 -= rhs.proj2;
    }
}

impl<const N1: u64, const N2: u64> Mul for &IntModCRT<N1, N2> {
    type Output = IntModCRT<N1, N2>;
    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<const N1: u64, const N2: u64> MulAssign<&IntModCRT<N1, N2>> for IntModCRT<N1, N2> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.proj1 *= rhs.proj1;
        self.proj2 *= rhs.proj2;
    }
}

/// Random sampling

impl<const N1: u64, const N2: u64> RandUniformSampled for IntModCRT<N1, N2> {
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        IntModCRT {
            proj1: IntMod::rand_uniform(rng),
            proj2: IntMod::rand_uniform(rng),
        }
    }
}

impl<const N1: u64, const N2: u64> RandZeroOneSampled for IntModCRT<N1, N2> {
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        rng.gen_range(0..2_u64).into()
    }
}

impl<const N1: u64, const N2: u64> RandDiscreteGaussianSampled for IntModCRT<N1, N2> {
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        DiscreteGaussian::sample::<_, NOISE_WIDTH_MILLIONTHS>(rng).into()
    }
}

/// Other methods
impl<const N1: u64, const N2: u64> IntModCRT<N1, N2> {
    pub fn norm(&self) -> u64 {
        let pos: u64 = u64::from(*self);
        let neg: u64 = u64::from(-*self);
        min(pos, neg)
    }
}

unsafe impl<const N1: u64, const N2: u64, const M1: u64, const M2: u64>
    RingCompatible<IntModCRT<M1, M2>> for IntModCRT<N1, N2>
{
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_into() {
        type Z55 = IntModCRT<5, 11>;

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
        type Z55 = IntModCRT<5, 11>;

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
        type Z55 = IntModCRT<5, 11>;

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
