//! The ring `Z_n` of integers modulo `n = n_1 * n_2`, internally represented by its residues modulo `n_1` and `n_2`.

use crate::fhe::discrete_gaussian::DiscreteGaussian;
use crate::fhe::gadget::RingElementDecomposable;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use crate::math::z_n::Z_N;
use rand::Rng;
use std::cmp::min;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO: documentation
// TODO: somewhat unsatisfactory -- can't generalize to N = N_1 * ... * N_k
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct Z_N_CRT<const N1: u64, const N2: u64> {
    a1: Z_N<N1>,
    a2: Z_N<N2>,
}

/// Conversions

impl<const N1: u64, const N2: u64> From<Z_N_CRT<N1, N2>> for u64 {
    /// Reconstructs the reduced form modulo `N`.
    fn from(a: Z_N_CRT<N1, N2>) -> Self {
        // This should be statically cached / type parametered, since
        // this conversion is likely used often enough that even the
        // current extra log factor will be very noticeable.

        let n1_inv: u128 = u64::from(Z_N::<N2>::from(N1).inverse()) as u128;
        let n2_inv: u128 = u64::from(Z_N::<N1>::from(N2).inverse()) as u128;
        let a1: u128 = u64::from(a.a1) as u128;
        let a2: u128 = u64::from(a.a2) as u128;
        let n1: u128 = N1.into();
        let n2: u128 = N2.into();
        ((n2_inv * n2 * a1 + n1_inv * n1 * a2) % (n1 * n2)) as u64
    }
}

impl<const N1: u64, const N2: u64> From<u64> for Z_N_CRT<N1, N2> {
    /// Converts u64 to Z_N_CRT by modular reductions.
    fn from(a: u64) -> Self {
        Z_N_CRT {
            a1: a.into(),
            a2: a.into(),
        }
    }
}

impl<const N1: u64, const N2: u64> From<i64> for Z_N_CRT<N1, N2> {
    /// Converts i64 to Z_N_CRT by modular reductions.
    fn from(a: i64) -> Self {
        if a < 0 {
            -Z_N_CRT::from(-a as u64)
        } else {
            Z_N_CRT::from(a as u64)
        }
    }
}

/// Math operations on owned `Z_N_CRT<N1, N2>`, including [`RingElement`] implementation.

impl<const N1: u64, const N2: u64> RingElement for Z_N_CRT<N1, N2> {
    fn zero() -> Self {
        0_u64.into()
    }
    fn one() -> Self {
        1_u64.into()
    }
}

impl<const N1: u64, const N2: u64> Add for Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn add(self, rhs: Self) -> Self::Output {
        Z_N_CRT {
            a1: self.a1 + rhs.a1,
            a2: self.a2 + rhs.a2,
        }
    }
}

impl<const N1: u64, const N2: u64> AddAssign for Z_N_CRT<N1, N2> {
    fn add_assign(&mut self, rhs: Self) {
        self.a1 += rhs.a1;
        self.a2 += rhs.a2;
    }
}

impl<const N1: u64, const N2: u64> Mul for Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn mul(self, rhs: Self) -> Self::Output {
        Z_N_CRT {
            a1: self.a1 * rhs.a1,
            a2: self.a2 * rhs.a2,
        }
    }
}

impl<const N1: u64, const N2: u64> MulAssign for Z_N_CRT<N1, N2> {
    fn mul_assign(&mut self, rhs: Self) {
        self.a1 *= rhs.a1;
        self.a2 *= rhs.a2;
    }
}

impl<const N1: u64, const N2: u64> Sub for Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<const N1: u64, const N2: u64> SubAssign for Z_N_CRT<N1, N2> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a1 -= rhs.a1;
        self.a2 -= rhs.a2;
    }
}

impl<const N1: u64, const N2: u64> Neg for Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn neg(mut self) -> Self::Output {
        self.a1 = -self.a1;
        self.a2 = -self.a2;
        self
    }
}

impl<const N1: u64, const N2: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for Z_N_CRT<N1, N2>
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

/// Formatting

impl<const N1: u64, const N2: u64> fmt::Debug for Z_N_CRT<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?})", self.a1, self.a2)
    }
}

/// Math operations on borrows `&Z_N_CRT<N1, N2>`, including [`RingElementRef`] implementation.

impl<const N1: u64, const N2: u64> RingElementRef<Z_N_CRT<N1, N2>> for &Z_N_CRT<N1, N2> {}

impl<const N1: u64, const N2: u64> Neg for &Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<const N1: u64, const N2: u64> Add for &Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<const N1: u64, const N2: u64> AddAssign<&Z_N_CRT<N1, N2>> for Z_N_CRT<N1, N2> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a1 += rhs.a1;
        self.a2 += rhs.a2;
    }
}

impl<const N1: u64, const N2: u64> Sub for &Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<const N1: u64, const N2: u64> SubAssign<&Z_N_CRT<N1, N2>> for Z_N_CRT<N1, N2> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.a1 -= rhs.a1;
        self.a2 -= rhs.a2;
    }
}

impl<const N1: u64, const N2: u64> Mul for &Z_N_CRT<N1, N2> {
    type Output = Z_N_CRT<N1, N2>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<const N1: u64, const N2: u64> MulAssign<&Z_N_CRT<N1, N2>> for Z_N_CRT<N1, N2> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.a1 *= rhs.a1;
        self.a2 *= rhs.a2;
    }
}

/// Random sampling

impl<const N1: u64, const N2: u64> RandUniformSampled for Z_N_CRT<N1, N2> {
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        Z_N_CRT {
            a1: rng.gen_range(0..N1).into(),
            a2: rng.gen_range(0..N2).into(),
        }
    }
}

impl<const N1: u64, const N2: u64> RandZeroOneSampled for Z_N_CRT<N1, N2> {
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        rng.gen_range(0..2_u64).into()
    }
}

impl<const N1: u64, const N2: u64> RandDiscreteGaussianSampled for Z_N_CRT<N1, N2> {
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        DiscreteGaussian::sample::<_, NOISE_WIDTH_MILLIONTHS>(rng).into()
    }
}

/// Other methods
impl<const N1: u64, const N2: u64> Z_N_CRT<N1, N2> {
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
        type Z_55 = Z_N_CRT<5, 11>;

        let a: Z_55 = 0_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z_55 = 1_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z_55 = 54_u64.into();
        assert_eq!(54_u64, a.into());

        let a: Z_55 = 55_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z_55 = 56_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z_55 = ((55 * 439885 + 16) as u64).into();
        assert_eq!(16_u64, a.into());
    }

    #[test]
    fn test_ops() {
        type Z_55 = Z_N_CRT<5, 11>;

        let a: Z_55 = 21_u64.into();
        let b: Z_55 = -a;
        assert_eq!(34_u64, b.into());

        let a: Z_55 = 0_u64.into();
        let b: Z_55 = -a;
        assert_eq!(a, b);

        let mut a: Z_55 = 23_u64.into();
        let b: Z_55 = 45_u64.into();
        assert_eq!(13_u64, (a + b).into());
        a += Z_55::from(45_u64);
        assert_eq!(13_u64, a.into());

        let mut a: Z_55 = 23_u64.into();
        let b: Z_55 = 45_u64.into();
        assert_eq!(33_u64, (a - b).into());
        a -= Z_55::from(45_u64);
        assert_eq!(33_u64, a.into());

        let mut a: Z_55 = 16_u64.into();
        let b: Z_55 = 4_u64.into();
        assert_eq!(9_u64, (a * b).into());
        a *= Z_55::from(4_u64);
        assert_eq!(9_u64, a.into());
    }

    #[test]
    fn test_norm() {
        type Z_55 = Z_N_CRT<5, 11>;

        let zero: Z_55 = 0_u64.into();
        let one_pos: Z_55 = 1_u64.into();
        let one_neg: Z_55 = 54_u64.into();
        let two_pos: Z_55 = 2_u64.into();
        let two_neg: Z_55 = 53_u64.into();
        let twentyseven_pos: Z_55 = 27_u64.into();
        let twentyseven_neg: Z_55 = 28_u64.into();
        assert_eq!(zero.norm(), 0);
        assert_eq!(one_pos.norm(), 1);
        assert_eq!(one_neg.norm(), 1);
        assert_eq!(two_pos.norm(), 2);
        assert_eq!(two_neg.norm(), 2);
        assert_eq!(twentyseven_pos.norm(), 27);
        assert_eq!(twentyseven_neg.norm(), 27);
    }
}
