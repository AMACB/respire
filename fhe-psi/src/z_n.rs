use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::distributions::Standard;
use rand::Rng;

use crate::ring_elem::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Z_N<const N: u64> {
    a: u64,
}

/*
 * Conversions
 */

impl<const N: u64> Into<u64> for Z_N<N> {
    fn into(self) -> u64 {
        self.a
    }
}

impl<const N: u64> From<u64> for Z_N<N> {
    fn from(a: u64) -> Self {
        Z_N {
            a: a % N
        }
    }
}

impl<const N: u64> From<i64> for Z_N<N> {
    fn from(a: i64) -> Self {
        Z_N {
            a: (a % (N as i64) + (N as i64)) as u64 % N
        }
    }
}

/*
 * Math Operations (owned)
 */

impl<const N: u64> Add for Z_N<N> {
    type Output = Z_N<N>;
    fn add(self, rhs: Self) -> Self::Output {
        (((self.a as u128 + rhs.a as u128) % (N as u128)) as u64).into()
    }
}

impl<const N: u64> AddAssign for Z_N<N> {
    fn add_assign(&mut self, rhs: Self) {
        self.a = (self.clone() + rhs).a;
    }
}

impl<const N: u64> Mul for Z_N<N> {
    type Output = Z_N<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        (((self.a as u128 * rhs.a as u128) % (N as u128)) as u64).into()
    }
}

impl<const N: u64> MulAssign for Z_N<N> {
    fn mul_assign(&mut self, rhs: Self) {
        self.a = (self.clone() * rhs).a;
    }
}

impl<const N: u64> Sub for Z_N<N> {
    type Output = Z_N<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<const N: u64> SubAssign for Z_N<N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a = (self.clone() - rhs).a
    }
}

impl<const N: u64> Neg for Z_N<N> {
    type Output = Z_N<N>;
    fn neg(self) -> Self::Output {
        ((N - self.a) % N).into()
    }
}

impl<const N: u64> RingElement for Z_N<N> {
    fn zero() -> Self {
        0_u64.into()
    }
    fn one() -> Self {
        1_u64.into()
    }
    fn random<T: Rng>(rng: &mut T) -> Self {
        // TODO: not actually uniform :clown:
        rng.sample_iter::<u64, _>(&Standard).next().unwrap().into()
    }
}

/*
 * Formatting
 */

impl<const N: u64> fmt::Debug for Z_N<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a <= 3 * N / 4 {
            write!(f, "{}", self.a)
        } else {
            write!(f, "-{}", N - self.a)
        }
    }
}

/*
 * Math Operations (borrows)
 */

impl<const N: u64> RingElementRef<Z_N<N>> for &Z_N<N> {}

impl<const N: u64> Neg for &Z_N<N> {
    type Output = Z_N<N>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<const N: u64> Add for &Z_N<N> {
    type Output = Z_N<N>;
    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs.clone()
    }
}

impl<const N: u64> AddAssign<&Z_N<N>> for Z_N<N> {
    fn add_assign(&mut self, rhs: &Self) {
        self.a = (self.clone() + rhs.clone()).a
    }
}

impl<const N: u64> Sub for &Z_N<N> {
    type Output = Z_N<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.clone() - rhs.clone()
    }
}

impl<const N: u64> SubAssign<&Z_N<N>> for Z_N<N> {
    fn sub_assign(&mut self, rhs: &Self) {
        self.a = (self.clone() - rhs.clone()).a
    }
}

impl<const N: u64> Mul for &Z_N<N> {
    type Output = Z_N<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

impl<const N: u64> MulAssign<&Z_N<N>> for Z_N<N> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.a = (self.clone() * rhs.clone()).a
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from_into() {
        type Z_31 = Z_N<31>;
        type Z_BIG = Z_N<{ u64::MAX - 1 }>;

        let a: Z_31 = 0_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z_31 = 1_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z_31 = 30_u64.into();
        assert_eq!(30_u64, a.into());

        let a: Z_31 = 31_u64.into();
        assert_eq!(0_u64, a.into());

        let a: Z_31 = 32_u64.into();
        assert_eq!(1_u64, a.into());

        let a: Z_31 = ((31 * 439885 + 4) as u64).into();
        assert_eq!(4_u64, a.into());

        let a: Z_BIG = (u64::MAX - 1).into();
        assert_eq!(0_u64, a.into());

        let a: Z_BIG = u64::MAX.into();
        assert_eq!(1_u64, a.into());
    }

    #[test]
    fn test_ops() {
        type Z_31 = Z_N<31>;
        type Z_BIG = Z_N<{ u64::MAX - 1 }>;

        let a: Z_31 = 10_u64.into();
        let b: Z_31 = -a;
        assert_eq!(21_u64, b.into());

        let mut a: Z_31 = 23_u64.into();
        let b: Z_31 = 24_u64.into();
        assert_eq!(16_u64, (a + b).into());
        a += Z_31::from(24_u64);
        assert_eq!(16_u64, a.into());

        let mut a: Z_31 = 23_u64.into();
        let b: Z_31 = 24_u64.into();
        assert_eq!(30_u64, (a - b).into());
        a -= Z_31::from(24_u64);
        assert_eq!(30_u64, a.into());

        let mut a: Z_31 = 16_u64.into();
        let b: Z_31 = 3_u64.into();
        assert_eq!(17_u64, (a * b).into());
        a *= Z_31::from(3_u64);
        assert_eq!(17_u64, a.into());


        let a: Z_BIG = 10_u64.into();
        let b: Z_BIG = -a;
        assert_eq!(u64::MAX - 10 - 1, b.into());

        let mut a: Z_BIG = (u64::MAX - 50005).into();
        let b: Z_BIG = 60006_u64.into();
        assert_eq!(10002_u64, (a + b).into());
        a += Z_BIG::from(60006_u64);
        assert_eq!(10002_u64, a.into());

        let mut a: Z_BIG = 50005_u64.into();
        let b: Z_BIG = 70007_u64.into();
        assert_eq!(u64::MAX - 20003, (a - b).into());
        a -= Z_BIG::from(70007_u64);
        assert_eq!(u64::MAX - 20003, a.into());

        let mut a: Z_BIG = (u64::MAX - 1 - 1984).into();
        let b: Z_BIG = (u64::MAX - 1 - 3968).into();
        assert_eq!(7872512_u64, (a * b).into());
        a *= Z_BIG::from(u64::MAX - 1 - 3968);
        assert_eq!(7872512_u64, a.into());

        let mut a: Z_BIG = (u64::MAX - 1 - 1984).into();
        let b: Z_BIG = 3968_u64.into();
        assert_eq!(u64::MAX - 1 - 7872512, (a * b).into());
        a *= Z_BIG::from(3968_u64);
        assert_eq!(u64::MAX - 1 - 7872512, a.into());
    }
}