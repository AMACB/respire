//! Polynomials over `Z_n`.

use crate::math::int_mod::IntMod;
use crate::math::ring_elem::*;
use std::iter;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::slice::Iter;

// TODO: Improve the in-place `RingElement` operations

/// Coefficient representation of a polynomial with coefficients mod `n`. While this type does
/// implement [`RingElement`], [`RingElementRef`], it is not intended for use if high efficiency is
/// required.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct IntModPoly<const N: u64> {
    pub(in crate::math) coeff: Vec<IntMod<N>>,
}

/// Conversions

impl<const N: u64> From<u64> for IntModPoly<N> {
    fn from(a: u64) -> Self {
        vec![a].into()
    }
}

impl<const N: u64> From<Vec<i64>> for IntModPoly<N> {
    fn from(coeff: Vec<i64>) -> Self {
        let v: Vec<IntMod<N>> = coeff.iter().map(|x| (*x).into()).collect();
        IntModPoly::from(v)
    }
}

impl<const N: u64> From<Vec<u64>> for IntModPoly<N> {
    fn from(coeff: Vec<u64>) -> Self {
        let v: Vec<IntMod<N>> = coeff.iter().map(|x| (*x).into()).collect();
        IntModPoly::from(v)
    }
}

impl<const N: u64> From<Vec<IntMod<N>>> for IntModPoly<N> {
    fn from(mut coeff: Vec<IntMod<N>>) -> Self {
        let mut idx = coeff.len();
        loop {
            if idx == 0 || coeff[idx - 1] != 0_u64.into() {
                break;
            }
            idx -= 1;
        }
        coeff.resize(idx, 0_u64.into());
        IntModPoly { coeff }
    }
}

/// RingElementRef implementation

impl<const N: u64> RingElementRef<IntModPoly<N>> for &IntModPoly<N> {}

impl<const N: u64> Add for &IntModPoly<N> {
    type Output = IntModPoly<N>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.coeff.len() < rhs.coeff.len() {
            return rhs + self;
        }

        let mut result_coeff: Vec<IntMod<N>> = vec![];
        let zero: IntMod<N> = 0_u64.into();
        result_coeff.resize(self.coeff.len(), zero);
        let self_iter = self.coeff.iter();
        let rhs_iter = rhs.coeff.iter().chain(iter::repeat(&zero));
        for (i, (a, b)) in self_iter.zip(rhs_iter).enumerate() {
            result_coeff[i] = a + b;
        }
        result_coeff.into()
    }
}

impl<const N: u64> Sub for &IntModPoly<N> {
    type Output = IntModPoly<N>;
    fn sub(self, _: Self) -> Self::Output {
        todo!()
    }
}

impl<const N: u64> Mul for &IntModPoly<N> {
    type Output = IntModPoly<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result_coeff: Vec<IntMod<N>> = vec![];
        result_coeff.resize(self.coeff.len() + rhs.coeff.len(), 0_u64.into());
        for (i, a) in self.coeff.iter().enumerate() {
            for (j, b) in rhs.coeff.iter().enumerate() {
                result_coeff[i + j] += a * b;
            }
        }
        result_coeff.into()
    }
}

impl<const N: u64> Neg for &IntModPoly<N> {
    type Output = IntModPoly<N>;
    fn neg(self) -> Self::Output {
        let result_coeff: Vec<IntMod<N>> = self.coeff.iter().map(|x| -x).collect();
        result_coeff.into()
    }
}

/// RingElement implementation

impl<const N: u64> RingElement for IntModPoly<N> {
    fn zero() -> IntModPoly<N> {
        vec![0_u64].into()
    }
    fn one() -> IntModPoly<N> {
        vec![1_u64].into()
    }
}

impl<'a, const N: u64> AddAssign<&'a Self> for IntModPoly<N> {
    fn add_assign(&mut self, rhs: &'a Self) {
        let result = &(*self) + rhs;
        self.coeff = result.coeff
    }
}

impl<'a, const N: u64> SubAssign<&'a Self> for IntModPoly<N> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        let result = &(*self) - rhs;
        self.coeff = result.coeff
    }
}

impl<'a, const N: u64> MulAssign<&'a Self> for IntModPoly<N> {
    fn mul_assign(&mut self, rhs: &'a Self) {
        let result = &(*self) * rhs;
        self.coeff = result.coeff;
    }
}

/// Other polynomial-specific operations.

impl<const N: u64> IntModPoly<N> {
    /// Constructs the polynomial `x`.
    pub fn x() -> IntModPoly<N> {
        vec![0_u64, 1_u64].into()
    }

    /// Evaluates the polynomial at the given point.
    pub fn eval(&self, x: IntMod<N>) -> IntMod<N> {
        let mut result: IntMod<N> = 0_u64.into();
        let mut current_pow: IntMod<N> = 1_u64.into();
        for a in &self.coeff {
            result += current_pow * *a;
            current_pow *= x;
        }
        result
    }

    /// Degree of polynomial. By convention 0 has degree -1.
    pub fn deg(&self) -> isize {
        (self.coeff.len() as isize) - 1
    }

    /// Iterator over the coefficients in order of increasing power (`x^0`, `x^1`, ...)
    pub fn coeff_iter(&self) -> Iter<'_, IntMod<{ N }>> {
        self.coeff.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const P: u64 = (1_u64 << 32) - 5;

    #[test]
    fn test_from_deg() {
        let p = IntModPoly::<P>::from(vec![42_u64, 6, 1, 0, 0, 0]);
        let q = IntModPoly::<P>::from(vec![42_u64, 6, 1, 0]);
        let r = IntModPoly::<P>::from(vec![42_u64, 6, 1]);
        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);
        assert_eq!(p.deg(), 2);
        assert_eq!(q.deg(), 2);
        assert_eq!(r.deg(), 2);

        let t = IntModPoly::<P>::from(vec![0_u64, 0]);
        let u = IntModPoly::<P>::from(vec![0_u64]);
        let v = IntModPoly::<P>::from(vec![0_u64; 0]);
        assert_eq!(t, u);
        assert_eq!(t, v);
        assert_eq!(u, v);
        assert_eq!(t.deg(), -1);
        assert_eq!(u.deg(), -1);
        assert_eq!(v.deg(), -1);
    }

    #[test]
    fn test_eval() {
        let zero = IntModPoly::<P>::zero();
        let one = IntModPoly::<P>::one();
        let x = IntModPoly::<P>::x();
        assert_eq!(zero.eval(0_u64.into()), 0_u64.into());
        assert_eq!(zero.eval(10_u64.into()), 0_u64.into());
        assert_eq!(zero.eval(31_u64.into()), 0_u64.into());
        assert_eq!(one.eval(0_u64.into()), 1_u64.into());
        assert_eq!(one.eval(10_u64.into()), 1_u64.into());
        assert_eq!(one.eval(31_u64.into()), 1_u64.into());
        assert_eq!(x.eval(0_u64.into()), 0_u64.into());
        assert_eq!(x.eval(10_u64.into()), 10_u64.into());
        assert_eq!(x.eval(31_u64.into()), 31_u64.into());

        let p = IntModPoly::<P>::from(vec![5_u64, 3, 1]);

        assert_eq!(p.eval((P - 3).into()), 5_u64.into());
        assert_eq!(p.eval((P - 2).into()), 3_u64.into());
        assert_eq!(p.eval((P - 1).into()), 3_u64.into());
        assert_eq!(p.eval(0_u64.into()), 5_u64.into());
        assert_eq!(p.eval(1_u64.into()), 9_u64.into());
        assert_eq!(p.eval(2_u64.into()), 15_u64.into());
        assert_eq!(p.eval(3_u64.into()), 23_u64.into());
    }

    #[test]
    fn test_add() {
        let p1 = IntModPoly::<P>::from(vec![5_u64, 3, 1]);
        let q1 = IntModPoly::<P>::from(vec![5_u64, 3]);
        assert_eq!(&p1 + &q1, vec![10_u64, 6, 1].into());
        assert_eq!(&q1 + &p1, vec![10_u64, 6, 1].into());

        let p2 = IntModPoly::<P>::from(vec![5_u64, 3, 1]);
        let q2 = IntModPoly::<P>::from(vec![5_u64, 3, P - 1]);
        assert_eq!(&p2 + &q2, vec![10_u64, 6].into());
        assert_eq!(&q2 + &p2, vec![10_u64, 6].into());
    }

    #[test]
    fn test_mul() {
        let p = IntModPoly::<P>::from(vec![5_u64, 3, 1]);
        let q = IntModPoly::<P>::from(vec![-4_i64, 2, 1]);
        let r = IntModPoly::<P>::from(vec![-20_i64, -2, 7, 5, 1]);
        assert_eq!(&p * &q, r.clone());

        let zero = IntModPoly::<P>::zero();
        let one = IntModPoly::<P>::one();
        assert_eq!(&p * &zero, zero.clone());
        assert_eq!(&q * &zero, zero.clone());
        assert_eq!(&r * &zero, zero.clone());
        assert_eq!(&zero * &p, zero.clone());
        assert_eq!(&zero * &q, zero.clone());
        assert_eq!(&zero * &r, zero.clone());
        assert_eq!(&p * &one, p.clone());
        assert_eq!(&q * &one, q.clone());
        assert_eq!(&r * &one, r.clone());
        assert_eq!(&one * &p, p.clone());
        assert_eq!(&one * &q, q.clone());
        assert_eq!(&one * &r, r.clone());
        assert_eq!(&zero * &zero, zero.clone());
    }
}
