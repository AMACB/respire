use std::iter;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::Rng;

use crate::ring_elem::{RingElement, RingElementRef};
use crate::z_n::Z_N;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolynomialZ_N<const N: u64> {
    coeff: Vec<Z_N<N>>,
}

/*
 * Conversions
 */

impl<const N: u64> From<Vec<u64>> for PolynomialZ_N<N> {
    fn from(coeff: Vec<u64>) -> Self {
        let v: Vec<Z_N<N>> = coeff.iter().map(|x| (*x).into()).collect();
        PolynomialZ_N::from(v)
    }
}

impl<const N: u64> From<Vec<Z_N<N>>> for PolynomialZ_N<N> {
    fn from(mut coeff: Vec<Z_N<N>>) -> Self {
        let mut idx = coeff.len();
        loop {
            if idx == 0 || coeff[idx - 1] != 0_u64.into() {
                break;
            }
            idx -= 1;
        }
        coeff.resize(idx, 0_u64.into());
        PolynomialZ_N { coeff }
    }
}

/*
 * RingElementRef implementation
 */

impl<const N: u64> RingElementRef<PolynomialZ_N<N>> for &PolynomialZ_N<N> {}

impl<const N: u64> Add for &PolynomialZ_N<N> {
    type Output = PolynomialZ_N<N>;
    fn add(self, rhs: Self) -> Self::Output {
        if self.coeff.len() < rhs.coeff.len() {
            return rhs + self;
        }

        let mut result_coeff: Vec<Z_N<N>> = vec![];
        let zero: Z_N<N> = 0_u64.into();
        result_coeff.resize(self.coeff.len(), zero);
        let self_iter = self.coeff.iter();
        let rhs_iter = rhs.coeff.iter().chain(iter::repeat(&zero));
        for (i, (a, b)) in self_iter.zip(rhs_iter).enumerate() {
            result_coeff[i] = a + b;
        }
        result_coeff.into()
    }
}

impl<const N: u64> Sub for &PolynomialZ_N<N> {
    type Output = PolynomialZ_N<N>;
    fn sub(self, _: Self) -> Self::Output {
        todo!()
    }
}

impl<const N: u64> Mul for &PolynomialZ_N<N> {
    type Output = PolynomialZ_N<N>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result_coeff: Vec<Z_N<N>> = vec![];
        result_coeff.resize(self.coeff.len() + rhs.coeff.len(), 0_u64.into());
        for (i, a) in self.coeff.iter().enumerate() {
            for (j, b) in rhs.coeff.iter().enumerate() {
                result_coeff[i + j] += a * b;
            }
        }
        result_coeff.into()
    }
}

impl<const N: u64> Neg for &PolynomialZ_N<N> {
    type Output = PolynomialZ_N<N>;
    fn neg(self) -> Self::Output {
        let result_coeff: Vec<Z_N<N>> = self.coeff.iter().map(|x| -x).collect();
        result_coeff.into()
    }
}

/*
 * RingElement implementation
 */

impl<const N: u64> RingElement for PolynomialZ_N<N> {
    fn zero() -> PolynomialZ_N<N> {
        vec![0_u64].into()
    }

    fn one() -> PolynomialZ_N<N> {
        vec![1_u64].into()
    }

    fn random<T: Rng>(_: &mut T) -> Self {
        unimplemented!()
    }
}

impl<'a, const N: u64> AddAssign<&'a Self> for PolynomialZ_N<N> {
    fn add_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

impl<'a, const N: u64> SubAssign<&'a Self> for PolynomialZ_N<N> {
    fn sub_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

impl<'a, const N: u64> MulAssign<&'a Self> for PolynomialZ_N<N> {
    fn mul_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

/*
 * Other Polynomial things
 */

impl<const N: u64> PolynomialZ_N<N> {
    pub fn x() -> PolynomialZ_N<N> {
        vec![0_u64, 1_u64].into()
    }

    pub fn eval(&self, x: Z_N<N>) -> Z_N<N> {
        let mut result: Z_N<N> = 0_u64.into();
        let mut current_pow: Z_N<N> = 1_u64.into();
        for a in &self.coeff {
            result += current_pow * *a;
            current_pow *= x;
        }
        result
    }

    // pub fn eval_gsw<const NN: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(&self, pk: &PublicKey<NN, M, P, Q, G_BASE, G_LEN>, ct: &Ciphertext<NN, M, P, Q, G_BASE, G_LEN>) -> Ciphertext<NN, M, P, Q, G_BASE, G_LEN> {
    //     let mut result = gsw::gsw::encrypt(&pk, &Z_N::new_u(0));
    //     let mut current_pow = gsw::gsw::encrypt(&pk, &Z_N::new_u(1));
    //     for a in &self.coeff {
    //         let term = &current_pow * &gsw::gsw::encrypt(&pk, &Z_N::new_u(*a as u64));
    //         result = &result + &term;
    //         current_pow = &current_pow * ct;
    //     }
    //     result
    // }
}

#[cfg(test)]
mod test {
    use super::*;

    const P: u64 = (1_u64 << 32) - 5;

    #[test]
    fn test_from() {
        let p = PolynomialZ_N::<P>::from(vec![42, 6, 1, 0, 0, 0]);
        let q = PolynomialZ_N::<P>::from(vec![42, 6, 1, 0]);
        let r = PolynomialZ_N::<P>::from(vec![42, 6, 1]);
        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);

        let t = PolynomialZ_N::<P>::from(vec![0, 0]);
        let u = PolynomialZ_N::<P>::from(vec![0]);
        let v = PolynomialZ_N::<P>::from(vec![0_u64; 0]);
        assert_eq!(t, u);
        assert_eq!(t, v);
        assert_eq!(u, v);
    }

    #[test]
    fn test_eval() {
        let zero = PolynomialZ_N::<P>::zero();
        let one = PolynomialZ_N::<P>::one();
        let x = PolynomialZ_N::<P>::x();
        assert_eq!(zero.eval(0_u64.into()), 0_u64.into());
        assert_eq!(zero.eval(10_u64.into()), 0_u64.into());
        assert_eq!(zero.eval(31_u64.into()), 0_u64.into());
        assert_eq!(one.eval(0_u64.into()), 1_u64.into());
        assert_eq!(one.eval(10_u64.into()), 1_u64.into());
        assert_eq!(one.eval(31_u64.into()), 1_u64.into());
        assert_eq!(x.eval(0_u64.into()), 0_u64.into());
        assert_eq!(x.eval(10_u64.into()), 10_u64.into());
        assert_eq!(x.eval(31_u64.into()), 31_u64.into());

        let p = PolynomialZ_N::<P>::from(vec![5, 3, 1]);

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
        let p1 = PolynomialZ_N::<P>::from(vec![5, 3, 1]);
        let q1 = PolynomialZ_N::<P>::from(vec![5, 3]);
        assert_eq!(&p1 + &q1, vec![10, 6, 1].into());
        assert_eq!(&q1 + &p1, vec![10, 6, 1].into());

        let p2 = PolynomialZ_N::<P>::from(vec![5, 3, 1]);
        let q2 = PolynomialZ_N::<P>::from(vec![5, 3, P - 1]);
        assert_eq!(&p2 + &q2, vec![10, 6].into());
        assert_eq!(&q2 + &p2, vec![10, 6].into());
    }

    #[test]
    fn test_mul() {
        let p = PolynomialZ_N::<P>::from(vec![5, 3, 1]);
        let q = PolynomialZ_N::<P>::from(vec![P - 4, 2, 1]);
        let r = PolynomialZ_N::<P>::from(vec![P - 20, P - 2, 7, 5, 1]);
        assert_eq!(&p * &q, r.clone());

        let zero = PolynomialZ_N::<P>::zero();
        let one = PolynomialZ_N::<P>::one();
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
