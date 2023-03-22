use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use rand::Rng;

use crate::z_n::Z_N;
use crate::polynomial::PolynomialZ_N;
use crate::ring_elem::{RingElement, RingElementRef};

/*
 * Z_q[X] / (X^D + 1)
 * D must be a power of 2 so X^D+1 is cyclotomic.
 */

/*
 * TODO: This is the stupid implementation. We will need:
 *  * something to account for roots of unity (type parameter probably)
 *  * something to bind these roots of unity to modulus (probably similar approach to FHE / GSW)
 */
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CyclotomicPolynomial<const D: usize, const N: u64> {
    polynomial: PolynomialZ_N<N>
}

/*
 * Conversions
 */

impl<const D: usize, const N: u64> From<PolynomialZ_N<N>> for CyclotomicPolynomial<D, N> {
    fn from(polynomial: PolynomialZ_N<N>) -> Self {
        let mut coeffs: Vec<Z_N<N>> = vec![];
        let zero: Z_N<N> = 0_u64.into();
        coeffs.resize(D, zero);
        for (i, x) in polynomial.coeff_iter().enumerate() {
            if i / D % 2 == 0 {
                coeffs[i % D] += x;
            }
            else {
                coeffs[i % D] -= x;
            }
        }
        CyclotomicPolynomial { polynomial: coeffs.into() }
    }
}

impl<const D: usize, const N: u64> From<Vec<u64>> for CyclotomicPolynomial<D, N> {
    fn from(coeff: Vec<u64>) -> Self {
        CyclotomicPolynomial::from(PolynomialZ_N::from(coeff))
    }
}

impl<const D: usize, const N: u64> From<Vec<Z_N<N>>> for CyclotomicPolynomial<D, N> {
    fn from(coeff: Vec<Z_N<N>>) -> Self {
        CyclotomicPolynomial::from(PolynomialZ_N::from(coeff))
    }
}

/*
 * RingElementRef implementation
 */

impl<const D: usize, const N: u64> RingElementRef<CyclotomicPolynomial<D,N>> for &CyclotomicPolynomial<D,N> {}

impl<const D: usize, const N: u64> Add for &CyclotomicPolynomial<D,N> {
    type Output = CyclotomicPolynomial<D,N>;
    fn add(self, rhs: Self) -> Self::Output {
        (&self.polynomial + &rhs.polynomial).into()
    }
}

impl<const D: usize, const N: u64> Sub for &CyclotomicPolynomial<D,N> {
    type Output = CyclotomicPolynomial<D,N>;
    fn sub(self, rhs: Self) -> Self::Output {
        (&self.polynomial - &rhs.polynomial).into()
    }
}

impl<const D: usize, const N: u64> Mul for &CyclotomicPolynomial<D,N> {
    type Output = CyclotomicPolynomial<D,N>;
    fn mul(self, rhs: Self) -> Self::Output {
        (&self.polynomial * &rhs.polynomial).into()
    }
}

impl<const D: usize, const N: u64> Neg for &CyclotomicPolynomial<D,N> {
    type Output = CyclotomicPolynomial<D,N>;
    fn neg(self) -> Self::Output {
        (-&self.polynomial).into()
    }
}

/*
 * RingElement implementation
 */

impl<const D: usize, const N: u64> RingElement for CyclotomicPolynomial<D,N> {
    fn zero() -> CyclotomicPolynomial<D,N> {
        vec![0_u64].into()
    }

    fn one() -> CyclotomicPolynomial<D,N> {
        vec![1_u64].into()
    }

    fn random<T: Rng>(_: &mut T) -> Self {
        unimplemented!()
    }
}

// TODO: make these not stupid

impl<'a, const D: usize, const N: u64> AddAssign<&'a Self> for CyclotomicPolynomial<D,N> {
    fn add_assign(&mut self, rhs: &'a Self) {
        let result = &(*self) + rhs;
        self.polynomial = result.polynomial
    }
}

impl<'a, const D: usize, const N: u64> SubAssign<&'a Self> for CyclotomicPolynomial<D,N> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        let result = &(*self) - rhs;
        self.polynomial = result.polynomial
    }
}

impl<'a, const D: usize, const N: u64> MulAssign<&'a Self> for CyclotomicPolynomial<D,N> {
    fn mul_assign(&mut self, rhs: &'a Self) {
        let result = &(*self) * rhs;
        self.polynomial = result.polynomial;
    }
}

/*
 * Other Polynomial things
 */

impl<const D: usize, const N: u64> CyclotomicPolynomial<D,N> {
}

#[cfg(test)]
mod test {
    use super::*;

    const D: usize = 4; // Z_q[X] / (X^4 + 1)
    const P: u64 = (1_u64 << 32) - 5;

    // TODO: add more tests.
    #[test]
    fn test_from_deg() {
        let p = CyclotomicPolynomial::<D,P>::from(vec![42, 6, 1, 0, 5]);
        let q = CyclotomicPolynomial::<D,P>::from(vec![37, 6, 1, 0]);
        let r = CyclotomicPolynomial::<D,P>::from(vec![41, 6, 1, 0, 5, 0, 0, 0, 1]);
        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);
    }
}
