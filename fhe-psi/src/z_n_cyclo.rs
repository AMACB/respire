use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::polynomial::PolynomialZ_N;
use crate::ring_elem::{RingElement, RingElementRef};
use crate::z_n::Z_N;

/*
 * Z_N[X] / (X^D + 1)
 * D must be a power of 2 so X^D+1 is cyclotomic.
 *
 * Z_N_CycloRaw is the coefficient representation
 * Z_N_CycloNTT is the evaluation representation
 */

/*
 * TODO: This is the stupid implementation. We will need:
 *  * something to account for roots of unity (type parameter probably)
 *  * something to bind these roots of unity to modulus (probably similar approach to FHE / GSW)
 */
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Z_N_CycloRaw<const D: usize, const N: u64> {
    coeff: [Z_N<N>; D],
}

/*
 * Conversions
 */

impl<const D: usize, const N: u64> From<[Z_N<N>; D]> for Z_N_CycloRaw<D, N> {
    fn from(coeff: [Z_N<N>; D]) -> Self {
        Self { coeff }
    }
}

impl<const D: usize, const N: u64> From<PolynomialZ_N<N>> for Z_N_CycloRaw<D, N> {
    fn from(polynomial: PolynomialZ_N<N>) -> Self {
        let mut coeff: [Z_N<N>; D] = [0_u64.into(); D];
        for (i, x) in polynomial.coeff_iter().enumerate() {
            if i / D % 2 == 0 {
                coeff[i % D] += x;
            } else {
                coeff[i % D] -= x;
            }
        }
        coeff.into()
    }
}

impl<const D: usize, const N: u64> From<Vec<u64>> for Z_N_CycloRaw<D, N> {
    fn from(coeff: Vec<u64>) -> Self {
        Z_N_CycloRaw::from(PolynomialZ_N::from(coeff))
    }
}

impl<const D: usize, const N: u64> From<Vec<Z_N<N>>> for Z_N_CycloRaw<D, N> {
    fn from(coeff: Vec<Z_N<N>>) -> Self {
        Z_N_CycloRaw::from(PolynomialZ_N::from(coeff))
    }
}

/*
 * RingElementRef implementation
 */

impl<const D: usize, const N: u64> RingElementRef<Z_N_CycloRaw<D, N>> for &Z_N_CycloRaw<D, N> {}

impl<const D: usize, const N: u64> Add for &Z_N_CycloRaw<D, N> {
    type Output = Z_N_CycloRaw<D, N>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result_coeff: [Z_N<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_coeff[i] = self.coeff[i] + rhs.coeff[i];
        }
        result_coeff.into()
    }
}

impl<const D: usize, const N: u64> Sub for &Z_N_CycloRaw<D, N> {
    type Output = Z_N_CycloRaw<D, N>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result_coeff: [Z_N<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_coeff[i] = self.coeff[i] - rhs.coeff[i];
        }
        result_coeff.into()
    }
}

impl<const D: usize, const N: u64> Mul for &Z_N_CycloRaw<D, N> {
    type Output = Z_N_CycloRaw<D, N>;
    fn mul(self, _: Self) -> Self::Output {
        todo!()
    }
}

impl<const D: usize, const N: u64> Neg for &Z_N_CycloRaw<D, N> {
    type Output = Z_N_CycloRaw<D, N>;
    fn neg(self) -> Self::Output {
        let mut result_coeff: [Z_N<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_coeff[i] = -self.coeff[i];
        }
        result_coeff.into()
    }
}

/*
 * RingElement implementation
 */

impl<const D: usize, const N: u64> RingElement for Z_N_CycloRaw<D, N> {
    fn zero() -> Z_N_CycloRaw<D, N> {
        [0_u64.into(); D].into()
    }

    fn one() -> Z_N_CycloRaw<D, N> {
        [1_u64.into(); D].into()
    }

    fn random<T: Rng>(_: &mut T) -> Self {
        unimplemented!()
    }
}

// TODO: make these not stupid

impl<'a, const D: usize, const N: u64> AddAssign<&'a Self> for Z_N_CycloRaw<D, N> {
    fn add_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.coeff[i] += rhs.coeff[i];
        }
    }
}

impl<'a, const D: usize, const N: u64> SubAssign<&'a Self> for Z_N_CycloRaw<D, N> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.coeff[i] -= rhs.coeff[i];
        }
    }
}

impl<'a, const D: usize, const N: u64> MulAssign<&'a Self> for Z_N_CycloRaw<D, N> {
    fn mul_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const D: usize = 4; // Z_q[X] / (X^4 + 1)
    const P: u64 = (1_u64 << 32) - 5;

    // TODO: add more tests.
    #[test]
    fn test_from_deg() {
        let p = Z_N_CycloRaw::<D, P>::from(vec![42, 6, 1, 0, 5]);
        let q = Z_N_CycloRaw::<D, P>::from(vec![37, 6, 1, 0]);
        let r = Z_N_CycloRaw::<D, P>::from(vec![41, 6, 1, 0, 5, 0, 0, 0, 1]);
        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);
    }
}
