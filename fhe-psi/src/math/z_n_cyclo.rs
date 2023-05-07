//! The cyclotomic ring `Z_n[x]/x^d + 1)`. `d` is assumed to be a power of `2`.

use crate::fhe::gadget::RingElementDecomposable;
use crate::math::matrix::Matrix;
use crate::math::ntt::*;
use crate::math::polynomial::PolynomialZ_N;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use crate::math::z_n::{NoReduce, Z_N};
use crate::math::z_n_cyclo_ntt::Z_N_CycloNTT;
use rand::Rng;
use std::cmp::{max, min};
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};
use std::slice::Iter;

// TODO
// This is the stupid implementation. We will need:
// * something to account for roots of unity (type parameter probably)
// * something to bind these roots of unity to modulus (probably similar approach to FHE / GSW)

/// The raw (coefficient) representation of an element of a cyclotomic ring.
///
/// Internally, this is an array of coefficients where the `i`th index corresponds to `x^i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Z_N_CycloRaw<const D: usize, const N: u64> {
    coeff: [Z_N<N>; D],
}

/// Conversions

impl<const D: usize, const N: u64> From<u64> for Z_N_CycloRaw<D, N> {
    fn from(a: u64) -> Self {
        let mut result = Self::zero();
        result.coeff[0] = a.into();
        result
    }
}

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

impl<const D: usize, const N: u64> TryFrom<&Z_N_CycloRaw<D, N>> for Z_N<N> {
    type Error = ();

    /// Inverse of `From<u64>`. Errors if element is not a constant.
    fn try_from(a: &Z_N_CycloRaw<D, N>) -> Result<Self, Self::Error> {
        for i in 1..D {
            if a.coeff[i] != Z_N::zero() {
                return Err(());
            }
        }
        Ok(a.coeff[0])
    }
}

impl<const D: usize, const N: u64, const W: u64> From<Z_N_CycloNTT<D, N, W>>
    for Z_N_CycloRaw<D, N>
{
    fn from(z_n_cyclo_ntt: Z_N_CycloNTT<D, N, W>) -> Self {
        (&z_n_cyclo_ntt).into()
    }
}

impl<const D: usize, const N: u64, const W: u64> From<&Z_N_CycloNTT<D, N, W>>
    for Z_N_CycloRaw<D, N>
{
    fn from(z_n_cyclo_ntt: &Z_N_CycloNTT<D, N, W>) -> Self {
        // TODO: this should be in the type, probably
        let mut log_d = 1;
        while (1 << log_d) < D {
            log_d += 1;
        }
        assert_eq!(1 << log_d, D);

        let root = W.into();

        let mut coeff: [Z_N<N>; D] = [0_u64.into(); D];
        for (i, x) in z_n_cyclo_ntt.points_iter().enumerate() {
            coeff[i] = x.clone();
        }

        bit_reverse_order(&mut coeff, log_d);
        ntt(&mut coeff, inverse(root * root), log_d);

        let mut inv_root_pow : Z_N<N> = 1u64.into();
        let inv_root = inverse(root);
        let inv_D = inverse((D as u64).into());
        for i in 0..D {
            // divide by degree
            coeff[i] *= inv_D;
            // negacyclic post-processing
            coeff[i] *= inv_root_pow;
            inv_root_pow *= inv_root;
        }

        return Self { coeff };
    }
}

/// [`RingElementRef`] implementation

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
    fn mul(self, rhs: Self) -> Self::Output {
        let mut self_poly: PolynomialZ_N<N> = self.coeff.to_vec().into();
        let rhs_poly: PolynomialZ_N<N> = rhs.coeff.to_vec().into();
        self_poly *= &rhs_poly;
        self_poly.into()
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

/// [`RingElement`] implementation

impl<const D: usize, const N: u64> RingElement for Z_N_CycloRaw<D, N> {
    fn zero() -> Z_N_CycloRaw<D, N> {
        [0_u64.into(); D].into()
    }
    fn one() -> Z_N_CycloRaw<D, N> {
        let mut result = Self::zero();
        result.coeff[0] = 1_u64.into();
        result
    }
}

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

impl<'a, const D: usize, const N: u64> MulAssign<Z_N<N>> for Z_N_CycloRaw<D, N> {
    fn mul_assign(&mut self, rhs: Z_N<N>) {
        for i in 0..D {
            self.coeff[i] *= rhs;
        }
    }
}

impl<'a, const D: usize, const N: u64> MulAssign<&'a Self> for Z_N_CycloRaw<D, N> {
    fn mul_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

impl<const D: usize, const NN: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for Z_N_CycloRaw<D, NN>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let mut a: [u64; D] = [0; D];
        for l in 0..D {
            a[l] = self.coeff[l].into();
        }
        for k in 0..LEN {
            let mut a_rem = Z_N_CycloRaw::zero();
            for l in 0..D {
                a_rem.coeff[l] = (a[l] % BASE).into();
                a[l] /= BASE;
            }
            mat[(i + k, j)] = a_rem;
        }
    }
}

/// Random sampling

impl<const D: usize, const N: u64> RandUniformSampled for Z_N_CycloRaw<D, N> {
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..D {
            result.coeff[i] = Z_N::<N>::rand_uniform(rng);
        }
        result
    }
}

impl<const D: usize, const N: u64> RandZeroOneSampled for Z_N_CycloRaw<D, N> {
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..(D / 64) {
            let rand = rng.gen::<u64>();
            for bit in 0..64 {
                result.coeff[i * 64 + bit] = NoReduce((rand >> bit) & 1).into();
            }
        }

        let rand = rng.gen::<u64>();
        for bit in 0..(D % 64) {
            result.coeff[(D / 64) * 64 + bit] = NoReduce((rand >> bit) & 1).into();
        }
        result
    }
}

impl<const D: usize, const N: u64> RandDiscreteGaussianSampled for Z_N_CycloRaw<D, N> {
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..D {
            result.coeff[i] = Z_N::<N>::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(rng);
        }
        result
    }
}

/// Other polynomial-specific operations.

// TODO: maybe don't need this bc of index
impl<const D: usize, const N: u64> Z_N_CycloRaw<D, N> {
    pub fn coeff_iter(&self) -> Iter<'_, Z_N<{ N }>> {
        self.coeff.iter()
    }
}

impl<const D: usize, const N: u64> Z_N_CycloRaw<D, N> {
    pub fn norm(&self) -> u64 {
        let mut worst: u64 = 0;
        for i in 0..D {
            let pos: u64 = self.coeff[i].into();
            let neg: u64 = (-self.coeff[i]).into();
            worst = max(worst, min(pos, neg));
        }
        worst
    }
}

impl<const D: usize, const N: u64> Index<usize> for Z_N_CycloRaw<D, N> {
    type Output = Z_N<N>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.coeff[index]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::matrix::Matrix;

    const D: usize = 4; // Z_q[X] / (X^4 + 1)
    const P: u64 = (1_u64 << 32) - 5;

    // TODO: add more tests.
    #[test]
    fn test_from_into() {
        let p = Z_N_CycloRaw::<D, P>::from(vec![42, 6, 1, 0, 5]);
        let q = Z_N_CycloRaw::<D, P>::from(vec![37, 6, 1, 0]);
        let r = Z_N_CycloRaw::<D, P>::from(vec![41, 6, 1, 0, 5, 0, 0, 0, 1]);
        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);

        let s = Z_N_CycloRaw::<D, P>::from(vec![9483, 1, 1, 1, 323, P - 12139, 10491, 1, 1]);
        let t = Z_N_CycloRaw::<D, P>::from(vec![9161, 12140, P - 10490, 0, 0]);
        assert_eq!(s, t);
    }

    #[test]
    fn test_ops() {
        let p = Z_N_CycloRaw::<D, P>::from(vec![0, 0, 0, 1]);
        let q = Z_N_CycloRaw::<D, P>::from(vec![0, 0, 2, 0]);
        let sum = Z_N_CycloRaw::<D, P>::from(vec![0, 0, 2, 1]);
        let diff = Z_N_CycloRaw::<D, P>::from(vec![0, 0, P - 2, 1]);
        let prod = Z_N_CycloRaw::<D, P>::from(vec![0, P - 2, 0, 0]);
        let square = Z_N_CycloRaw::<D, P>::from(vec![0, 0, P - 1, 0]);
        let neg = Z_N_CycloRaw::<D, P>::from(vec![0, 0, 0, P - 1]);
        assert_eq!(&p + &q, sum);
        assert_eq!(&p - &q, diff);
        assert_eq!(&p * &q, prod);
        assert_eq!(&p * &p, square);
        assert_eq!(-&p, neg);
    }

    #[test]
    fn test_matrix() {
        let mut M: Matrix<2, 2, Z_N_CycloRaw<D, P>> = Matrix::zero();
        M[(0, 0)] = Z_N_CycloRaw::<D, P>::from(vec![0, 0, 0, 1]);
        M[(0, 1)] = Z_N_CycloRaw::<D, P>::from(vec![0, 0, 1, 0]);
        M[(1, 0)] = Z_N_CycloRaw::<D, P>::from(vec![0, 1, 0, 0]);
        M[(1, 1)] = Z_N_CycloRaw::<D, P>::from(vec![1, 0, 0, 0]);
        // M =
        // [ x^3 x^2 ]
        // [ x   1   ]
        let M_square = &M * &M;
        assert_eq!(
            M_square[(0, 0)],
            Z_N_CycloRaw::<D, P>::from(vec![0, 0, P - 1, 1])
        ); // x^3 + x^6
        assert_eq!(
            M_square[(0, 1)],
            Z_N_CycloRaw::<D, P>::from(vec![0, P - 1, 1, 0])
        ); // x^2 + x^5
        assert_eq!(
            M_square[(1, 0)],
            Z_N_CycloRaw::<D, P>::from(vec![P - 1, 1, 0, 0])
        ); // x + x^4
        assert_eq!(
            M_square[(1, 1)],
            Z_N_CycloRaw::<D, P>::from(vec![1, 0, 0, 1])
        ); // 1 + x^3

        let M_double = &M + &M;
        assert_eq!(
            M_double[(0, 0)],
            Z_N_CycloRaw::<D, P>::from(vec![0, 0, 0, 2])
        );
        assert_eq!(
            M_double[(0, 1)],
            Z_N_CycloRaw::<D, P>::from(vec![0, 0, 2, 0])
        );
        assert_eq!(
            M_double[(1, 0)],
            Z_N_CycloRaw::<D, P>::from(vec![0, 2, 0, 0])
        );
        assert_eq!(
            M_double[(1, 1)],
            Z_N_CycloRaw::<D, P>::from(vec![2, 0, 0, 0])
        );

        let M_neg = -&M;
        assert_eq!(
            M_neg[(0, 0)],
            Z_N_CycloRaw::<D, P>::from(vec![0, 0, 0, P - 1])
        );
        assert_eq!(
            M_neg[(0, 1)],
            Z_N_CycloRaw::<D, P>::from(vec![0, 0, P - 1, 0])
        );
        assert_eq!(
            M_neg[(1, 0)],
            Z_N_CycloRaw::<D, P>::from(vec![0, P - 1, 0, 0])
        );
        assert_eq!(
            M_neg[(1, 1)],
            Z_N_CycloRaw::<D, P>::from(vec![P - 1, 0, 0, 0])
        );
    }
}
