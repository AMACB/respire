//! The cyclotomic ring `Z_n[x]/x^d + 1)`. `d` is assumed to be a power of `2`.

use crate::math::gadget::{IntModDecomposition, RingElementDecomposable};
use crate::math::int_mod::{IntMod, NoReduce};
use crate::math::int_mod_crt::IntModCRT;
use crate::math::int_mod_cyclo_crt::IntModCycloCRT;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::int_mod_poly::IntModPoly;
use crate::math::matrix::Matrix;
use crate::math::ntt::*;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use crate::math::simd_utils::Aligned32;
use rand::Rng;
use std::cmp::max;
use std::iter;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};

/// The raw (coefficient) representation of an element of a cyclotomic ring.
///
/// Internally, this is an array of coefficients where the `i`th index corresponds to `x^i`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct IntModCyclo<const D: usize, const N: u64> {
    pub coeff: [IntMod<N>; D],
}

impl<const D: usize, const N: u64> IntModCyclo<D, N> {
    pub fn into_aligned(self) -> Aligned32<[IntMod<N>; D]> {
        Aligned32(self.coeff)
    }
}

/// Conversions

impl<const D: usize, const N: u64> From<u64> for IntModCyclo<D, N> {
    fn from(a: u64) -> Self {
        let mut result = Self::zero();
        result.coeff[0] = a.into();
        result
    }
}

impl<const D: usize, const N: u64> From<i64> for IntModCyclo<D, N> {
    fn from(a: i64) -> Self {
        let mut result = Self::zero();
        result.coeff[0] = a.into();
        result
    }
}

impl<const D: usize, const N: u64> From<[IntMod<N>; D]> for IntModCyclo<D, N> {
    fn from(coeff: [IntMod<N>; D]) -> Self {
        Self { coeff }
    }
}

impl<const D: usize, const N: u64> From<IntModPoly<N>> for IntModCyclo<D, N> {
    fn from(polynomial: IntModPoly<N>) -> Self {
        let mut coeff: [IntMod<N>; D] = [0_u64.into(); D];
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

impl<const D: usize, const N: u64> From<Vec<u64>> for IntModCyclo<D, N> {
    fn from(coeff: Vec<u64>) -> Self {
        IntModCyclo::from(IntModPoly::from(coeff))
    }
}

impl<const D: usize, const N: u64> From<Vec<i64>> for IntModCyclo<D, N> {
    fn from(coeff: Vec<i64>) -> Self {
        IntModCyclo::from(IntModPoly::from(coeff))
    }
}

impl<const D: usize, const N: u64> From<Vec<IntMod<N>>> for IntModCyclo<D, N> {
    fn from(coeff: Vec<IntMod<N>>) -> Self {
        IntModCyclo::from(IntModPoly::from(coeff))
    }
}

impl<const D: usize, const N: u64> TryFrom<&IntModCyclo<D, N>> for IntMod<N> {
    type Error = ();

    /// Inverse of `From<u64>`. Errors if element is not a constant.
    fn try_from(a: &IntModCyclo<D, N>) -> Result<Self, Self::Error> {
        for i in 1..D {
            if a.coeff[i] != IntMod::zero() {
                return Err(());
            }
        }
        Ok(a.coeff[0])
    }
}

impl<
        const D: usize,
        const N1: u64,
        const N2: u64,
        const N1_INV: u64,
        const N2_INV: u64,
        const N: u64,
    > From<&IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>> for IntModCyclo<D, N>
{
    fn from(a: &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>) -> Self {
        assert_eq!(N, N1 * N2);
        let mut result = IntModCyclo::zero();
        for i in 0..D {
            result.coeff[i] = IntMod::from(u64::from(IntModCRT::<N1, N2, N1_INV, N2_INV>::from((
                a.proj1[i], a.proj2[i],
            ))));
        }
        result
    }
}

impl<
        const D: usize,
        const N1: u64,
        const N2: u64,
        const N1_INV: u64,
        const N2_INV: u64,
        const N: u64,
    > From<&IntModCycloCRTEval<D, N1, N2, N1_INV, N2_INV>> for IntModCyclo<D, N>
{
    fn from(a: &IntModCycloCRTEval<D, N1, N2, N1_INV, N2_INV>) -> Self {
        IntModCyclo::from(&IntModCycloCRT::from(a))
    }
}

impl<const D: usize, const N: u64> From<IntModCycloEval<D, N>> for IntModCyclo<D, N> {
    fn from(a_eval: IntModCycloEval<D, N>) -> Self {
        let mut values_aligned = a_eval.into_aligned();
        ntt_neg_backward::<D, N>(&mut values_aligned);
        IntModCyclo::from(values_aligned.0)
    }
}

// TODO: this does a clone, which the user may not be aware about...
impl<const D: usize, const N: u64> From<&IntModCyclo<D, N>> for IntModCycloEval<D, N> {
    fn from(a: &IntModCyclo<D, N>) -> Self {
        (a.clone()).into()
    }
}

/// [`RingElementRef`] implementation

impl<const D: usize, const N: u64> RingElementRef<IntModCyclo<D, N>> for &IntModCyclo<D, N> {}

impl<const D: usize, const N: u64> Add for &IntModCyclo<D, N> {
    type Output = IntModCyclo<D, N>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result_coeff: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_coeff[i] = self.coeff[i] + rhs.coeff[i];
        }
        result_coeff.into()
    }
}

impl<const D: usize, const N: u64> Sub for &IntModCyclo<D, N> {
    type Output = IntModCyclo<D, N>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result_coeff: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_coeff[i] = self.coeff[i] - rhs.coeff[i];
        }
        result_coeff.into()
    }
}

impl<const D: usize, const N: u64> Mul for &IntModCyclo<D, N> {
    type Output = IntModCyclo<D, N>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut self_poly: IntModPoly<N> = self.coeff.to_vec().into();
        let rhs_poly: IntModPoly<N> = rhs.coeff.to_vec().into();
        self_poly *= &rhs_poly;
        self_poly.into()
    }
}

impl<const D: usize, const N: u64> Neg for &IntModCyclo<D, N> {
    type Output = IntModCyclo<D, N>;
    fn neg(self) -> Self::Output {
        let mut result_coeff: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_coeff[i] = -self.coeff[i];
        }
        result_coeff.into()
    }
}

/// [`RingElement`] implementation

impl<const D: usize, const N: u64> RingElement for IntModCyclo<D, N> {
    fn zero() -> IntModCyclo<D, N> {
        [IntMod::zero(); D].into()
    }
    fn one() -> IntModCyclo<D, N> {
        let mut result = Self::zero();
        result.coeff[0] = IntMod::one();
        result
    }
}

impl<'a, const D: usize, const N: u64> AddAssign<&'a Self> for IntModCyclo<D, N> {
    fn add_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.coeff[i] += rhs.coeff[i];
        }
    }
}

impl<'a, const D: usize, const N: u64> SubAssign<&'a Self> for IntModCyclo<D, N> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.coeff[i] -= rhs.coeff[i];
        }
    }
}

impl<const D: usize, const N: u64> MulAssign<IntMod<N>> for IntModCyclo<D, N> {
    fn mul_assign(&mut self, rhs: IntMod<N>) {
        for i in 0..D {
            self.coeff[i] *= rhs;
        }
    }
}

impl<'a, const D: usize, const N: u64> MulAssign<&'a Self> for IntModCyclo<D, N> {
    fn mul_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

impl<const D: usize, const NN: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for IntModCyclo<D, NN>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let mut decomps = Vec::<IntModDecomposition<BASE, LEN>>::with_capacity(D);
        for coeff_idx in 0..D {
            decomps.push(IntModDecomposition::<BASE, LEN>::new(
                u64::from(self.coeff[coeff_idx]),
                NN,
            ));
        }
        for k in 0..LEN {
            for coeff_idx in 0..D {
                mat[(i + k, j)].coeff[coeff_idx] = IntMod::from(decomps[coeff_idx].next().unwrap());
            }
        }
    }
}

/// Misc

impl<const D: usize, const N: u64> IntModCyclo<D, N> {
    /// Compute the automorphism x --> x^k. This only makes sense for odd `k`.
    pub fn auto(&self, k: usize) -> Self {
        // TODO test this
        let mut result = IntModCyclo::zero();
        for i in 0..D {
            let pow = (i * k) % (2 * D);
            let neg = pow >= D;
            let reduced_pow = if !neg { pow } else { pow - D };
            result.coeff[reduced_pow] = if neg { -self.coeff[i] } else { self.coeff[i] };
        }
        result
    }

    /// Multiply by x^k
    pub fn mul_x_pow(&self, k: usize) -> Self {
        let mut result = Self::zero();
        let k_reduced = k % D;
        let neg = (k % (2 * D)) >= D;
        for i in 0..k_reduced {
            result.coeff[i] = if neg {
                // Double negate
                self.coeff[D - k_reduced + i]
            } else {
                -self.coeff[D - k_reduced + i]
            }
        }
        for i in k_reduced..D {
            result.coeff[i] = if neg {
                -self.coeff[i - k_reduced]
            } else {
                self.coeff[i - k_reduced]
            }
        }
        result
    }

    /// Applies `Z_N::scale_up_into` coefficient-wise.
    pub fn scale_up_into<const M: u64>(&self) -> IntModCyclo<D, M> {
        let mut result = IntModCyclo::zero();
        for i in 0..D {
            result.coeff[i] = self.coeff[i].scale_up_into();
        }
        result
    }

    /// Applies `Z_N::include_into` coefficient-wise.
    pub fn include_into<const M: u64>(&self) -> IntModCyclo<D, M> {
        let mut result = IntModCyclo::zero();
        for i in 0..D {
            result.coeff[i] = self.coeff[i].include_into();
        }
        result
    }

    /// Applies `Z_N::project_into` coefficient-wise.
    pub fn project_into<const M: u64>(&self) -> IntModCyclo<D, M> {
        let mut result = IntModCyclo::zero();
        for i in 0..D {
            result.coeff[i] = self.coeff[i].project_into();
        }
        result
    }

    /// Applies `Z_N::round_down_into` coefficient-wise.
    pub fn round_down_into<const M: u64>(&self) -> IntModCyclo<D, M> {
        let mut result = IntModCyclo::zero();
        for i in 0..D {
            result.coeff[i] = self.coeff[i].round_down_into();
        }
        result
    }

    pub fn project_dim<const D_SMALL: usize>(&self) -> IntModCyclo<D_SMALL, N> {
        assert_eq!(D % D_SMALL, 0);
        let result_coeff_vec: Vec<IntMod<N>> =
            self.coeff.iter().step_by(D / D_SMALL).copied().collect();
        let result_coeff: [IntMod<N>; D_SMALL] = result_coeff_vec.try_into().unwrap();
        IntModCyclo::from(result_coeff)
    }

    pub fn include_dim<const D_LARGE: usize>(&self) -> IntModCyclo<D_LARGE, N> {
        assert_eq!(D_LARGE % D, 0);
        let result_coeff_vec: Vec<IntMod<N>> = self
            .coeff
            .iter()
            .flat_map(|x| {
                [*x].into_iter()
                    .chain(iter::repeat(IntMod::zero()).take(D_LARGE / D - 1))
            })
            .collect();
        let result_coeff: [IntMod<N>; D_LARGE] = result_coeff_vec.try_into().unwrap();
        IntModCyclo::from(result_coeff)
    }
}

unsafe impl<const D: usize, const N: u64, const M: u64> RingCompatible<IntModCyclo<D, M>>
    for IntModCyclo<D, N>
{
}

/// Random sampling

impl<const D: usize, const N: u64> RandUniformSampled for IntModCyclo<D, N> {
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..D {
            result.coeff[i] = IntMod::<N>::rand_uniform(rng);
        }
        result
    }
}

impl<const D: usize, const N: u64> RandZeroOneSampled for IntModCyclo<D, N> {
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

impl<const D: usize, const N: u64> RandDiscreteGaussianSampled for IntModCyclo<D, N> {
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..D {
            result.coeff[i] = IntMod::<N>::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(rng);
        }
        result
    }
}

/// Other polynomial-specific operations.

impl<const D: usize, const N: u64> NormedRingElement for IntModCyclo<D, N> {
    fn norm(&self) -> u64 {
        let mut worst: u64 = 0;
        for i in 0..D {
            worst = max(worst, self.coeff[i].norm());
        }
        worst
    }
}

impl<const D: usize, const N: u64> Index<usize> for IntModCyclo<D, N> {
    type Output = IntMod<N>;
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
        let p = IntModCyclo::<D, P>::from(vec![42_u64, 6, 1, 0, 5]);
        let q = IntModCyclo::<D, P>::from(vec![37_u64, 6, 1, 0]);
        let r = IntModCyclo::<D, P>::from(vec![41_u64, 6, 1, 0, 5, 0, 0, 0, 1]);
        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);

        let s = IntModCyclo::<D, P>::from(vec![9483_i64, 1, 1, 1, 323, -12139, 10491, 1, 1]);
        let t = IntModCyclo::<D, P>::from(vec![9161_i64, 12140, -10490, 0, 0]);
        assert_eq!(s, t);
    }

    #[test]
    fn test_mul_x_pow() {
        let x0 = IntModCyclo::<D, P>::from(vec![1_u64, 0, 0, 0]);
        let x1 = IntModCyclo::<D, P>::from(vec![0_u64, 1, 0, 0]);
        let x2 = IntModCyclo::<D, P>::from(vec![0_u64, 0, 1, 0]);
        let x3 = IntModCyclo::<D, P>::from(vec![0_u64, 0, 0, 1]);
        for off in [0, 8, 16] {
            assert_eq!(x1.mul_x_pow(off), x1);
            assert_eq!(x1.mul_x_pow(1 + off), x2);
            assert_eq!(x1.mul_x_pow(2 + off), x3);
            assert_eq!(x1.mul_x_pow(3 + off), -&x0);
            assert_eq!(x1.mul_x_pow(4 + off), -&x1);
            assert_eq!(x1.mul_x_pow(5 + off), -&x2);
            assert_eq!(x1.mul_x_pow(6 + off), -&x3);
            assert_eq!(x1.mul_x_pow(7 + off), x0);
        }

        let p = IntModCyclo::<D, P>::from(vec![1_u64, 2, 3, 4]);
        let q = IntModCyclo::<D, P>::from(vec![-3_i64, -4, 1, 2]);
        assert_eq!(p.mul_x_pow(2), q);
        assert_eq!(p.mul_x_pow(6), -&q);
        assert_eq!(p.mul_x_pow(10), q);
    }

    #[test]
    fn test_ops() {
        let p = IntModCyclo::<D, P>::from(vec![0_u64, 0, 0, 1]);
        let q = IntModCyclo::<D, P>::from(vec![0_u64, 0, 2, 0]);
        let sum = IntModCyclo::<D, P>::from(vec![0_u64, 0, 2, 1]);
        let diff = IntModCyclo::<D, P>::from(vec![0_i64, 0, -2, 1]);
        let prod = IntModCyclo::<D, P>::from(vec![0_i64, -2, 0, 0]);
        let square = IntModCyclo::<D, P>::from(vec![0_i64, 0, -1, 0]);
        let neg = IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, -1]);
        assert_eq!(&p + &q, sum);
        assert_eq!(&p - &q, diff);
        assert_eq!(&p * &q, prod);
        assert_eq!(&p * &p, square);
        assert_eq!(-&p, neg);
    }

    #[test]
    fn test_scale_round() {
        type R31 = IntModCyclo<4, 31>;
        type R1000 = IntModCyclo<4, 1000>;

        const MAX_ERROR: i64 = ((1000 / 31) - 1) / 2;
        assert_eq!(MAX_ERROR, 15);

        let orig: R31 = vec![5_u64, 30_u64, 11_u64, 0_u64].into();
        let scaled_with_error: R1000 =
            &orig.scale_up_into() + &vec![-MAX_ERROR, MAX_ERROR, MAX_ERROR / 2, -MAX_ERROR].into();
        let recovered: R31 = scaled_with_error.round_down_into();
        assert_eq!(orig, recovered);
    }

    #[test]
    fn test_convert() {
        type R31 = IntModCyclo<4, 31>;
        type R0 = IntModCyclo<4, 0>;
        let x: R31 = vec![1_u64, 2_u64, 3_u64, 4_u64].into();
        let mut x: R0 = x.convert();
        x += &vec![10_u64, 10_u64, 10_u64, 10_u64].into();
        let x: R31 = x.convert();
        assert_eq!(x, vec![11_u64, 12_u64, 13_u64, 14_u64].into())
    }

    #[test]
    fn test_matrix() {
        let mut m: Matrix<2, 2, IntModCyclo<D, P>> = Matrix::zero();
        m[(0, 0)] = IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, 1]);
        m[(0, 1)] = IntModCyclo::<D, P>::from(vec![0_i64, 0, 1, 0]);
        m[(1, 0)] = IntModCyclo::<D, P>::from(vec![0_i64, 1, 0, 0]);
        m[(1, 1)] = IntModCyclo::<D, P>::from(vec![1_i64, 0, 0, 0]);
        // m =
        // [ x^3 x^2 ]
        // [ x   1   ]
        let m_square = &m * &m;
        assert_eq!(
            m_square[(0, 0)],
            IntModCyclo::<D, P>::from(vec![0_i64, 0, -1, 1])
        ); // x^3 + x^6
        assert_eq!(
            m_square[(0, 1)],
            IntModCyclo::<D, P>::from(vec![0_i64, -1, 1, 0])
        ); // x^2 + x^5
        assert_eq!(
            m_square[(1, 0)],
            IntModCyclo::<D, P>::from(vec![-1_i64, 1, 0, 0])
        ); // x + x^4
        assert_eq!(
            m_square[(1, 1)],
            IntModCyclo::<D, P>::from(vec![1_i64, 0, 0, 1])
        ); // 1 + x^3

        let m_double = &m + &m;
        assert_eq!(
            m_double[(0, 0)],
            IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, 2])
        );
        assert_eq!(
            m_double[(0, 1)],
            IntModCyclo::<D, P>::from(vec![0_i64, 0, 2, 0])
        );
        assert_eq!(
            m_double[(1, 0)],
            IntModCyclo::<D, P>::from(vec![0_i64, 2, 0, 0])
        );
        assert_eq!(
            m_double[(1, 1)],
            IntModCyclo::<D, P>::from(vec![2_i64, 0, 0, 0])
        );

        let m_neg = -&m;
        assert_eq!(
            m_neg[(0, 0)],
            IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, -1])
        );
        assert_eq!(
            m_neg[(0, 1)],
            IntModCyclo::<D, P>::from(vec![0_i64, 0, -1, 0])
        );
        assert_eq!(
            m_neg[(1, 0)],
            IntModCyclo::<D, P>::from(vec![0_i64, -1, 0, 0])
        );
        assert_eq!(
            m_neg[(1, 1)],
            IntModCyclo::<D, P>::from(vec![-1_i64, 0, 0, 0])
        );
    }
}
