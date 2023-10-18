//! The cyclotomic ring `Z_n[x]/x^d + 1)`, where `n = n_1 * n_2` and `d` is assumed to be a power of `2`.

use crate::math::discrete_gaussian::DiscreteGaussian;
use crate::math::gadget::{IntModDecomposition, RingElementDecomposable};
use crate::math::int_mod::IntMod;
use crate::math::int_mod_crt::IntModCRT;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::int_mod_poly::IntModPoly;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::cmp::max;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO: documentation

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct IntModCycloCRT<const D: usize, const N1: u64, const N2: u64> {
    pub proj1: IntModCyclo<D, N1>,
    pub proj2: IntModCyclo<D, N2>,
}

/// Conversions

impl<const D: usize, const N1: u64, const N2: u64> From<u64> for IntModCycloCRT<D, N1, N2> {
    fn from(a: u64) -> Self {
        Self {
            proj1: a.into(),
            proj2: a.into(),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> From<(IntModCyclo<D, N1>, IntModCyclo<D, N2>)>
    for IntModCycloCRT<D, N1, N2>
{
    fn from(a: (IntModCyclo<D, N1>, IntModCyclo<D, N2>)) -> Self {
        Self {
            proj1: a.0,
            proj2: a.1,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N: u64> From<&IntModCyclo<D, N>>
    for IntModCycloCRT<D, N1, N2>
{
    fn from(a: &IntModCyclo<D, N>) -> Self {
        assert_eq!(N, N1 * N2);
        Self {
            proj1: a.project_into(),
            proj2: a.project_into(),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> From<IntModCycloCRT<D, N1, N2>>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn from(a: IntModCycloCRT<D, N1, N2>) -> Self {
        let p1_ntt: IntModCycloEval<D, N1> = a.proj1.into();
        let p2_ntt: IntModCycloEval<D, N2> = a.proj2.into();
        (p1_ntt, p2_ntt).into()
    }
}

impl<const D: usize, const N1: u64, const N2: u64> From<&IntModCycloCRTEval<D, N1, N2>>
    for IntModCycloCRT<D, N1, N2>
{
    fn from(a: &IntModCycloCRTEval<D, N1, N2>) -> Self {
        let p1_raw: IntModCyclo<D, N1> = (&a.proj1).into();
        let p2_raw: IntModCyclo<D, N2> = (&a.proj2).into();
        (p1_raw, p2_raw).into()
    }
}

// impl<const D: usize, const N1: u64, const N2: u64> From<[Z_N<N>; D]> for Z_N_CycloRaw_CRT<D, N1, N2> {
//     fn from(coeff: [Z_N<N>; D]) -> Self {
//         Self { coeff }
//     }
// }

// impl<const D: usize, const N1: u64, const N2: u64> From<PolynomialZ_N<N>> for Z_N_CycloRaw_CRT<D, N1, N2> {
//     fn from(polynomial: PolynomialZ_N<N>) -> Self {
//         let mut coeff: [Z_N<N>; D] = [0_u64.into(); D];
//         for (i, x) in polynomial.coeff_iter().enumerate() {
//             if i / D % 2 == 0 {
//                 coeff[i % D] += x;
//             } else {
//                 coeff[i % D] -= x;
//             }
//         }
//         coeff.into()
//     }
// }

impl<const D: usize, const N1: u64, const N2: u64> From<Vec<u64>> for IntModCycloCRT<D, N1, N2> {
    fn from(coeff: Vec<u64>) -> Self {
        IntModCycloCRT {
            proj1: IntModCyclo::from(IntModPoly::from(coeff.clone())),
            proj2: IntModCyclo::from(IntModPoly::from(coeff)),
        }
    }
}

/// [`RingElementRef`] implementation
impl<const D: usize, const N1: u64, const N2: u64> RingElementRef<IntModCycloCRT<D, N1, N2>>
    for &IntModCycloCRT<D, N1, N2>
{
}

impl<const D: usize, const N1: u64, const N2: u64> Add for &IntModCycloCRT<D, N1, N2> {
    type Output = IntModCycloCRT<D, N1, N2>;
    fn add(self, rhs: Self) -> Self::Output {
        IntModCycloCRT {
            proj1: &self.proj1 + &rhs.proj1,
            proj2: &self.proj2 + &rhs.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> Sub for &IntModCycloCRT<D, N1, N2> {
    type Output = IntModCycloCRT<D, N1, N2>;
    fn sub(self, rhs: Self) -> Self::Output {
        IntModCycloCRT {
            proj1: &self.proj1 - &rhs.proj1,
            proj2: &self.proj2 - &rhs.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> Mul for &IntModCycloCRT<D, N1, N2> {
    type Output = IntModCycloCRT<D, N1, N2>;
    fn mul(self, rhs: Self) -> Self::Output {
        IntModCycloCRT {
            proj1: &self.proj1 * &rhs.proj1,
            proj2: &self.proj2 * &rhs.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> Neg for &IntModCycloCRT<D, N1, N2> {
    type Output = IntModCycloCRT<D, N1, N2>;
    fn neg(self) -> Self::Output {
        IntModCycloCRT {
            proj1: -&self.proj1,
            proj2: -&self.proj2,
        }
    }
}

/// [`RingElement`] implementation

impl<const D: usize, const N1: u64, const N2: u64> RingElement for IntModCycloCRT<D, N1, N2> {
    fn zero() -> IntModCycloCRT<D, N1, N2> {
        IntModCycloCRT {
            proj1: IntModCyclo::zero(),
            proj2: IntModCyclo::zero(),
        }
    }
    fn one() -> IntModCycloCRT<D, N1, N2> {
        IntModCycloCRT {
            proj1: IntModCyclo::one(),
            proj2: IntModCyclo::one(),
        }
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64> AddAssign<&'a Self>
    for IntModCycloCRT<D, N1, N2>
{
    fn add_assign(&mut self, rhs: &'a Self) {
        self.proj1 += &rhs.proj1;
        self.proj2 += &rhs.proj2;
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64> SubAssign<&'a Self>
    for IntModCycloCRT<D, N1, N2>
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.proj1 -= &rhs.proj1;
        self.proj2 -= &rhs.proj2;
    }
}

// impl<'a, const D: usize, const N1: u64, const N2: u64> MulAssign<Z_N<N>> for Z_N_CycloRaw_CRT<D, N1, N2> {
//     fn mul_assign(&mut self, rhs: Z_N<N>) {
//         self.p1 *= &rhs.p1;
//         self.p2 *= &rhs.p2;
//     }
// }

impl<'a, const D: usize, const N1: u64, const N2: u64> MulAssign<&'a Self>
    for IntModCycloCRT<D, N1, N2>
{
    fn mul_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for IntModCycloCRT<D, N1, N2>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        for coeff_idx in 0..D {
            let coeff = u64::from(IntModCRT::<N1, N2>::from((
                self.proj1.coeff[coeff_idx],
                self.proj2.coeff[coeff_idx],
            )));
            let decomp = IntModDecomposition::<BASE, LEN>::new(coeff, N1 * N2);
            for (k, u) in decomp.enumerate() {
                let u_crt = IntModCRT::<N1, N2>::from(u);
                mat[(i + k, j)].proj1.coeff[coeff_idx] = u_crt.proj1;
                mat[(i + k, j)].proj2.coeff[coeff_idx] = u_crt.proj2;
            }
        }
    }
}

/// Random sampling

impl<const D: usize, const N1: u64, const N2: u64> RandUniformSampled
    for IntModCycloCRT<D, N1, N2>
{
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        IntModCycloCRT {
            proj1: IntModCyclo::rand_uniform(rng),
            proj2: IntModCyclo::rand_uniform(rng),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> RandZeroOneSampled
    for IntModCycloCRT<D, N1, N2>
{
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        let mut v = vec![0u64; D];
        for i in 0..(D / 64) {
            let rand = rng.gen::<u64>();
            for bit in 0..64 {
                v[i * 64 + bit] = (rand >> bit) & 1;
            }
        }

        let rand = rng.gen::<u64>();
        for bit in 0..(D % 64) {
            v[(D / 64) * 64 + bit] = (rand >> bit) & 1;
        }
        IntModCycloCRT {
            proj1: v.clone().into(),
            proj2: v.into(),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> RandDiscreteGaussianSampled
    for IntModCycloCRT<D, N1, N2>
{
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        let mut v = vec![0i64; D];
        v.iter_mut()
            .map(|x| *x = DiscreteGaussian::sample::<_, NOISE_WIDTH_MILLIONTHS>(rng))
            .count();
        IntModCycloCRT {
            proj1: v.clone().into(),
            proj2: v.into(),
        }
    }
}

// Other polynomial-specific operations.

impl<const D: usize, const N1: u64, const N2: u64> IntModCycloCRT<D, N1, N2> {
    pub fn auto(&self, k: usize) -> Self {
        (self.proj1.auto(k), self.proj2.auto(k)).into()
    }

    pub fn norm(&self) -> u64 {
        let mut worst: u64 = 0;
        for i in 0..D {
            let val: IntModCRT<N1, N2> = (self.proj1[i], self.proj2[i]).into();
            worst = max(worst, val.norm());
        }
        worst
    }

    /// Applies `Z_N::round_down_into` coefficient-wise.
    pub fn round_down_into<const M: u64>(&self) -> IntModCyclo<D, M> {
        let mut rounded_coeffs = [IntMod::zero(); D];
        for (idx, out) in rounded_coeffs.iter_mut().enumerate() {
            let coeff: IntModCRT<N1, N2> = (self.proj1[idx], self.proj2[idx]).into();
            *out = coeff.round_down_into();
        }
        rounded_coeffs.into()
    }
}

unsafe impl<const D: usize, const N1: u64, const N2: u64, const M1: u64, const M2: u64>
    RingCompatible<IntModCycloCRT<D, M1, M2>> for IntModCycloCRT<D, N1, N2>
{
}

// TODO: this should be a TryFrom
impl<const D: usize, const N1: u64, const N2: u64> From<&IntModCycloCRT<D, N1, N2>>
    for IntModCRT<N1, N2>
{
    fn from(a: &IntModCycloCRT<D, N1, N2>) -> Self {
        (a.proj1[0], a.proj2[0]).into()
    }
}

// impl<const D: usize, const N1: u64, const N2: u64> Index<usize> for Z_N_CycloRaw_CRT<D, N1, N2> {
//     type Output = (Z_N<N1>, Z_N<N2>);
//     fn index(&self, index: usize) -> &Self::Output {
//         &(self.p1[0], self.p2[0])
//     }
// }

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::math::matrix::Matrix;

//     const D: usize = 4; // Z_q[X] / (X^4 + 1)
//     const P: u64 = (1_u64 << 32) - 5;

//     // TODO: add more tests.
//     #[test]
//     fn test_from_into() {
//         let p = Z_N_CycloRaw_CRT::<D, P>::from(vec![42, 6, 1, 0, 5]);
//         let q = Z_N_CycloRaw_CRT::<D, P>::from(vec![37, 6, 1, 0]);
//         let r = Z_N_CycloRaw_CRT::<D, P>::from(vec![41, 6, 1, 0, 5, 0, 0, 0, 1]);
//         assert_eq!(p, q);
//         assert_eq!(p, r);
//         assert_eq!(q, r);

//         let s = Z_N_CycloRaw_CRT::<D, P>::from(vec![9483, 1, 1, 1, 323, P - 12139, 10491, 1, 1]);
//         let t = Z_N_CycloRaw_CRT::<D, P>::from(vec![9161, 12140, P - 10490, 0, 0]);
//         assert_eq!(s, t);
//     }

//     #[test]
//     fn test_ops() {
//         let p = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 0, 1]);
//         let q = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 2, 0]);
//         let sum = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 2, 1]);
//         let diff = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, P - 2, 1]);
//         let prod = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, P - 2, 0, 0]);
//         let square = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, P - 1, 0]);
//         let neg = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 0, P - 1]);
//         assert_eq!(&p + &q, sum);
//         assert_eq!(&p - &q, diff);
//         assert_eq!(&p * &q, prod);
//         assert_eq!(&p * &p, square);
//         assert_eq!(-&p, neg);
//     }

//     #[test]
//     fn test_matrix() {
//         let mut M: Matrix<2, 2, Z_N_CycloRaw_CRT<D, P>> = Matrix::zero();
//         M[(0, 0)] = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 0, 1]);
//         M[(0, 1)] = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 1, 0]);
//         M[(1, 0)] = Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 1, 0, 0]);
//         M[(1, 1)] = Z_N_CycloRaw_CRT::<D, P>::from(vec![1, 0, 0, 0]);
//         // M =
//         // [ x^3 x^2 ]
//         // [ x   1   ]
//         let M_square = &M * &M;
//         assert_eq!(
//             M_square[(0, 0)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, P - 1, 1])
//         ); // x^3 + x^6
//         assert_eq!(
//             M_square[(0, 1)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, P - 1, 1, 0])
//         ); // x^2 + x^5
//         assert_eq!(
//             M_square[(1, 0)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![P - 1, 1, 0, 0])
//         ); // x + x^4
//         assert_eq!(
//             M_square[(1, 1)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![1, 0, 0, 1])
//         ); // 1 + x^3

//         let M_double = &M + &M;
//         assert_eq!(
//             M_double[(0, 0)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 0, 2])
//         );
//         assert_eq!(
//             M_double[(0, 1)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 2, 0])
//         );
//         assert_eq!(
//             M_double[(1, 0)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 2, 0, 0])
//         );
//         assert_eq!(
//             M_double[(1, 1)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![2, 0, 0, 0])
//         );

//         let M_neg = -&M;
//         assert_eq!(
//             M_neg[(0, 0)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, 0, P - 1])
//         );
//         assert_eq!(
//             M_neg[(0, 1)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, 0, P - 1, 0])
//         );
//         assert_eq!(
//             M_neg[(1, 0)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![0, P - 1, 0, 0])
//         );
//         assert_eq!(
//             M_neg[(1, 1)],
//             Z_N_CycloRaw_CRT::<D, P>::from(vec![P - 1, 0, 0, 0])
//         );
//     }
// }
