use crate::math::gadget::RingElementDecomposable;
use crate::math::int_mod::IntMod;
use crate::math::int_mod_crt::IntModCRT;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt::IntModCycloCRT;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::int_mod_poly::IntModPoly;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO: documentation

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct IntModCycloCRTEval<const D: usize, const N1: u64, const N2: u64> {
    pub proj1: IntModCycloEval<D, N1>,
    pub proj2: IntModCycloEval<D, N2>,
}

/// Conversions

impl<const D: usize, const N1: u64, const N2: u64> From<u64> for IntModCycloCRTEval<D, N1, N2> {
    fn from(a: u64) -> Self {
        Self {
            proj1: a.into(),
            proj2: a.into(),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64>
    From<(IntModCycloEval<D, N1>, IntModCycloEval<D, N2>)> for IntModCycloCRTEval<D, N1, N2>
{
    fn from(a: (IntModCycloEval<D, N1>, IntModCycloEval<D, N2>)) -> Self {
        IntModCycloCRTEval {
            proj1: a.0,
            proj2: a.1,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> From<IntModCycloCRTEval<D, N1, N2>>
    for IntModCycloCRT<D, N1, N2>
{
    fn from(a: IntModCycloCRTEval<D, N1, N2>) -> Self {
        let p1_raw: IntModCyclo<D, N1> = a.proj1.into();
        let p2_raw: IntModCyclo<D, N2> = a.proj2.into();
        (p1_raw, p2_raw).into()
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N: u64> From<&IntModCyclo<D, N>>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn from(a: &IntModCyclo<D, N>) -> Self {
        IntModCycloCRTEval::from(&IntModCycloCRT::from(a))
    }
}

impl<const D: usize, const N1: u64, const N2: u64> From<&IntModCycloCRT<D, N1, N2>>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn from(a: &IntModCycloCRT<D, N1, N2>) -> Self {
        let p1_eval: IntModCycloEval<D, N1> = (&a.proj1).into();
        let p2_eval: IntModCycloEval<D, N2> = (&a.proj2).into();
        (p1_eval, p2_eval).into()
    }
}

impl<const D: usize, const N1: u64, const N2: u64> From<Vec<u64>>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn from(coeff: Vec<u64>) -> Self {
        IntModCycloCRTEval {
            proj1: IntModCycloEval::from(IntModPoly::from(coeff.clone())),
            proj2: IntModCycloEval::from(IntModPoly::from(coeff)),
        }
    }
}

/// [`RingElementRef`] implementation

impl<const D: usize, const N1: u64, const N2: u64> RingElementRef<IntModCycloCRTEval<D, N1, N2>>
    for &IntModCycloCRTEval<D, N1, N2>
{
}

impl<const D: usize, const N1: u64, const N2: u64> Add for &IntModCycloCRTEval<D, N1, N2> {
    type Output = IntModCycloCRTEval<D, N1, N2>;
    fn add(self, rhs: Self) -> Self::Output {
        IntModCycloCRTEval {
            proj1: &self.proj1 + &rhs.proj1,
            proj2: &self.proj2 + &rhs.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> Sub for &IntModCycloCRTEval<D, N1, N2> {
    type Output = IntModCycloCRTEval<D, N1, N2>;
    fn sub(self, rhs: Self) -> Self::Output {
        IntModCycloCRTEval {
            proj1: &self.proj1 - &rhs.proj1,
            proj2: &self.proj2 - &rhs.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> Mul for &IntModCycloCRTEval<D, N1, N2> {
    type Output = IntModCycloCRTEval<D, N1, N2>;
    fn mul(self, rhs: Self) -> Self::Output {
        IntModCycloCRTEval {
            proj1: &self.proj1 * &rhs.proj1,
            proj2: &self.proj2 * &rhs.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> Neg for &IntModCycloCRTEval<D, N1, N2> {
    type Output = IntModCycloCRTEval<D, N1, N2>;
    fn neg(self) -> Self::Output {
        IntModCycloCRTEval {
            proj1: -&self.proj1,
            proj2: -&self.proj2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64> IntModCycloCRTEval<D, N1, N2> {
    fn add_eq_mul_fallback(&mut self, a: &Self, b: &Self) {
        for i in 0..D {
            self.proj1.evals[i] += a.proj1.evals[i] * b.proj1.evals[i];
            self.proj2.evals[i] += a.proj2.evals[i] * b.proj2.evals[i];
        }
    }
}

/// [`RingElement`] implementation

impl<const D: usize, const N1: u64, const N2: u64> RingElement for IntModCycloCRTEval<D, N1, N2> {
    fn zero() -> IntModCycloCRTEval<D, N1, N2> {
        IntModCycloCRTEval {
            proj1: IntModCycloEval::zero(),
            proj2: IntModCycloEval::zero(),
        }
    }
    fn one() -> IntModCycloCRTEval<D, N1, N2> {
        IntModCycloCRTEval {
            proj1: IntModCycloEval::one(),
            proj2: IntModCycloEval::one(),
        }
    }

    #[cfg(not(target_feature = "avx2"))]
    fn add_eq_mul(&mut self, a: &Self, b: &Self) {
        self.add_eq_mul_fallback(a, b);
    }

    #[cfg(target_feature = "avx2")]
    fn add_eq_mul(&mut self, a: &Self, b: &Self) {
        if N1 != 0 || N2 != 0 || D % 4 != 0 {
            return self.add_eq_mul_fallback(a, b);
        }

        use crate::math::simd_utils::*;
        use std::arch::x86_64::*;
        unsafe {
            for i in 0..D / 4 {
                let a_p1_ptr =
                    a.proj1.evals.get_unchecked(4 * i) as *const IntMod<N1> as *const __m256i;
                let b_p1_ptr =
                    b.proj1.evals.get_unchecked(4 * i) as *const IntMod<N1> as *const __m256i;
                let self_p1_ptr =
                    self.proj1.evals.get_unchecked_mut(4 * i) as *mut IntMod<N1> as *mut __m256i;
                _mm256_ptr_add_eq_mul32(self_p1_ptr, a_p1_ptr, b_p1_ptr);
            }
            for i in 0..D / 4 {
                let a_p2_ptr =
                    a.proj2.evals.get_unchecked(4 * i) as *const IntMod<N2> as *const __m256i;
                let b_p2_ptr =
                    b.proj2.evals.get_unchecked(4 * i) as *const IntMod<N2> as *const __m256i;
                let self_p2_ptr =
                    self.proj2.evals.get_unchecked_mut(4 * i) as *mut IntMod<N2> as *mut __m256i;
                _mm256_ptr_add_eq_mul32(self_p2_ptr, a_p2_ptr, b_p2_ptr);
            }
        }
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64> AddAssign<&'a Self>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn add_assign(&mut self, rhs: &'a Self) {
        self.proj1 += &rhs.proj1;
        self.proj2 += &rhs.proj2;
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64> SubAssign<&'a Self>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.proj1 -= &rhs.proj1;
        self.proj2 -= &rhs.proj2;
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64> MulAssign<&'a Self>
    for IntModCycloCRTEval<D, N1, N2>
{
    fn mul_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.proj1.evals[i] *= rhs.proj1.evals[i];
            self.proj2.evals[i] *= rhs.proj2.evals[i];
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for IntModCycloCRTEval<D, N1, N2>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        // TODO: do the decomposition and/or NTTs in place
        let self_coeff = IntModCycloCRT::<D, N1, N2>::from(self);
        let mut tmp = Matrix::<LEN, 1, IntModCycloCRT<D, N1, N2>>::zero();
        <IntModCycloCRT<D, N1, N2> as RingElementDecomposable<BASE, LEN>>::decompose_into_mat(
            &self_coeff,
            &mut tmp,
            0,
            0,
        );
        for k in 0..LEN {
            mat[(i + k, j)] = IntModCycloCRTEval::from(tmp[(k, 0)].clone());
        }
    }
}

/// Random sampling

impl<const D: usize, const N1: u64, const N2: u64> RandUniformSampled
    for IntModCycloCRTEval<D, N1, N2>
{
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        IntModCycloCRTEval {
            proj1: IntModCycloEval::rand_uniform(rng),
            proj2: IntModCycloEval::rand_uniform(rng),
        }
    }
}

// Other polynomial-specific operations.

impl<const D: usize, const N1: u64, const N2: u64> IntModCycloCRTEval<D, N1, N2> {
    pub fn norm(&self) -> u64 {
        let p: IntModCycloCRT<D, N1, N2> = self.into();
        p.norm()
    }
}

impl<const D: usize, const N1: u64, const N2: u64> IntModCycloCRTEval<D, N1, N2> {
    pub fn reduce_mod(a: &mut IntModCycloCRTEval<D, 0, 0>) {
        for i in 0..D {
            a.proj1.evals[i] = IntMod::<N1>::from(u64::from(a.proj1.evals[i])).convert();
            a.proj2.evals[i] = IntMod::<N2>::from(u64::from(a.proj2.evals[i])).convert();
        }
    }

    pub fn auto(&self, k: usize) -> Self {
        (self.proj1.auto(k), self.proj2.auto(k)).into()
    }

    pub fn mul_x_pow(&self, k: usize) -> Self {
        (self.proj1.mul_x_pow(k), self.proj2.mul_x_pow(k)).into()
    }
}

// TODO: this should be a TryFrom
impl<const D: usize, const N1: u64, const N2: u64> From<&IntModCycloCRTEval<D, N1, N2>>
    for IntModCRT<N1, N2>
{
    fn from(a: &IntModCycloCRTEval<D, N1, N2>) -> Self {
        let p: IntModCycloCRT<D, N1, N2> = a.into();
        (&p).into()
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::math::matrix::Matrix;

//     const D: usize = 4; // Z_q[X] / (X^4 + 1)
//     const P: u64 = (1_u64 << 32) - 5;

//     // TODO: add more tests.
//     #[test]
//     fn test_from_into() {
//         let p = Z_N_CycloNTT_CRT::<D, P>::from(vec![42, 6, 1, 0, 5]);
//         let q = Z_N_CycloNTT_CRT::<D, P>::from(vec![37, 6, 1, 0]);
//         let r = Z_N_CycloNTT_CRT::<D, P>::from(vec![41, 6, 1, 0, 5, 0, 0, 0, 1]);
//         assert_eq!(p, q);
//         assert_eq!(p, r);
//         assert_eq!(q, r);

//         let s = Z_N_CycloNTT_CRT::<D, P>::from(vec![9483, 1, 1, 1, 323, P - 12139, 10491, 1, 1]);
//         let t = Z_N_CycloNTT_CRT::<D, P>::from(vec![9161, 12140, P - 10490, 0, 0]);
//         assert_eq!(s, t);
//     }

//     #[test]
//     fn test_ops() {
//         let p = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 0, 1]);
//         let q = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 2, 0]);
//         let sum = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 2, 1]);
//         let diff = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, P - 2, 1]);
//         let prod = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, P - 2, 0, 0]);
//         let square = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, P - 1, 0]);
//         let neg = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 0, P - 1]);
//         assert_eq!(&p + &q, sum);
//         assert_eq!(&p - &q, diff);
//         assert_eq!(&p * &q, prod);
//         assert_eq!(&p * &p, square);
//         assert_eq!(-&p, neg);
//     }

//     #[test]
//     fn test_matrix() {
//         let mut M: Matrix<2, 2, Z_N_CycloNTT_CRT<D, P>> = Matrix::zero();
//         M[(0, 0)] = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 0, 1]);
//         M[(0, 1)] = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 1, 0]);
//         M[(1, 0)] = Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 1, 0, 0]);
//         M[(1, 1)] = Z_N_CycloNTT_CRT::<D, P>::from(vec![1, 0, 0, 0]);
//         // M =
//         // [ x^3 x^2 ]
//         // [ x   1   ]
//         let M_square = &M * &M;
//         assert_eq!(
//             M_square[(0, 0)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, P - 1, 1])
//         ); // x^3 + x^6
//         assert_eq!(
//             M_square[(0, 1)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, P - 1, 1, 0])
//         ); // x^2 + x^5
//         assert_eq!(
//             M_square[(1, 0)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![P - 1, 1, 0, 0])
//         ); // x + x^4
//         assert_eq!(
//             M_square[(1, 1)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![1, 0, 0, 1])
//         ); // 1 + x^3

//         let M_double = &M + &M;
//         assert_eq!(
//             M_double[(0, 0)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 0, 2])
//         );
//         assert_eq!(
//             M_double[(0, 1)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 2, 0])
//         );
//         assert_eq!(
//             M_double[(1, 0)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 2, 0, 0])
//         );
//         assert_eq!(
//             M_double[(1, 1)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![2, 0, 0, 0])
//         );

//         let M_neg = -&M;
//         assert_eq!(
//             M_neg[(0, 0)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, 0, P - 1])
//         );
//         assert_eq!(
//             M_neg[(0, 1)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, 0, P - 1, 0])
//         );
//         assert_eq!(
//             M_neg[(1, 0)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![0, P - 1, 0, 0])
//         );
//         assert_eq!(
//             M_neg[(1, 1)],
//             Z_N_CycloNTT_CRT::<D, P>::from(vec![P - 1, 0, 0, 0])
//         );
//     }
// }
