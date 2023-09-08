//! The cyclotomic ring `Z_n[x]/x^d + 1)`, where `n = n_1 * n_2` and `d` is assumed to be a power of `2`.

use crate::math::discrete_gaussian::DiscreteGaussian;
use crate::math::gadget::RingElementDecomposable;
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
pub struct IntModCycloCRT<
    const D: usize,
    const N1: u64,
    const N2: u64,
    const N1_INV: u64,
    const N2_INV: u64,
> {
    p1: IntModCyclo<D, N1>,
    p2: IntModCyclo<D, N2>,
}

/// Conversions

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> From<u64>
    for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn from(a: u64) -> Self {
        Self {
            p1: a.into(),
            p2: a.into(),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    From<(IntModCyclo<D, N1>, IntModCyclo<D, N2>)> for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn from(a: (IntModCyclo<D, N1>, IntModCyclo<D, N2>)) -> Self {
        IntModCycloCRT { p1: a.0, p2: a.1 }
    }
}

impl<
        const D: usize,
        const N1: u64,
        const N2: u64,
        const N1_INV: u64,
        const N2_INV: u64,
        const W1: u64,
        const W2: u64,
    > From<&IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>>
    for IntModCycloCRTEval<D, N1, N2, N1_INV, N2_INV, W1, W2>
{
    fn from(a: &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>) -> Self {
        let p1_ntt: IntModCycloEval<D, N1, W1> = (&a.p1).into();
        let p2_ntt: IntModCycloEval<D, N2, W2> = (&a.p2).into();
        (p1_ntt, p2_ntt).into()
    }
}

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> From<[Z_N<N>; D]> for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV> {
//     fn from(coeff: [Z_N<N>; D]) -> Self {
//         Self { coeff }
//     }
// }

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> From<PolynomialZ_N<N>> for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV> {
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

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    From<Vec<u64>> for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn from(coeff: Vec<u64>) -> Self {
        IntModCycloCRT {
            p1: IntModCyclo::from(IntModPoly::from(coeff.clone())),
            p2: IntModCyclo::from(IntModPoly::from(coeff)),
        }
    }
}

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> From<Vec<Z_N<N>>> for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV> {
//     fn from(coeff: Vec<Z_N<N>>) -> Self {
//         Z_N_CycloRaw_CRT::from(PolynomialZ_N::from(coeff))
//     }
// }

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> TryFrom<&Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV>> for Z_N<N> {
//     type Error = ();

//     /// Inverse of `From<u64>`. Errors if element is not a constant.
//     fn try_from(a: &Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV>) -> Result<Self, Self::Error> {
//         for i in 1..D {
//             if a.coeff[i] != Z_N::zero() {
//                 return Err(());
//             }
//         }
//         Ok(a.coeff[0])
//     }
// }

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64, const W: u64> From<Z_N_CycloNTT<D, N, W>>
//     for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV>
// {
//     fn from(a: Z_N_CycloNTT<D, N, W>) -> Self {
//         (&a).into()
//     }
// }

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64, const W: u64> From<&Z_N_CycloNTT<D, N, W>>
//     for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV>
// {
//     fn from(a_eval: &Z_N_CycloNTT<D, N, W>) -> Self {
//         // TODO: this should be in the type, probably
//         let mut log_d = 1;
//         while (1 << log_d) < D {
//             log_d += 1;
//         }
//         assert_eq!(1 << log_d, D);

//         let root: Z_N<N> = W.into();

//         let mut coeff: [Z_N<N>; D] = [0_u64.into(); D];
//         for (i, x) in a_eval.points_iter().enumerate() {
//             coeff[i] = x.clone();
//         }

//         bit_reverse_order(&mut coeff, log_d);
//         ntt(&mut coeff, (root * root).inverse(), log_d);

//         let mut inv_root_pow: Z_N<N> = 1u64.into();
//         let inv_root = root.inverse();
//         let inv_D = Z_N::<N>::from(D as u64).inverse();
//         for i in 0..D {
//             // divide by degree
//             coeff[i] *= inv_D;
//             // negacyclic post-processing
//             coeff[i] *= inv_root_pow;
//             inv_root_pow *= inv_root;
//         }

//         return Self { coeff };
//     }
// }

/// [`RingElementRef`] implementation

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    RingElementRef<IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>>
    for &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Add
    for &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>;
    fn add(self, rhs: Self) -> Self::Output {
        IntModCycloCRT {
            p1: &self.p1 + &rhs.p1,
            p2: &self.p2 + &rhs.p2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Sub
    for &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>;
    fn sub(self, rhs: Self) -> Self::Output {
        IntModCycloCRT {
            p1: &self.p1 - &rhs.p1,
            p2: &self.p2 - &rhs.p2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Mul
    for &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>;
    fn mul(self, rhs: Self) -> Self::Output {
        IntModCycloCRT {
            p1: &self.p1 * &rhs.p1,
            p2: &self.p2 * &rhs.p2,
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Neg
    for &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    type Output = IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>;
    fn neg(self) -> Self::Output {
        IntModCycloCRT {
            p1: -&self.p1,
            p2: -&self.p2,
        }
    }
}

/// [`RingElement`] implementation

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> RingElement
    for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn zero() -> IntModCycloCRT<D, N1, N2, N1_INV, N2_INV> {
        IntModCycloCRT {
            p1: IntModCyclo::zero(),
            p2: IntModCyclo::zero(),
        }
    }
    fn one() -> IntModCycloCRT<D, N1, N2, N1_INV, N2_INV> {
        IntModCycloCRT {
            p1: IntModCyclo::one(),
            p2: IntModCyclo::one(),
        }
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    AddAssign<&'a Self> for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn add_assign(&mut self, rhs: &'a Self) {
        self.p1 += &rhs.p1;
        self.p2 += &rhs.p2;
    }
}

impl<'a, const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    SubAssign<&'a Self> for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.p1 -= &rhs.p1;
        self.p2 -= &rhs.p2;
    }
}

// impl<'a, const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> MulAssign<Z_N<N>> for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV> {
//     fn mul_assign(&mut self, rhs: Z_N<N>) {
//         self.p1 *= &rhs.p1;
//         self.p2 *= &rhs.p2;
//     }
// }

impl<'a, const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    MulAssign<&'a Self> for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn mul_assign(&mut self, _: &'a Self) {
        todo!()
    }
}

impl<
        const D: usize,
        const N1: u64,
        const N2: u64,
        const N1_INV: u64,
        const N2_INV: u64,
        const BASE: u64,
        const LEN: usize,
    > RingElementDecomposable<BASE, LEN> for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let mut a: [u64; D] = [0; D];
        for l in 0..D {
            let coeff: IntModCRT<N1, N2, N1_INV, N2_INV> = (self.p1[l], self.p2[l]).into();
            a[l] = coeff.into();
        }
        for k in 0..LEN {
            let mut a1_rem: [IntMod<N1>; D] = [IntMod::zero(); D];
            let mut a2_rem: [IntMod<N2>; D] = [IntMod::zero(); D];
            for l in 0..D {
                let x = a[l] % BASE;
                a[l] /= BASE;

                a1_rem[l] = x.into();
                a2_rem[l] = x.into();
            }
            let p1 = a1_rem.into();
            let p2 = a2_rem.into();
            mat[(i + k, j)] = (p1, p2).into();
        }
    }
}

/// Random sampling

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    RandUniformSampled for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        IntModCycloCRT {
            p1: IntModCyclo::rand_uniform(rng),
            p2: IntModCyclo::rand_uniform(rng),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    RandZeroOneSampled for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
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
            p1: v.clone().into(),
            p2: v.into(),
        }
    }
}

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    RandDiscreteGaussianSampled for IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        let mut v = vec![0i64; D];
        v.iter_mut()
            .map(|x| *x = DiscreteGaussian::sample::<_, NOISE_WIDTH_MILLIONTHS>(rng))
            .count();
        IntModCycloCRT {
            p1: v.clone().into(),
            p2: v.into(),
        }
    }
}

// Other polynomial-specific operations.

// TODO: maybe don't need this bc of index
// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV> {
//     pub fn coeff_iter(&self) -> Iter<'_, Z_N<{ N }>> {
//         self.coeff.iter()
//     }
// }

impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>
{
    pub fn norm(&self) -> u64 {
        let mut worst: u64 = 0;
        for i in 0..D {
            let val: IntModCRT<N1, N2, N1_INV, N2_INV> = (self.p1[i], self.p2[i]).into();
            worst = max(worst, val.norm());
        }
        worst
    }
}

// TODO: this should be a TryFrom
impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64>
    From<&IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>> for IntModCRT<N1, N2, N1_INV, N2_INV>
{
    fn from(a: &IntModCycloCRT<D, N1, N2, N1_INV, N2_INV>) -> Self {
        (a.p1[0], a.p2[0]).into()
    }
}

// impl<const D: usize, const N1: u64, const N2: u64, const N1_INV: u64, const N2_INV: u64> Index<usize> for Z_N_CycloRaw_CRT<D, N1, N2, N1_INV, N2_INV> {
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
