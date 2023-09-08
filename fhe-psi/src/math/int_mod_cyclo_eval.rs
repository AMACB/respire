//! The cyclotomic ring `Z_n[x]/x^d + 1)`, represented as its DFT. `d` is assumed to be a power of `2`.

use crate::math::gadget::RingElementDecomposable;
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_poly::IntModPoly;
use crate::math::matrix::Matrix;
use crate::math::ntt::*;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::slice::Iter;

// TODO
// We need a way to bind a root of the right order to the type.
// Options (there's probably more): compute on the fly w.r.t the type, or add it as a type parameter.

/// The DFT (pointwise evaluations) representation of an element of a cyclotomic ring.
///
/// Internally, this is an array of evaluations, where the `i`th index corresponds to `f(w^{2*i+1})`. `w` here is the `2*D`th root of unity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntModCycloEval<const D: usize, const N: u64, const W: u64> {
    points: [IntMod<N>; D],
}

/// Conversions

impl<const D: usize, const N: u64, const W: u64> From<u64> for IntModCycloEval<D, N, W> {
    fn from(a: u64) -> Self {
        let points = [a.into(); D];
        Self { points }
    }
}

// TODO: This conversion is unintuitive, since it would be taking pointwise coordinates instead of degrees, which breaks the convention from polynomial / Z_N_CycloRaw.
// It is currently kept intact since other functions require this.

impl<const D: usize, const N: u64, const W: u64> From<[IntMod<N>; D]> for IntModCycloEval<D, N, W> {
    fn from(points: [IntMod<N>; D]) -> Self {
        Self { points }
    }
}

impl<const D: usize, const N: u64, const W: u64> From<IntModCyclo<D, N>>
    for IntModCycloEval<D, N, W>
{
    fn from(a: IntModCyclo<D, N>) -> Self {
        (&a).into()
    }
}

impl<const D: usize, const N: u64, const W: u64> From<&IntModCyclo<D, N>>
    for IntModCycloEval<D, N, W>
{
    fn from(a: &IntModCyclo<D, N>) -> Self {
        // TODO: this should be in the type, probably
        let mut log_d = 1;
        while (1 << log_d) < D {
            log_d += 1;
        }
        assert_eq!(1 << log_d, D);

        let root: IntMod<N> = W.into();

        let mut root_power: IntMod<N> = 1u64.into();
        let mut points: [IntMod<N>; D] = [0_u64.into(); D];
        for (i, x) in a.coeff_iter().enumerate() {
            points[i] = x.clone();

            // negacyclic preprocessing
            points[i] *= root_power;
            root_power *= root;
        }

        bit_reverse_order(&mut points, log_d);
        ntt(&mut points, root * root, log_d);

        return Self { points };
    }
}

impl<const D: usize, const N: u64, const W: u64> From<IntModPoly<N>> for IntModCycloEval<D, N, W> {
    fn from(polynomial: IntModPoly<N>) -> Self {
        IntModCycloEval::from(IntModCyclo::from(polynomial))
    }
}

// See above.
// impl<const D: usize, const N: u64, const W: u64> From<Vec<u64>> for Z_N_CycloNTT<D, N, W> {
//     fn from(coeff: Vec<u64>) -> Self {
//         Z_N_CycloNTT::from(Z_N_CycloRaw::from(PolynomialZ_N::from(coeff)))
//     }
// }

// impl<const D: usize, const N: u64, const W: u64> From<Vec<Z_N<N>>> for Z_N_CycloNTT<D, N, W> {
//     fn from(coeff: Vec<Z_N<N>>) -> Self {
//         Z_N_CycloNTT::from(Z_N_CycloRaw::from(PolynomialZ_N::from(coeff)))
//     }
// }

impl<const D: usize, const N: u64, const W: u64> TryFrom<&IntModCycloEval<D, N, W>> for IntMod<N> {
    type Error = ();

    /// Inverse of `From<u64>`. Errors if element is not a constant.
    fn try_from(a: &IntModCycloEval<D, N, W>) -> Result<Self, Self::Error> {
        for i in 1..D {
            if a.points[i] != a.points[0] {
                return Err(());
            }
        }
        Ok(a.points[0])
    }
}

/// [`RingElementRef`] implementation

impl<const D: usize, const N: u64, const W: u64> RingElementRef<IntModCycloEval<D, N, W>>
    for &IntModCycloEval<D, N, W>
{
}

impl<const D: usize, const N: u64, const W: u64> Add for &IntModCycloEval<D, N, W> {
    type Output = IntModCycloEval<D, N, W>;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result_points: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_points[i] = self.points[i] + rhs.points[i];
        }
        result_points.into()
    }
}

impl<const D: usize, const N: u64, const W: u64> Sub for &IntModCycloEval<D, N, W> {
    type Output = IntModCycloEval<D, N, W>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result_points: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_points[i] = self.points[i] - rhs.points[i];
        }
        result_points.into()
    }
}

impl<const D: usize, const N: u64, const W: u64> Mul for &IntModCycloEval<D, N, W> {
    type Output = IntModCycloEval<D, N, W>;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut result_points: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_points[i] = self.points[i] * rhs.points[i];
        }
        result_points.into()
    }
}

impl<const D: usize, const N: u64, const W: u64> Neg for &IntModCycloEval<D, N, W> {
    type Output = IntModCycloEval<D, N, W>;
    fn neg(self) -> Self::Output {
        let mut result_points: [IntMod<N>; D] = [0_u64.into(); D];
        for i in 0..D {
            result_points[i] = -self.points[i];
        }
        result_points.into()
    }
}

/// [`RingElement`] implementation

impl<const D: usize, const N: u64, const W: u64> RingElement for IntModCycloEval<D, N, W> {
    fn zero() -> IntModCycloEval<D, N, W> {
        [0_u64.into(); D].into()
    }
    fn one() -> IntModCycloEval<D, N, W> {
        [1_u64.into(); D].into()
    }
}

impl<'a, const D: usize, const N: u64, const W: u64> AddAssign<&'a Self>
    for IntModCycloEval<D, N, W>
{
    fn add_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.points[i] += rhs.points[i];
        }
    }
}

impl<'a, const D: usize, const N: u64, const W: u64> SubAssign<&'a Self>
    for IntModCycloEval<D, N, W>
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.points[i] -= rhs.points[i];
        }
    }
}

impl<'a, const D: usize, const N: u64, const W: u64> MulAssign<&'a Self>
    for IntModCycloEval<D, N, W>
{
    fn mul_assign(&mut self, rhs: &'a Self) {
        for i in 0..D {
            self.points[i] *= rhs.points[i];
        }
    }
}

impl<const D: usize, const NN: u64, const W: u64, const BASE: u64, const LEN: usize>
    RingElementDecomposable<BASE, LEN> for IntModCycloEval<D, NN, W>
{
    fn decompose_into_mat<const N: usize, const M: usize>(
        &self,
        mat: &mut Matrix<N, M, Self>,
        i: usize,
        j: usize,
    ) {
        let self_raw = IntModCyclo::from(self);
        let mut a: [u64; D] = [0; D];
        for (l, coeff) in self_raw.coeff_iter().enumerate() {
            a[l] = (*coeff).into();
        }
        for k in 0..LEN {
            let mut a_rem: [IntMod<NN>; D] = [0_u64.into(); D];
            for l in 0..D {
                a_rem[l] = (a[l] % BASE).into();
                a[l] /= BASE;
            }
            mat[(i + k, j)] = IntModCycloEval::from(IntModCyclo::from(a_rem));
        }
    }
}

/// Random sampling

impl<const D: usize, const N: u64, const W: u64> RandUniformSampled for IntModCycloEval<D, N, W> {
    // Random coordinates is the same thing.
    fn rand_uniform<T: Rng>(rng: &mut T) -> Self {
        let mut result = Self::zero();
        for i in 0..D {
            result.points[i] = IntMod::<N>::rand_uniform(rng);
        }
        result
    }
}

// These functions cannot be natively supported by Z_N_CycloNTT, but can be done by calling the associated Z_N_CycloRaw versions.
impl<const D: usize, const N: u64, const W: u64> RandZeroOneSampled for IntModCycloEval<D, N, W> {
    fn rand_zero_one<T: Rng>(rng: &mut T) -> Self {
        IntModCyclo::rand_zero_one(rng).into()
    }
}

impl<const D: usize, const N: u64, const W: u64> RandDiscreteGaussianSampled
    for IntModCycloEval<D, N, W>
{
    fn rand_discrete_gaussian<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> Self {
        IntModCyclo::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(rng).into()
    }
}

/// Other polynomial-specific operations.

impl<const D: usize, const N: u64, const W: u64> IntModCycloEval<D, N, W> {
    pub fn points_iter(&self) -> Iter<'_, IntMod<{ N }>> {
        self.points.iter()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::matrix::Matrix;

    const D: usize = 4; // Z_q[X] / (X^4 + 1)
    const P: u64 = 268369921u64;
    const W: u64 = 185593570u64; // 8th root of unity

    // TODO: add more tests.
    #[test]
    fn test_from_into() {
        let mut p =
            IntModCycloEval::<D, P, W>::from([1u64.into(), 1u64.into(), 1u64.into(), 1u64.into()]);
        let mut q = IntModCycloEval::<D, P, W>::from(IntModCyclo::from(vec![1u64]));
        assert_eq!(p, q);
        assert_eq!(IntModCyclo::<D, P>::from(p), IntModCyclo::<D, P>::from(q));

        let root: IntMod<P> = W.into();
        p = IntModCycloEval::<D, P, W>::from([
            root.pow(1u64),
            root.pow(3u64),
            root.pow(5u64),
            root.pow(7u64),
        ]);
        q = IntModCycloEval::<D, P, W>::from(IntModCyclo::from(vec![0u64, 1u64]));
        assert_eq!(p, q);
        assert_eq!(IntModCyclo::<D, P>::from(p), IntModCyclo::<D, P>::from(q));
    }

    #[test]
    fn test_ops() {
        let p = IntModCyclo::<D, P>::from(vec![1_u64, 2, 3, 4]);
        let q = IntModCyclo::<D, P>::from(vec![5_u64, 6, 7, 8]);

        let p_ntt: IntModCycloEval<D, P, W> = p.clone().into();
        let q_ntt: IntModCycloEval<D, P, W> = q.clone().into();

        assert_eq!(IntModCycloEval::<D, P, W>::from(&p + &q), &p_ntt + &q_ntt);
        assert_eq!(IntModCycloEval::<D, P, W>::from(&p - &q), &p_ntt - &q_ntt);
        assert_eq!(IntModCycloEval::<D, P, W>::from(&p * &q), &p_ntt * &q_ntt);
    }

    #[test]
    fn test_matrix() {
        let mut m: Matrix<2, 2, IntModCycloEval<D, P, W>> = Matrix::zero();
        m[(0, 0)] = IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, 1]).into();
        m[(0, 1)] = IntModCyclo::<D, P>::from(vec![0_i64, 0, 1, 0]).into();
        m[(1, 0)] = IntModCyclo::<D, P>::from(vec![0_i64, 1, 0, 0]).into();
        m[(1, 1)] = IntModCyclo::<D, P>::from(vec![1_i64, 0, 0, 0]).into();
        // m =
        // [ x^3 x^2 ]
        // [ x   1   ]
        let m_square = &m * &m;
        assert_eq!(
            IntModCyclo::<D, P>::from(m_square[(0, 0)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, 0, -1, 1])
        ); // x^3 + x^6
        assert_eq!(
            IntModCyclo::<D, P>::from(m_square[(0, 1)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, -1, 1, 0])
        ); // x^2 + x^5
        assert_eq!(
            IntModCyclo::<D, P>::from(m_square[(1, 0)].clone()),
            IntModCyclo::<D, P>::from(vec![-1_i64, 1, 0, 0])
        ); // x + x^4
        assert_eq!(
            IntModCyclo::<D, P>::from(m_square[(1, 1)].clone()),
            IntModCyclo::<D, P>::from(vec![1_i64, 0, 0, 1])
        ); // 1 + x^3

        let m_double = &m + &m;
        assert_eq!(
            IntModCyclo::<D, P>::from(m_double[(0, 0)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, 2])
        );
        assert_eq!(
            IntModCyclo::<D, P>::from(m_double[(0, 1)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, 0, 2, 0])
        );
        assert_eq!(
            IntModCyclo::<D, P>::from(m_double[(1, 0)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, 2, 0, 0])
        );
        assert_eq!(
            IntModCyclo::<D, P>::from(m_double[(1, 1)].clone()),
            IntModCyclo::<D, P>::from(vec![2_i64, 0, 0, 0])
        );

        let m_neg = -&m;
        assert_eq!(
            IntModCyclo::<D, P>::from(m_neg[(0, 0)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, 0, 0, -1])
        );
        assert_eq!(
            IntModCyclo::<D, P>::from(m_neg[(0, 1)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, 0, -1, 0])
        );
        assert_eq!(
            IntModCyclo::<D, P>::from(m_neg[(1, 0)].clone()),
            IntModCyclo::<D, P>::from(vec![0_i64, -1, 0, 0])
        );
        assert_eq!(
            IntModCyclo::<D, P>::from(m_neg[(1, 1)].clone()),
            IntModCyclo::<D, P>::from(vec![-1_i64, 0, 0, 0])
        );
    }
}
