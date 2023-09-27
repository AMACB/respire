//! The cyclotomic ring `Z_n[x]/x^d + 1)`, represented as its DFT. `d` is assumed to be a power of `2`.

use crate::math::gadget::{IntModDecomposition, RingElementDecomposable};
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_poly::IntModPoly;
use crate::math::matrix::Matrix;
use crate::math::ntt::*;
use crate::math::number_theory::mod_pow;
use crate::math::rand_sampled::*;
use crate::math::ring_elem::*;
use crate::math::utils::reverse_bits;
use rand::Rng;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO
// We need a way to bind a root of the right order to the type.
// Options (there's probably more): compute on the fly w.r.t the type, or add it as a type parameter.

/// The DFT (pointwise evaluations) representation of an element of a cyclotomic ring.
///
/// Internally, this is an array of evaluations, where the `i`th index corresponds to `f(w^{2*bit_reverse(i)+1})`.
/// `w` here is the `2*D`th root of unity.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct IntModCycloEval<const D: usize, const N: u64, const W: u64> {
    pub(in crate::math) points: [IntMod<N>; D],
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

// TODO: this does a clone, which the user may not be aware about...
impl<const D: usize, const N: u64, const W: u64> From<&IntModCycloEval<D, N, W>>
    for IntModCyclo<D, N>
{
    fn from(a_eval: &IntModCycloEval<D, N, W>) -> Self {
        (a_eval.clone()).into()
    }
}

impl<const D: usize, const N: u64, const W: u64> From<IntModCyclo<D, N>>
    for IntModCycloEval<D, N, W>
{
    fn from(a: IntModCyclo<D, N>) -> Self {
        let points = ntt_neg_forward::<D, N, W>(a.coeff);
        IntModCycloEval::from(points)
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
        let self_coeff = IntModCyclo::from(self);
        let mut decomps = Vec::<IntModDecomposition<BASE, LEN>>::with_capacity(D);
        for coeff_idx in 0..D {
            decomps.push(IntModDecomposition::<BASE, LEN>::new(
                u64::from(self_coeff.coeff[coeff_idx]),
                NN,
            ))
        }
        for k in 0..LEN {
            let mut result_coeff = IntModCyclo::zero();
            for coeff_idx in 0..D {
                result_coeff.coeff[coeff_idx] = IntMod::from(decomps[coeff_idx].next().unwrap());
            }
            mat[(i + k, j)] = IntModCycloEval::from(result_coeff);
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

/// Other operations.

impl<const D: usize, const N: u64, const W: u64> IntModCycloEval<D, N, W> {
    /// Compute the automorphism x --> x^k. This only makes sense for odd `k`.
    pub fn auto(&self, k: usize) -> Self {
        let mut result = Self::zero();
        let k_half = (k - 1) / 2;
        for i in 0..D {
            let to = i;
            let from = (2 * k_half * i + k_half + i) % D;
            result.points[reverse_bits::<D>(to)] = self.points[reverse_bits::<D>(from)];
        }
        result
    }

    /// Multiply by x^k
    pub fn mul_x_pow(&self, k: usize) -> Self {
        let mut result = Self::zero();
        let mut w_curr = IntMod::from(mod_pow(W, k as u64, N));
        let w_k_sq = w_curr * w_curr;
        for i in 0..D {
            let i_rev = reverse_bits::<D>(i);
            result.points[i_rev] = self.points[i_rev] * w_curr;
            w_curr *= w_k_sq;
        }
        result
    }
}

unsafe impl<const D: usize, const N: u64, const W: u64, const M: u64, const WW: u64>
    RingCompatible<IntModCycloEval<D, M, WW>> for IntModCycloEval<D, N, W>
{
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::matrix::Matrix;
    use crate::math::number_theory::find_sqrt_primitive_root;

    const D: usize = 4; // Z_q[X] / (X^4 + 1)
    const P: u64 = 268369921u64;
    const W: u64 = find_sqrt_primitive_root(D, P); // 8th root of unity

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
    fn test_mul_x_pow() {
        let p = IntModCyclo::<D, P>::from(vec![1_u64, 2, 3, 4]);
        let q = IntModCyclo::<D, P>::from(vec![-3_i64, -4, 1, 2]);
        let p_eval = IntModCycloEval::<D, P, W>::from(p.clone());
        assert_eq!(IntModCyclo::<D, P>::from(p_eval.mul_x_pow(2)), q);
        assert_eq!(IntModCyclo::<D, P>::from(p_eval.mul_x_pow(6)), -&q);
        assert_eq!(IntModCyclo::<D, P>::from(p_eval.mul_x_pow(10)), q);
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

    #[test]
    fn test_auto() {
        const D2: usize = 16;
        const W2: u64 = find_sqrt_primitive_root(D2, P);
        type RRaw = IntModCyclo<D2, P>;
        type REval = IntModCycloEval<D2, P, W2>;

        let x_pow = |a: usize, neg: bool| -> IntModCyclo<D2, P> {
            let mut x = [0_u64; D2];
            x[a] = if neg { P - 1_u64 } else { 1_u64 };
            x.to_vec().into()
        };

        for exp in [1, 3, 5, 7, 9, 11, 13, 15] {
            for a in 0..D2 {
                let x_a = x_pow(a, false);
                let expected_a = exp * a % (2 * D2);
                let expected = x_pow(expected_a % D2, expected_a >= D2);
                assert_eq!(RRaw::from(REval::from(x_a).auto(exp)), expected);
            }
        }
    }
}
