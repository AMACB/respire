// pub trait PIRScheme {
//     type Database;
//     type Index;
//     type Record;
//
//     type PublicParams;
//     type QueryKey;
//     type State;
//     type Query;
//     type Response;
//
//     fn setup() -> (Self::PublicParams, Self::QueryKey);
//
//     fn query(qk: &Self::QueryKey, idx: &Self::Index) -> (Self::State, Self::Query);
//
//     fn answer(
//         pp: &Self::PublicParams,
//         database: &Self::Database,
//         q: &Self::Query,
//     ) -> Self::Response;
//
//     fn extract(qk: &Self::QueryKey, st: &Self::State, r: &Self::Response) -> Self::Record;
// }
//
// pub struct SPIRAL<
//     /* Hybrid Encoding Params */
//     const N: usize,
//     const N_PLUS_ONE: usize,
//     const M: usize,
//     const Q: u64,
//     const D: usize,
//     const G_LEN: usize,
//     const G_BASE: u64,
//     const NOISE_WIDTH_MILLIONTHS: u64,
//     /* Other params */
//     const P: u64,
//     const Q1: u64,
//     const Q2: u64,
//     const ETA1: usize,
//     const ETA2: usize
// > {}
//
// pub struct SPIRALParamsRaw {
//     pub hybrid: HybridEncodingParamsRaw,
//     pub P: u64,
//     pub Q1: u64,
//     pub Q2: u64,
//     pub ETA1: usize,
//     pub ETA2: usize,
// }
//
// impl SPIRALParamsRaw {
//     pub const fn expand(&self) -> SPIRALParams {
//         SPIRALParams {
//             hybrid: self.hybrid.expand(),
//             P: self.P,
//             Q1: self.Q1,
//             Q2: self.Q2,
//             ETA1: self.ETA1,
//             ETA2: self.ETA2,
//         }
//     }
// }
//
// pub struct SPIRALParams {
//     pub hybrid: HybridEncodingParams,
//     pub P: u64,
//     pub Q1: u64,
//     pub Q2: u64,
//     pub ETA1: usize,
//     pub ETA2: usize,
// }
//
// #[macro_export]
// macro_rules! spiral {
//     ($params: expr) => {
//         SPIRAL<
//             {$params.hybrid.gsw.N},
//             {$params.hybrid.gsw.N_PLUS_ONE},
//             {$params.hybrid.gsw.M},
//             {$params.hybrid.gsw.Q},
//             {$params.hybrid.gsw.D},
//             {$params.hybrid.gsw.G_LEN},
//             {$params.hybrid.gsw.G_BASE},
//             {$params.hybrid.gsw.NOISE_WIDTH_MILLIONTHS},
//             {$params.P},
//             {$params.Q1},
//             {$params.Q2},
//             {$params.ETA1},
//             {$params.ETA2},
//         >
//     }
// }
//
// impl<
//     const N: usize,
//     const N_PLUS_ONE: usize,
//     const M: usize,
//     const Q: u64,
//     const D: usize,
//     const G_LEN: usize,
//     const G_BASE: u64,
//     const NOISE_WIDTH_MILLIONTHS: u64,
//     const P: u64,
//     const Q1: u64,
//     const Q2: u64,
//     const ETA1: usize,
//     const ETA2: usize
// > PIRScheme for SPIRAL<N, N_PLUS_ONE, M, Q, D, G_LEN, G_BASE, NOISE_WIDTH_MILLIONTHS, P, Q1, Q2, ETA1, ETA2> {}

use crate::math::gadget::{build_gadget, gadget_inverse};
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::utils::{floor_log, mod_inverse};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub struct SPIRAL {}

pub const N: usize = 2;
pub const Q_A: u64 = 268369921;
pub const Q_B: u64 = 249561089;
pub const Q: u64 = Q_A * Q_B;
pub const Q_A_INV: u64 = mod_inverse(Q_A, Q_B);
pub const Q_B_INV: u64 = mod_inverse(Q_B, Q_A);
pub const D: usize = 2048;
pub const W1: u64 = 66294444;
pub const W2: u64 = 30909463;
pub const G_BASE: u64 = 2;
pub const G_LEN: usize = floor_log(G_BASE, Q) + 1;
pub const M: usize = (N + 1) * G_LEN;

pub const NOISE_WIDTH_MILLIONTHS: u64 = 6_400_000;

pub const P: u64 = 1 << 8;

pub const ETA1: usize = 9;
pub const ETA2: usize = 6;
pub const DB_SIZE: usize = 1 << (ETA1 + ETA2);

pub const ETA1_MASK: usize = (1 << ETA1) - 1;
pub const ETA2_MASK: usize = (1 << ETA2) - 1;

pub type RingP = IntModCyclo<D, P>;
pub type RingQ = IntModCyclo<D, Q>;
pub type RingQFast = IntModCycloCRTEval<D, Q_A, Q_B, Q_A_INV, Q_B_INV, W1, W2>;
pub type MatrixRegevCiphertext = Matrix<{ N + 1 }, N, RingQFast>;
pub type GSWCiphertext = Matrix<{ N + 1 }, M, RingQFast>;

pub type QueryKey = Matrix<N, 1, RingQFast>;
pub type Query = (Vec<MatrixRegevCiphertext>, Vec<GSWCiphertext>);
pub type Response = MatrixRegevCiphertext;

pub type MatrixP = Matrix<N, N, RingP>;
pub type MatrixQ = Matrix<N, N, RingQ>;
pub type MatrixQFast = Matrix<N, N, RingQFast>;
pub type Database = Vec<MatrixP>;
pub type DatabasePreprocessed = Vec<MatrixQFast>;

impl SPIRAL {
    fn encode_regev(qk: &QueryKey, mu: &MatrixQ) -> MatrixRegevCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, N, RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, N, RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<{ N + 1 }, N, RingQFast> = Matrix::stack(
            &a_t,
            &(&(&(qk * &a_t) + &e_mat) + &mu.into_ring(|x| RingQFast::from(x))),
        );
        c_mat
    }

    fn encode_gsw(qk: &QueryKey, mu: &RingP) -> GSWCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M, RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, M, RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<{ N + 1 }, M, RingQFast> = &Matrix::stack(&a_t, &(&(qk * &a_t) + &e_mat))
            + &(&build_gadget::<RingQFast, { N + 1 }, M, G_BASE, G_LEN>()
                * &RingQFast::from(&mu.include_into::<Q>()));
        c_mat
    }

    fn regev_sub_hom(
        lhs: &MatrixRegevCiphertext,
        rhs: &MatrixRegevCiphertext,
    ) -> MatrixRegevCiphertext {
        lhs - rhs
    }
    fn regev_mul_scalar(lhs: &MatrixRegevCiphertext, rhs: &MatrixQFast) -> MatrixRegevCiphertext {
        lhs * rhs
    }
    fn regev_add_eq_mul_scalar(
        lhs: &mut MatrixRegevCiphertext,
        rhs_a: &MatrixRegevCiphertext,
        rhs_b: &MatrixQFast,
    ) {
        lhs.add_eq_mul(rhs_a, rhs_b);
    }

    fn hybrid_mul_hom(regev: &MatrixRegevCiphertext, gsw: &GSWCiphertext) -> MatrixRegevCiphertext {
        gsw * &gadget_inverse::<RingQFast, { N + 1 }, M, N, G_BASE, G_LEN>(regev)
    }

    fn decode_regev(qk: &QueryKey, c: &MatrixRegevCiphertext) -> MatrixQ {
        (&Matrix::append(&-qk, &Matrix::<N, N, _>::identity()) * c).into_ring(|x| RingQ::from(x))
    }

    pub fn setup() -> QueryKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, RingQFast> = Matrix::rand_uniform(&mut rng);
        s
    }

    pub fn query(qk: &QueryKey, idx: usize) -> Query {
        let idx_i = (idx >> ETA2) & ETA1_MASK;
        let idx_j = idx & ETA2_MASK;

        let mut regevs: Vec<MatrixRegevCiphertext> = Vec::with_capacity(1 << ETA1);
        for i in 0..(1 << ETA1) {
            regevs.push(if i == idx_i {
                let ident = MatrixP::identity();
                Self::encode_regev(&qk, &ident.into_ring(|x| x.scale_up_into()))
            } else {
                Self::encode_regev(&qk, &MatrixQ::zero())
            });
        }

        let mut gsws: Vec<GSWCiphertext> = Vec::with_capacity(ETA2);
        for j in 0..ETA2 {
            gsws.push(if (idx_j >> (ETA2 - j - 1)) & 1 != 0 {
                Self::encode_gsw(&qk, &1_u64.into())
            } else {
                Self::encode_gsw(&qk, &0_u64.into())
            });
        }

        (regevs, gsws)
    }

    pub fn answer(d: &DatabasePreprocessed, q: &Query) -> Response {
        let d_at = |i: usize, j: usize| &d[(i << ETA2) + j];
        let mut prev: Vec<MatrixRegevCiphertext> = Vec::with_capacity(1 << ETA2);
        for j in 0..(1 << ETA2) {
            let mut sum = Self::regev_mul_scalar(&q.0[0], d_at(0, j));
            for i in 1..(1 << ETA1) {
                Self::regev_add_eq_mul_scalar(&mut sum, &q.0[i], d_at(i, j));
            }
            prev.push(sum);
        }

        for r in 0..ETA2 {
            let curr_size = 1 << (ETA2 - r - 1);
            let mut curr: Vec<MatrixRegevCiphertext> = Vec::with_capacity(curr_size);
            for j in 0..curr_size {
                let b = &q.1[r];
                let c0 = &prev[j];
                let c1 = &prev[curr_size + j];
                let c1_sub_c0 = Self::regev_sub_hom(c1, c0);
                let mut result = Self::hybrid_mul_hom(&c1_sub_c0, &b);
                result += c0;
                curr.push(result);
            }
            prev = curr;
        }
        prev.remove(0)
    }

    pub fn extract(qk: &QueryKey, r: &Response) -> MatrixP {
        Self::decode_regev(qk, r).into_ring(|x| x.round_down_into())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_spiral_one() {
        test_spiral_n([11111].into_iter());
    }
    #[test]
    fn test_spiral() {
        test_spiral_n([0, 11111, DB_SIZE - 1].into_iter());
    }

    #[ignore]
    #[test]
    fn test_spiral_stress() {
        test_spiral_n(0..DB_SIZE);
    }

    fn test_spiral_n<I: Iterator<Item = usize>>(iter: I) {
        let mut db: Database = Vec::with_capacity(DB_SIZE);
        for i in 0..DB_SIZE as u64 {
            let mut record: MatrixP = Matrix::zero();
            record[(0, 0)] = vec![
                i % 100,
                (i / 100) % 100,
                (i / 10000) % 100,
                (i / 1000000) % 100,
            ]
            .into();
            record[(0, 1)] = vec![i % 256, (i / 256) % 256, 0, 0].into();
            record[(1, 1)] = (i * 37 % 256).into();
            db.push(record);
        }

        let start = Instant::now();
        let mut db_pre: DatabasePreprocessed = Vec::with_capacity(DB_SIZE);
        for i in 0..DB_SIZE {
            db_pre.push(db[i].into_ring(|x| RingQFast::from(&x.include_into::<Q>())));
        }
        let end = Instant::now();
        eprintln!("{:?} to preprocess", end - start);

        let qk = SPIRAL::setup();
        let check = |idx: usize| {
            let cts = SPIRAL::query(&qk, idx);
            let result = SPIRAL::answer(&db_pre, &cts);
            let extracted = SPIRAL::extract(&qk, &result);
            assert_eq!(&extracted, &db[idx])
        };

        for i in iter {
            let start = Instant::now();
            check(i);
            let end = Instant::now();
            eprintln!("{:?} for query {}", end - start, i);
        }
    }
}
