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

use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::matrix::Matrix;
use crate::pir::encoding::EncodingScheme;
use crate::pir::gsw_encoding::GSWEncoding;
use crate::pir::matrix_regev_encoding::{
    HybridEncodingParams, HybridEncodingParamsRaw, MatrixRegevEncoding,
};
use crate::{gsw_encoding, matrix_regev_encoding};

pub struct SPIRAL {}

pub const N: usize = 2;
pub const Q: u64 = 268369921;
pub const D: usize = 2048;
pub const W: u64 = 63703579;
pub const G_BASE: u64 = 2;
pub const NOISE_WIDTH_MILLIONTHS: u64 = 1;

pub const P: u64 = 1 << 8;
// const Q1: u64 = 1 << 10;
// const Q2: u64 = 1 << 21;

pub const ETA1: usize = 9;
pub const ETA2: usize = 6;
pub const DB_SIZE: usize = 1 << (ETA1 + ETA2);

pub const ETA1_MASK: usize = (1 << ETA1) - 1;
pub const ETA2_MASK: usize = (1 << ETA2) - 1;

pub const HYBRID_PARAMS: HybridEncodingParams = HybridEncodingParamsRaw {
    N,
    Q,
    D,
    W,
    G_BASE,
    NOISE_WIDTH_MILLIONTHS,
}
.expand();

pub type Regev = matrix_regev_encoding!(HYBRID_PARAMS.matrix_regev);
pub type GSW = gsw_encoding!(HYBRID_PARAMS.gsw);

pub type RegevCT = <Regev as EncodingScheme>::Ciphertext;
pub type GSWCT = <GSW as EncodingScheme>::Ciphertext;

pub type QueryKey = <Regev as EncodingScheme>::SecretKey;
pub type Query = (Vec<RegevCT>, Vec<GSWCT>);
pub type Response = RegevCT;

pub type Record = Matrix<N, N, IntModCyclo<D, P>>;
pub type RecordPreprocessed = Matrix<N, N, IntModCycloEval<D, Q, W>>;
pub type Database = Vec<Record>;
pub type DatabasePreprocessed = Vec<RecordPreprocessed>;

impl SPIRAL {
    pub fn setup() -> QueryKey {
        Regev::keygen()
    }

    pub fn query(qk: &QueryKey, idx: usize) -> Query {
        let idx_i = (idx >> ETA2) & ETA1_MASK;
        let idx_j = idx & ETA2_MASK;

        let mut regevs: Vec<RegevCT> = Vec::with_capacity(1 << ETA1);
        for i in 0..(1 << ETA1) {
            regevs.push(if i == idx_i {
                let ident = Matrix::<N, N, IntModCyclo<D, P>>::identity();
                Regev::encode(&qk, &ident.into_ring(|x| x.scale_up_into()))
            } else {
                Regev::encode(&qk, &Matrix::zero())
            });
        }

        let mut gsws: Vec<GSWCT> = Vec::with_capacity(ETA2);
        for j in 0..ETA2 {
            gsws.push(if (idx_j >> (ETA2 - j - 1)) & 1 != 0 {
                GSW::encode(&qk, &1_u64.into())
            } else {
                GSW::encode(&qk, &0_u64.into())
            });
        }

        (regevs, gsws)
    }

    pub fn answer(d: &DatabasePreprocessed, q: &Query) -> Response {
        let d_at = |i: usize, j: usize| &d[(i << ETA2) + j];
        let mut prev: Vec<RegevCT> = Vec::with_capacity(1 << ETA2);
        for j in 0..(1 << ETA2) {
            // Regev scalar mul
            let mut sum = &q.0[0] * d_at(0, j);
            for i in 1..(1 << ETA1) {
                // Regev hom add, Regev scalar mul
                sum.add_eq_mul(&q.0[i], d_at(i, j));
            }
            prev.push(sum);
        }

        for r in 0..ETA2 {
            let curr_size = 1 << (ETA2 - r - 1);
            let mut curr: Vec<RegevCT> = Vec::with_capacity(curr_size);
            for j in 0..curr_size {
                let b = &q.1[r];
                let c0 = &prev[j];
                let c1 = &prev[curr_size + j];
                let c1_sub_c0 = Regev::sub_hom(c1, c0);
                let mut result = Regev::mul_hom_gsw::<
                    { HYBRID_PARAMS.gsw.M },
                    { HYBRID_PARAMS.gsw.G_BASE },
                    { HYBRID_PARAMS.gsw.G_LEN },
                >(b, &c1_sub_c0);
                result += c0;
                curr.push(result);
            }
            prev = curr;
        }
        prev.remove(0)
    }

    pub fn extract(qk: &QueryKey, r: &Response) -> Record {
        Regev::decode(&qk, &r).into_ring(|x| x.round_down_into())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

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
            let mut record: Record = Matrix::zero();
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
            db_pre.push(db[i].into_ring(|x| IntModCycloEval::from(x.include_into())));
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
