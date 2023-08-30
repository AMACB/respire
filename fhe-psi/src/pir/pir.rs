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

use crate::math::matrix::Matrix;
use crate::math::z_n_cyclo::Z_N_CycloRaw;
use crate::pir::encoding::EncodingScheme;
use crate::pir::gsw_encoding::GSWEncoding;
use crate::pir::matrix_regev_encoding::{
    HybridEncodingParams, HybridEncodingParamsRaw, MatrixRegevEncoding,
};
use crate::{gsw_encoding, matrix_regev_encoding};

struct SPIRAL {}

const N: usize = 2;
const Q: u64 = 268369921;
const D: usize = 4;
const W: u64 = 185593570;
const G_BASE: u64 = 2;
const NOISE_WIDTH_MILLIONTHS: u64 = 6_400_000;

const P: u64 = 1 << 8;
// const Q1: u64 = 1 << 10;
// const Q2: u64 = 1 << 21;

const ETA1: usize = 9;
const ETA2: usize = 6;

const ETA1_MASK: usize = (1 << ETA1) - 1;
const ETA2_MASK: usize = (1 << ETA2) - 1;

const HYBRID_PARAMS: HybridEncodingParams = HybridEncodingParamsRaw {
    N,
    Q,
    D,
    W,
    G_BASE,
    NOISE_WIDTH_MILLIONTHS,
}
.expand();

type Regev = matrix_regev_encoding!(HYBRID_PARAMS.matrix_regev);
type GSW = gsw_encoding!(HYBRID_PARAMS.gsw);

type RegevCT = <Regev as EncodingScheme>::Ciphertext;
type GSWCT = <GSW as EncodingScheme>::Ciphertext;

type QueryKey = <Regev as EncodingScheme>::SecretKey;
type Query = (Vec<RegevCT>, Vec<GSWCT>);
type Response = RegevCT;

type Record = Matrix<N, N, Z_N_CycloRaw<D, P>>;
type Database = Vec<Record>;

impl SPIRAL {
    fn setup() -> QueryKey {
        Regev::keygen()
    }

    fn query(qk: &QueryKey, idx: usize) -> Query {
        let idx_i = (idx >> ETA2) & ETA1_MASK;
        let idx_j = idx & ETA2_MASK;

        let mut regevs: Vec<RegevCT> = vec![];
        regevs.reserve(1 << ETA1);
        for i in 0..(1 << ETA1) {
            regevs.push(if i == idx_i {
                let ident = Matrix::<N, N, Z_N_CycloRaw<D, P>>::identity();
                Regev::encode(&qk, &ident.into_ring(|x| x.scale_up_into()))
            } else {
                Regev::encode(&qk, &Matrix::zero())
            });
        }

        let mut gsws: Vec<GSWCT> = vec![];
        gsws.reserve(ETA2);
        for j in 0..ETA2 {
            gsws.push(if (idx_j >> (ETA2 - j - 1)) & 1 != 0 {
                GSW::encode(&qk, &1_u64.into())
            } else {
                GSW::encode(&qk, &0_u64.into())
            });
        }

        (regevs, gsws)
    }

    fn answer(d: &Database, q: &Query) -> Response {
        let d_at = |i: usize, j: usize| {
            let record_p = &d[(i << ETA2) + j];
            record_p.into_ring(|x| x.include_into())
        };
        let mut prev: Vec<RegevCT> = vec![];
        prev.reserve(1 << ETA2);
        for j in 0..(1 << ETA2) {
            let mut sum = Regev::mul_scalar(&q.0[0], &d_at(0, j));
            for i in 1..(1 << ETA1) {
                sum = Regev::add_hom(&sum, &Regev::mul_scalar(&q.0[i], &d_at(i, j)));
            }
            prev.push(sum);
        }

        for r in 0..ETA2 {
            let mut curr: Vec<RegevCT> = vec![];
            let curr_size = 1 << (ETA2 - r - 1);
            curr.reserve(curr_size);
            for j in 0..curr_size {
                let C0 = &prev[j];
                let C1 = &prev[curr_size + j];
                let C1_sub_C0 = Regev::sub_hom(C1, C0);
                curr.push(Regev::add_hom(
                    &Regev::mul_hom_gsw::<
                        { HYBRID_PARAMS.gsw.M },
                        { HYBRID_PARAMS.gsw.G_BASE },
                        { HYBRID_PARAMS.gsw.G_LEN },
                    >(&q.1[r], &C1_sub_C0),
                    &C0,
                ));
            }
            prev = curr;
        }
        prev.remove(0)
    }

    fn extract(qk: &QueryKey, r: &Response) -> Record {
        Regev::decode(&qk, &r).into_ring(|x| x.round_down_into())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_spiral() {
        let mut db: Database = vec![];
        db.reserve(1 << (ETA1 + ETA2));
        for i in 0_u64..(1 << (ETA1 + ETA2)) {
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

        let qk = SPIRAL::setup();
        let check = |idx: usize| {
            let cts = SPIRAL::query(&qk, idx);
            let result = SPIRAL::answer(&db, &cts);
            let extracted = SPIRAL::extract(&qk, &result);
            assert_eq!(&extracted, &db[idx])
        };
        check(0);
        check(11111);
        check(32767);
    }
}
