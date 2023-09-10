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

use crate::math::gadget::{build_gadget, gadget_inverse};
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::utils::{floor_log, mod_inverse};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

pub struct SPIRALImpl<
    const N: usize,
    const N_PLUS_ONE: usize,
    const Q: u64,
    const Q_A: u64,
    const Q_B: u64,
    const Q_A_INV: u64,
    const Q_B_INV: u64,
    const D: usize,
    const W1: u64,
    const W2: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    const M: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
    const P: u64,
    const ETA1: usize,
    const ETA2: usize,
> {}

#[allow(non_snake_case)]
pub struct SPIRALParamsRaw {
    pub N: usize,
    pub Q_A: u64,
    pub Q_B: u64,
    pub D: usize,
    pub W1: u64,
    pub W2: u64,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub ETA1: usize,
    pub ETA2: usize,
}

impl SPIRALParamsRaw {
    pub const fn expand(&self) -> SPIRALParams {
        let q = self.Q_A * self.Q_B;
        let t = floor_log(self.G_BASE, q) + 1;
        SPIRALParams {
            N: self.N,
            N_PLUS_ONE: self.N + 1,
            Q: q,
            Q_A: self.Q_A,
            Q_B: self.Q_B,
            Q_A_INV: mod_inverse(self.Q_A, self.Q_B),
            Q_B_INV: mod_inverse(self.Q_B, self.Q_A),
            D: self.D,
            W1: self.W1,
            W2: self.W2,
            G_BASE: self.G_BASE,
            G_LEN: t,
            M: (self.N + 1) * t,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
            P: self.P,
            ETA1: self.ETA1,
            ETA2: self.ETA2,
        }
    }
}

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct SPIRALParams {
    pub N: usize,
    pub N_PLUS_ONE: usize,
    pub Q: u64,
    pub Q_A: u64,
    pub Q_B: u64,
    pub Q_A_INV: u64,
    pub Q_B_INV: u64,
    pub D: usize,
    pub W1: u64,
    pub W2: u64,
    pub G_BASE: u64,
    pub G_LEN: usize,
    pub M: usize,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub ETA1: usize,
    pub ETA2: usize,
}

#[macro_export]
macro_rules! spiral {
    ($params: expr) => {
        SPIRALImpl<
            {$params.N},
            {$params.N_PLUS_ONE},
            {$params.Q},
            {$params.Q_A},
            {$params.Q_B},
            {$params.Q_A_INV},
            {$params.Q_B_INV},
            {$params.D},
            {$params.W1},
            {$params.W2},
            {$params.G_BASE},
            {$params.G_LEN},
            {$params.M},
            {$params.NOISE_WIDTH_MILLIONTHS},
            {$params.P},
            {$params.ETA1},
            {$params.ETA2},
        >
    }
}

pub trait SPIRAL {
    // Type aliases
    type RingP;
    type RingQ;
    type RingQFast;
    type MatrixP;
    type MatrixQ;
    type MatrixQFast;
    type MatrixRegevCiphertext;
    type GSWCiphertext;

    // Associated types
    type QueryKey;
    type Query;
    type Response;
    type Record;
    type RecordPreprocessed;

    // Constants
    const DB_SIZE: usize;
    const ETA1: usize;
    const ETA2: usize;

    fn preprocess(record: &Self::Record) -> Self::RecordPreprocessed;
    fn setup() -> Self::QueryKey;
    fn query(qk: &Self::QueryKey, idx: usize) -> Self::Query;
    fn answer(db: &Vec<Self::RecordPreprocessed>, query: &Self::Query) -> Self::Response;
    fn extract(qk: &Self::QueryKey, r: &Self::Response) -> Self::Record;
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const Q: u64,
        const Q_A: u64,
        const Q_B: u64,
        const Q_A_INV: u64,
        const Q_B_INV: u64,
        const D: usize,
        const W1: u64,
        const W2: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const M: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
        const P: u64,
        const ETA1: usize,
        const ETA2: usize,
    > SPIRAL
    for SPIRALImpl<
        N,
        N_PLUS_ONE,
        Q,
        Q_A,
        Q_B,
        Q_A_INV,
        Q_B_INV,
        D,
        W1,
        W2,
        G_BASE,
        G_LEN,
        M,
        NOISE_WIDTH_MILLIONTHS,
        P,
        ETA1,
        ETA2,
    >
{
    // Type aliases
    type RingP = IntModCyclo<D, P>;
    type RingQ = IntModCyclo<D, Q>;
    type RingQFast = IntModCycloCRTEval<D, Q_A, Q_B, Q_A_INV, Q_B_INV, W1, W2>;
    type MatrixP = Matrix<N, N, Self::RingP>;
    type MatrixQ = Matrix<N, N, Self::RingQ>;
    type MatrixQFast = Matrix<N, N, Self::RingQFast>;
    type MatrixRegevCiphertext = Matrix<N_PLUS_ONE, N, Self::RingQFast>;
    type GSWCiphertext = Matrix<N_PLUS_ONE, M, Self::RingQFast>;

    // Associated types
    type QueryKey = Matrix<N, 1, Self::RingQFast>;
    type Query = (Vec<Self::MatrixRegevCiphertext>, Vec<Self::GSWCiphertext>);
    type Response = Self::MatrixRegevCiphertext;
    type Record = Self::MatrixP;
    type RecordPreprocessed = Self::MatrixQFast;

    // Constants
    const DB_SIZE: usize = 1 << (ETA1 + ETA2);
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;
    // const ETA1_MASK: usize = (1 << ETA1) - 1;
    // const ETA2_MASK: usize = (1 << ETA2) - 1;

    fn preprocess(record: &<Self as SPIRAL>::Record) -> <Self as SPIRAL>::RecordPreprocessed {
        record.into_ring(|x| <Self as SPIRAL>::RingQFast::from(&x.include_into::<Q>()))
    }

    fn setup() -> <Self as SPIRAL>::QueryKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn query(qk: &<Self as SPIRAL>::QueryKey, idx: usize) -> <Self as SPIRAL>::Query {
        // Rust won't allow these to be const for some annoying reason.
        #[allow(non_snake_case)]
        let ETA1_MASK: usize = (1 << Self::ETA1) - 1;
        #[allow(non_snake_case)]
        let ETA2_MASK: usize = (1 << Self::ETA2) - 1;
        let idx_i = (idx >> ETA2) & ETA1_MASK;
        let idx_j = idx & ETA2_MASK;

        let mut regevs: Vec<<Self as SPIRAL>::MatrixRegevCiphertext> =
            Vec::with_capacity(1 << ETA1);
        let one = <Self as SPIRAL>::MatrixP::identity().into_ring(|x| x.scale_up_into());
        let zero = <Self as SPIRAL>::MatrixQ::zero();
        for i in 0..(1 << ETA1) {
            regevs.push(if i == idx_i {
                Self::encode_regev(&qk, &one)
            } else {
                Self::encode_regev(&qk, &zero)
            });
        }

        let mut gsws: Vec<<Self as SPIRAL>::GSWCiphertext> = Vec::with_capacity(ETA2);
        for j in 0..ETA2 {
            gsws.push(if (idx_j >> (ETA2 - j - 1)) & 1 != 0 {
                Self::encode_gsw(&qk, &1_u64.into())
            } else {
                Self::encode_gsw(&qk, &0_u64.into())
            });
        }

        (regevs, gsws)
    }

    fn answer(
        db: &Vec<<Self as SPIRAL>::RecordPreprocessed>,
        (regevs, gsws): &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::Response {
        let db_at = |i: usize, j: usize| &db[(i << ETA2) + j];
        let mut curr: Vec<<Self as SPIRAL>::MatrixRegevCiphertext> = Vec::with_capacity(1 << ETA2);
        for j in 0..(1 << ETA2) {
            let mut sum = Self::regev_mul_scalar(&regevs[0], db_at(0, j));
            for i in 1..(1 << ETA1) {
                Self::regev_add_eq_mul_scalar(&mut sum, &regevs[i], db_at(i, j));
            }
            curr.push(sum.convert_ring());
        }

        for r in 0..ETA2 {
            let curr_size = 1 << (ETA2 - r - 1);
            curr.truncate(2 * curr_size);
            for j in 0..curr_size {
                let b = &gsws[r];
                let c0 = &curr[j];
                let c1 = &curr[curr_size + j];
                let c1_sub_c0 = Self::regev_sub_hom(c1, c0);
                curr[j] += &Self::hybrid_mul_hom(&c1_sub_c0, &b);
            }
        }
        curr.remove(0)
    }

    fn extract(
        qk: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
    ) -> <Self as SPIRAL>::MatrixP {
        Self::decode_regev(qk, r).into_ring(|x| x.round_down_into())
    }
}

impl<
        const N: usize,
        const N_PLUS_ONE: usize,
        const Q: u64,
        const Q_A: u64,
        const Q_B: u64,
        const Q_A_INV: u64,
        const Q_B_INV: u64,
        const D: usize,
        const W1: u64,
        const W2: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const M: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
        const P: u64,
        const ETA1: usize,
        const ETA2: usize,
    >
    SPIRALImpl<
        N,
        N_PLUS_ONE,
        Q,
        Q_A,
        Q_B,
        Q_A_INV,
        Q_B_INV,
        D,
        W1,
        W2,
        G_BASE,
        G_LEN,
        M,
        NOISE_WIDTH_MILLIONTHS,
        P,
        ETA1,
        ETA2,
    >
{
    fn encode_regev(
        qk: &<Self as SPIRAL>::QueryKey,
        mu: &<Self as SPIRAL>::MatrixQ,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, N, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, N, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<N_PLUS_ONE, N, <Self as SPIRAL>::RingQFast> = Matrix::stack(
            &a_t,
            &(&(&(qk * &a_t) + &e_mat) + &mu.into_ring(|x| <Self as SPIRAL>::RingQFast::from(x))),
        );
        c_mat
    }

    fn encode_gsw(
        qk: &<Self as SPIRAL>::QueryKey,
        mu: &<Self as SPIRAL>::RingP,
    ) -> <Self as SPIRAL>::GSWCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, M, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<N_PLUS_ONE, M, <Self as SPIRAL>::RingQFast> =
            &Matrix::stack(&a_t, &(&(qk * &a_t) + &e_mat))
                + &(&build_gadget::<<Self as SPIRAL>::RingQFast, N_PLUS_ONE, M, G_BASE, G_LEN>()
                    * &<Self as SPIRAL>::RingQFast::from(&mu.include_into::<Q>()));
        c_mat
    }

    fn regev_sub_hom(
        lhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
        rhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
        lhs - rhs
    }
    fn regev_mul_scalar(
        lhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
        rhs: &<Self as SPIRAL>::MatrixQFast,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
        lhs * rhs
    }
    fn regev_add_eq_mul_scalar(
        lhs: &mut <Self as SPIRAL>::MatrixRegevCiphertext,
        rhs_a: &<Self as SPIRAL>::MatrixRegevCiphertext,
        rhs_b: &<Self as SPIRAL>::MatrixQFast,
    ) {
        lhs.add_eq_mul(rhs_a, rhs_b);
    }

    fn hybrid_mul_hom(
        regev: &<Self as SPIRAL>::MatrixRegevCiphertext,
        gsw: &<Self as SPIRAL>::GSWCiphertext,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
        gsw * &gadget_inverse::<<Self as SPIRAL>::RingQFast, N_PLUS_ONE, M, N, G_BASE, G_LEN>(regev)
    }

    fn decode_regev(
        qk: &<Self as SPIRAL>::QueryKey,
        c: &<Self as SPIRAL>::MatrixRegevCiphertext,
    ) -> <Self as SPIRAL>::MatrixQ {
        (&Matrix::append(&-qk, &Matrix::<N, N, _>::identity()) * c)
            .into_ring(|x| <Self as SPIRAL>::RingQ::from(x))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    const SPIRAL_TEST_PARAMS: SPIRALParams = SPIRALParamsRaw {
        N: 2,
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        W1: 66294444,
        W2: 30909463,
        G_BASE: 128,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 1 << 8,
        ETA1: 9,
        ETA2: 6,
    }
    .expand();

    type SPIRALTest = spiral!(SPIRAL_TEST_PARAMS);

    #[test]
    fn test_spiral_one() {
        run_spiral::<SPIRALTest, _>([11111].into_iter());
    }
    #[test]
    fn test_spiral() {
        run_spiral::<SPIRALTest, _>([0, 11111, SPIRALTest::DB_SIZE - 1].into_iter());
    }

    #[ignore]
    #[test]
    fn test_spiral_stress() {
        run_spiral::<SPIRALTest, _>(0..SPIRALTest::DB_SIZE);
    }

    fn run_spiral<
        TheSPIRAL: SPIRAL<Record = Matrix<2, 2, IntModCyclo<2048, 256>>>,
        I: Iterator<Item = usize>,
    >(
        iter: I,
    ) {
        let mut db: Vec<<TheSPIRAL as SPIRAL>::Record> = Vec::with_capacity(SPIRALTest::DB_SIZE);
        for i in 0..TheSPIRAL::DB_SIZE as u64 {
            let mut record: <TheSPIRAL as SPIRAL>::Record = Matrix::zero();
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
        let mut db_pre: Vec<<TheSPIRAL as SPIRAL>::RecordPreprocessed> =
            Vec::with_capacity(TheSPIRAL::DB_SIZE);
        for i in 0..TheSPIRAL::DB_SIZE {
            db_pre.push(TheSPIRAL::preprocess(&db[i]));
        }
        let end = Instant::now();
        eprintln!("{:?} to preprocess", end - start);

        let qk = TheSPIRAL::setup();
        let check = |idx: usize| {
            let cts = TheSPIRAL::query(&qk, idx);
            let result = TheSPIRAL::answer(&db_pre, &cts);
            let extracted = TheSPIRAL::extract(&qk, &result);
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
