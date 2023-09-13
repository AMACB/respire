use crate::math::gadget::{build_gadget, gadget_inverse};
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::matrix::Matrix;
use crate::math::number_theory::find_sqrt_primitive_root;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::ring_elem::{RingCompatible, RingElement};
use crate::math::utils::{ceil_log, floor_log, mod_inverse};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::cmp::max;

pub struct SPIRALImpl<
    const N: usize,
    const N_PLUS_ONE: usize,
    const Q: u64,
    const Q_A: u64,
    const Q_B: u64,
    const Q_A_INV: u64,
    const Q_B_INV: u64,
    const D: usize,
    const W_A: u64,
    const W_B: u64,
    const G_BASE: u64,
    const G_LEN: usize,
    const M: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
    const P: u64,
    const ETA1: usize,
    const ETA2: usize,
    const FOLD_BASE: usize,
> {}

#[allow(non_snake_case)]
pub struct SPIRALParamsRaw {
    pub N: usize,
    pub Q_A: u64,
    pub Q_B: u64,
    pub D: usize,
    pub G_BASE: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub ETA1: usize,
    pub ETA2: usize,
    pub FOLD_BASE: usize,
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
            W_A: find_sqrt_primitive_root(self.D, self.Q_A),
            W_B: find_sqrt_primitive_root(self.D, self.Q_B),
            G_BASE: self.G_BASE,
            G_LEN: t,
            M: (self.N + 1) * t,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
            P: self.P,
            ETA1: self.ETA1,
            ETA2: self.ETA2,
            FOLD_BASE: self.FOLD_BASE,
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
    pub W_A: u64,
    pub W_B: u64,
    pub G_BASE: u64,
    pub G_LEN: usize,
    pub M: usize,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub ETA1: usize,
    pub ETA2: usize,
    pub FOLD_BASE: usize,
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
            {$params.W_A},
            {$params.W_B},
            {$params.G_BASE},
            {$params.G_LEN},
            {$params.M},
            {$params.NOISE_WIDTH_MILLIONTHS},
            {$params.P},
            {$params.ETA1},
            {$params.ETA2},
            {$params.FOLD_BASE},
        >
    }
}

pub trait SPIRAL {
    // Type aliases
    type RingP;
    type RingQ;
    type Ring0;
    type RingQFast;
    type Ring0Fast;
    type MatrixP;
    type MatrixQ;
    type MatrixQFast;
    type MatrixRegevCiphertext;
    type MatrixRegevCiphertext0;
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
    fn response_error(
        qk: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
        actual: &<Self as SPIRAL>::Record,
    ) -> f64;
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
        const W_A: u64,
        const W_B: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const M: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
        const P: u64,
        const ETA1: usize,
        const ETA2: usize,
        const FOLD_BASE: usize,
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
        W_A,
        W_B,
        G_BASE,
        G_LEN,
        M,
        NOISE_WIDTH_MILLIONTHS,
        P,
        ETA1,
        ETA2,
        FOLD_BASE,
    >
{
    // Type aliases
    type RingP = IntModCyclo<D, P>;
    type RingQ = IntModCyclo<D, Q>;
    type Ring0 = IntModCyclo<D, 0>;
    type RingQFast = IntModCycloCRTEval<D, Q_A, Q_B, Q_A_INV, Q_B_INV, W_A, W_B>;
    type Ring0Fast = IntModCycloCRTEval<D, 0, 0, 0, 0, 0, 0>;
    type MatrixP = Matrix<N, N, Self::RingP>;
    type MatrixQ = Matrix<N, N, Self::RingQ>;
    type MatrixQFast = Matrix<N, N, Self::RingQFast>;
    type MatrixRegevCiphertext = Matrix<N_PLUS_ONE, N, Self::RingQFast>;
    type MatrixRegevCiphertext0 = Matrix<N_PLUS_ONE, N, Self::Ring0Fast>;
    type GSWCiphertext = Matrix<N_PLUS_ONE, M, Self::RingQFast>;

    // Associated types
    type QueryKey = Matrix<N, 1, Self::RingQFast>;
    type Query = (Vec<Self::MatrixRegevCiphertext>, Vec<Self::GSWCiphertext>);
    type Response = Self::MatrixRegevCiphertext;
    type Record = Self::MatrixP;
    type RecordPreprocessed = Self::MatrixQFast;

    // Constants
    const DB_SIZE: usize = 2_usize.pow(ETA1 as u32) * FOLD_BASE.pow(ETA2 as u32);
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;

    fn preprocess(record: &<Self as SPIRAL>::Record) -> <Self as SPIRAL>::RecordPreprocessed {
        record.into_ring(|x| <Self as SPIRAL>::RingQFast::from(&x.include_into::<Q>()))
    }

    fn setup() -> <Self as SPIRAL>::QueryKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let s: Matrix<N, 1, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        s
    }

    fn query(qk: &<Self as SPIRAL>::QueryKey, idx: usize) -> <Self as SPIRAL>::Query {
        assert!(idx < Self::DB_SIZE);
        let fold_size: usize = FOLD_BASE.pow(Self::ETA2 as u32);

        let idx_i = idx / fold_size;
        let idx_j = idx % fold_size;

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
        let mut digits = Vec::with_capacity(ETA2);
        let mut idx_j_curr = idx_j;
        for _ in 0..ETA2 {
            digits.push(idx_j_curr % FOLD_BASE);
            idx_j_curr /= FOLD_BASE;
        }
        let encode_bit_gsw = |b: bool| -> <Self as SPIRAL>::GSWCiphertext {
            match b {
                false => Self::encode_gsw(&qk, &0_u64.into()),
                true => Self::encode_gsw(&qk, &1_u64.into()),
            }
        };
        for digit in digits.into_iter().rev() {
            for which in 1..FOLD_BASE {
                gsws.push(encode_bit_gsw(digit == which));
            }
        }

        (regevs, gsws)
    }

    fn answer(
        db: &Vec<<Self as SPIRAL>::RecordPreprocessed>,
        (regevs, gsws): &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::Response {
        let fold_size: usize = FOLD_BASE.pow(Self::ETA2 as u32);

        let db_at = |i: usize, j: usize| &db[i * fold_size + j];
        let mut curr: Vec<<Self as SPIRAL>::MatrixRegevCiphertext> = Vec::with_capacity(fold_size);
        for j in 0..fold_size {
            // Norm is at most N * max(Q_A, Q_B)^2 for each term
            // Add one for margin
            let reduce_every = 1 << (64 - 2 * ceil_log(2, max(Q_A, Q_B)) - N - 1);
            let mut sum = Self::regev_mul_scalar_no_reduce(&regevs[0], db_at(0, j));
            for i in 1..(1 << ETA1) {
                Self::regev_add_eq_mul_scalar_no_reduce(&mut sum, &regevs[i], db_at(i, j));
                if i % reduce_every == 0 {
                    sum.iter_do(|r| Self::RingQFast::reduce_mod(r));
                }
            }
            sum.iter_do(|r| Self::RingQFast::reduce_mod(r));
            curr.push(sum.convert_ring());
        }

        let mut curr_size = fold_size;
        for gsw_idx in 0..ETA2 {
            curr.truncate(curr_size);
            for fold_idx in 0..curr_size / FOLD_BASE {
                let c0 = curr[fold_idx].clone();
                for i in 1..FOLD_BASE {
                    let c_i = &curr[i * curr_size / FOLD_BASE + fold_idx];
                    let c_i_sub_c0 = Self::regev_sub_hom(c_i, &c0);
                    let b = &gsws[gsw_idx * (FOLD_BASE - 1) + i - 1];
                    let c_i_sub_c0_mul_b = Self::hybrid_mul_hom(&c_i_sub_c0, &b);
                    curr[fold_idx] += &c_i_sub_c0_mul_b;
                }
            }
            curr_size /= FOLD_BASE;
        }
        curr.remove(0)
    }

    fn extract(
        qk: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
    ) -> <Self as SPIRAL>::Record {
        Self::decode_regev(qk, r).into_ring(|x| x.round_down_into())
    }

    fn response_error(
        qk: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
        actual: &<Self as SPIRAL>::Record,
    ) -> f64 {
        let actual_scaled = actual.into_ring(|x| x.scale_up_into());
        let decoded = Self::decode_regev(qk, r);
        let diff = &actual_scaled - &decoded;
        (diff.norm() as f64) / (Q as f64)
        // let mut err = 0_f64;
        // let mut ct = 0_usize;
        // for i in 0..N {
        //     for j in 0..N {
        //         for e in diff[(i, j)].coeff_iter() {
        //             let rel_e = (u64::from(*e) as f64) / (Q as f64);
        //             err += rel_e * rel_e;
        //             ct += 1;
        //         }
        //     }
        // }
        // (err / (ct as f64)) * (Q as f64)
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
        const W_A: u64,
        const W_B: u64,
        const G_BASE: u64,
        const G_LEN: usize,
        const M: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
        const P: u64,
        const ETA1: usize,
        const ETA2: usize,
        const FOLD_BASE: usize,
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
        W_A,
        W_B,
        G_BASE,
        G_LEN,
        M,
        NOISE_WIDTH_MILLIONTHS,
        P,
        ETA1,
        ETA2,
        FOLD_BASE,
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

    // fn regev_mul_scalar(
    //     lhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
    //     rhs: &<Self as SPIRAL>::MatrixQFast,
    // ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
    //     lhs * rhs
    // }

    fn regev_mul_scalar_no_reduce(
        lhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
        rhs: &<Self as SPIRAL>::MatrixQFast,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext0 {
        let mut result: <Self as SPIRAL>::MatrixRegevCiphertext0 = Matrix::zero();
        lhs.mul_iter_do(&rhs, |(r, c), lhs_r, rhs_r| {
            result[(r, c)].add_eq_mul(&lhs_r.convert_ref(), &rhs_r.convert_ref());
        });
        result
    }

    // fn regev_add_eq_mul_scalar(
    //     lhs: &mut <Self as SPIRAL>::MatrixRegevCiphertext,
    //     rhs_a: &<Self as SPIRAL>::MatrixRegevCiphertext,
    //     rhs_b: &<Self as SPIRAL>::MatrixQFast,
    // ) {
    //     lhs.add_eq_mul(rhs_a, rhs_b);
    // }

    fn regev_add_eq_mul_scalar_no_reduce(
        lhs: &mut <Self as SPIRAL>::MatrixRegevCiphertext0,
        rhs_a: &<Self as SPIRAL>::MatrixRegevCiphertext,
        rhs_b: &<Self as SPIRAL>::MatrixQFast,
    ) {
        rhs_a.mul_iter_do(rhs_b, |(r, c), rhs_a_r, rhs_b_r| {
            lhs[(r, c)].add_eq_mul(rhs_a_r.convert_ref(), rhs_b_r.convert_ref());
        });
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
    use rand::Rng;
    use std::time::Instant;

    const SPIRAL_TEST_PARAMS: SPIRALParams = SPIRALParamsRaw {
        N: 2,
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        G_BASE: 1 << 20,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 1 << 8,
        ETA1: 9,
        ETA2: 3,
        FOLD_BASE: 4,
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
        let mut rng = ChaCha20Rng::from_entropy();
        run_spiral::<SPIRALTest, _>((0..).map(|_| rng.gen_range(0_usize..SPIRALTest::DB_SIZE)))
    }

    // struct RunResult {
    //     success: bool,
    //     noise: f64,
    //     // preprocess_time: Duration,
    //     // setup_time: Duration,
    //     query_time: Duration,
    //     answer_time: Duration,
    //     extract_time: Duration,
    // }

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

        let pre_start = Instant::now();
        let mut db_pre: Vec<<TheSPIRAL as SPIRAL>::RecordPreprocessed> =
            Vec::with_capacity(TheSPIRAL::DB_SIZE);
        for i in 0..TheSPIRAL::DB_SIZE {
            db_pre.push(TheSPIRAL::preprocess(&db[i]));
        }
        let pre_end = Instant::now();
        eprintln!("{:?} to preprocess", pre_end - pre_start);

        let setup_start = Instant::now();
        let qk = TheSPIRAL::setup();
        let setup_end = Instant::now();
        eprintln!("{:?} to setup", setup_end - setup_start);

        let check = |idx: usize| {
            eprintln!("Running with idx = {}", idx);
            let query_start = Instant::now();
            let cts = TheSPIRAL::query(&qk, idx);
            let query_end = Instant::now();
            let query_total = query_end - query_start;

            let answer_start = Instant::now();
            let result = TheSPIRAL::answer(&db_pre, &cts);
            let answer_end = Instant::now();
            let answer_total = answer_end - answer_start;

            let extract_start = Instant::now();
            let extracted = TheSPIRAL::extract(&qk, &result);
            let extract_end = Instant::now();
            let extract_total = extract_end - extract_start;

            if &extracted != &db[idx] {
                eprintln!("  **** protocol failed");
            }
            eprintln!("  {:?} total", query_total + answer_total + extract_total);
            eprintln!("    {:?} to query", query_total);
            eprintln!("    {:?} to answer", answer_total);
            eprintln!("    {:?} to extract", extract_total);
            let err = TheSPIRAL::response_error(&qk, &result, &db[idx]);
            eprintln!("  relative error: 2^({})", err.log2());
        };

        for i in iter {
            check(i);
        }
    }
}
