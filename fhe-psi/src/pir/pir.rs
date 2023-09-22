use std::cmp::max;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::math::gadget::{build_gadget, gadget_inverse, RingElementDecomposable};
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::matrix::Matrix;
use crate::math::number_theory::find_sqrt_primitive_root;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::ring_elem::{RingCompatible, RingElement};
use crate::math::utils::{ceil_log, floor_log, mod_inverse};

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
    const Z_GSW: u64,
    const T_GSW: usize,
    const Z_COEFF: u64,
    const T_COEFF: usize,
    const Z_CONV: u64,
    const T_CONV: usize,
    const T_CONV_TIMES_TWO: usize,
    const M_CONV: usize,
    const M: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
    const P: u64,
    const ETA1: usize,
    const ETA2: usize,
    const Z_FOLD: usize,
> {}

#[allow(non_snake_case)]
pub struct SPIRALParamsRaw {
    pub N: usize,
    pub Q_A: u64,
    pub Q_B: u64,
    pub D: usize,
    pub Z_GSW: u64,
    pub Z_COEFF: u64,
    pub Z_CONV: u64,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub ETA1: usize,
    pub ETA2: usize,
    pub Z_FOLD: usize,
}

impl SPIRALParamsRaw {
    pub const fn expand(&self) -> SPIRALParams {
        let q = self.Q_A * self.Q_B;
        let t_gsw = floor_log(self.Z_GSW, q) + 1;
        let t_coeff = floor_log(self.Z_COEFF, q) + 1;
        let t_conv = floor_log(self.Z_CONV, q) + 1;
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
            Z_GSW: self.Z_GSW,
            T_GSW: t_gsw,
            Z_COEFF: self.Z_COEFF,
            T_COEFF: t_coeff,
            Z_CONV: self.Z_CONV,
            T_CONV: t_conv,
            T_CONV_TIMES_TWO: t_conv * 2,
            M_CONV: self.N * t_conv,
            M: (self.N + 1) * t_gsw,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
            P: self.P,
            ETA1: self.ETA1,
            ETA2: self.ETA2,
            Z_FOLD: self.Z_FOLD,
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
    pub Z_GSW: u64,
    pub T_GSW: usize,
    pub Z_COEFF: u64,
    pub T_COEFF: usize,
    pub Z_CONV: u64,
    pub T_CONV: usize,
    pub T_CONV_TIMES_TWO: usize,
    pub M_CONV: usize,
    pub M: usize,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub ETA1: usize,
    pub ETA2: usize,
    pub Z_FOLD: usize,
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
            {$params.Z_GSW},
            {$params.T_GSW},
            {$params.Z_COEFF},
            {$params.T_COEFF},
            {$params.Z_CONV},
            {$params.T_CONV},
            {$params.T_CONV_TIMES_TWO},
            {$params.M_CONV},
            {$params.M},
            {$params.NOISE_WIDTH_MILLIONTHS},
            {$params.P},
            {$params.ETA1},
            {$params.ETA2},
            {$params.Z_FOLD},
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
    type ScalarRegevCiphertext;
    type MatrixRegevCiphertext;
    type MatrixRegevCiphertext0;
    type GSWCiphertext;
    type ScalarKey;
    type MatrixKey;
    type AutoKey;
    type ScalToMatKey;
    type RegevToGSWKey;

    // Associated types
    type QueryKey;
    type PublicParams;
    type Query;
    type QueryExpanded;
    type Response;
    type Record;
    type RecordPreprocessed;

    // Constants
    const DB_SIZE: usize;
    const ETA1: usize;
    const ETA2: usize;
    const REGEV_COUNT: usize;
    const REGEV_EXPAND_ITERS: usize;
    const GSW_COUNT: usize;
    const GSW_EXPAND_ITERS: usize;

    fn preprocess(record: &Self::Record) -> Self::RecordPreprocessed;
    fn setup() -> (Self::QueryKey, Self::PublicParams);
    fn query(qk: &Self::QueryKey, idx: usize) -> Self::Query;
    fn query_expand(pp: &Self::PublicParams, q: &Self::Query) -> Self::QueryExpanded;
    fn answer(
        db: &Vec<Self::RecordPreprocessed>,
        q_expanded: &Self::QueryExpanded,
    ) -> Self::Response;
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
        const Z_GSW: u64,
        const T_GSW: usize,
        const Z_COEFF: u64,
        const T_COEFF: usize,
        const Z_CONV: u64,
        const T_CONV: usize,
        const T_CONV_TIMES_TWO: usize,
        const M_CONV: usize,
        const M: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
        const P: u64,
        const ETA1: usize,
        const ETA2: usize,
        const Z_FOLD: usize,
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
        Z_GSW,
        T_GSW,
        Z_COEFF,
        T_COEFF,
        Z_CONV,
        T_CONV,
        T_CONV_TIMES_TWO,
        M_CONV,
        M,
        NOISE_WIDTH_MILLIONTHS,
        P,
        ETA1,
        ETA2,
        Z_FOLD,
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
    type ScalarRegevCiphertext = Matrix<2, 1, Self::RingQFast>;
    type MatrixRegevCiphertext = Matrix<N_PLUS_ONE, N, Self::RingQFast>;
    type MatrixRegevCiphertext0 = Matrix<N_PLUS_ONE, N, Self::Ring0Fast>;
    type GSWCiphertext = Matrix<N_PLUS_ONE, M, Self::RingQFast>;

    type ScalarKey = Self::RingQFast;
    type MatrixKey = Matrix<N, 1, Self::RingQFast>;
    type AutoKey = (Matrix<2, T_COEFF, Self::RingQFast>, usize);
    type ScalToMatKey = Matrix<N_PLUS_ONE, M_CONV, Self::RingQFast>;
    type RegevToGSWKey = (
        Matrix<N_PLUS_ONE, T_CONV_TIMES_TWO, Self::RingQFast>,
        Self::ScalToMatKey,
    );

    // Associated types
    type QueryKey = (Self::ScalarKey, Self::MatrixKey);
    type PublicParams = (Vec<Self::AutoKey>, Self::ScalToMatKey, Self::RegevToGSWKey);
    type Query = Self::ScalarRegevCiphertext;
    type QueryExpanded = (Vec<Self::MatrixRegevCiphertext>, Vec<Self::GSWCiphertext>);
    type Response = Self::MatrixRegevCiphertext;
    type Record = Self::MatrixP;
    type RecordPreprocessed = Self::MatrixQFast;

    // Constants
    const DB_SIZE: usize = 2_usize.pow(ETA1 as u32) * Z_FOLD.pow(ETA2 as u32);
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;

    const REGEV_COUNT: usize = 1 << ETA1;
    const REGEV_EXPAND_ITERS: usize = ETA1;
    const GSW_COUNT: usize = ETA2 * (Z_FOLD - 1) * T_GSW;
    const GSW_EXPAND_ITERS: usize = ceil_log(2, Self::GSW_COUNT as u64);

    fn preprocess(record: &<Self as SPIRAL>::Record) -> <Self as SPIRAL>::RecordPreprocessed {
        record.into_ring(|x| <Self as SPIRAL>::RingQFast::from(&x.include_into::<Q>()))
    }

    fn setup() -> (<Self as SPIRAL>::QueryKey, <Self as SPIRAL>::PublicParams) {
        // Scalar Regev secret key
        let s_scalar = Self::scalar_regev_setup();

        // Matrix Regev / GSW secret key
        let s_matrix = Self::matrix_regev_setup();

        // Automorphism keys
        let auto_key_ct = 1 + max(Self::REGEV_EXPAND_ITERS, Self::GSW_EXPAND_ITERS);
        let mut auto_keys: Vec<<Self as SPIRAL>::AutoKey> = Vec::with_capacity(auto_key_ct);
        for i in 0..auto_key_ct {
            let tau_power = (D >> i) + 1;
            auto_keys.push(Self::auto_setup(tau_power, &s_scalar));
        }

        // Scalar to matrix key. Technically, this could be shared by the regev to GSW key.
        let scal_to_mat_key = Self::scal_to_mat_setup(&s_scalar, &s_matrix);

        // Regev to GSW key
        let regev_to_gsw_key = Self::regev_to_gsw_setup(&s_scalar, &s_matrix);

        (
            (s_scalar, s_matrix),
            (auto_keys, scal_to_mat_key, regev_to_gsw_key),
        )
    }

    fn query((s_scalar, _): &<Self as SPIRAL>::QueryKey, idx: usize) -> <Self as SPIRAL>::Query {
        assert!(idx < Self::DB_SIZE);
        let fold_size: usize = Z_FOLD.pow(Self::ETA2 as u32);

        let idx_i = idx / fold_size;
        let idx_j = idx % fold_size;

        let mut packed_vec: Vec<IntMod<Q>> = Vec::with_capacity(D);
        for _ in 0..D {
            packed_vec.push(IntMod::zero());
        }

        let inv_even = IntMod::from(mod_inverse(1 << (1 + Self::REGEV_EXPAND_ITERS), Q));
        for i in 0_usize..(1 << ETA1) {
            packed_vec[2 * i] = (IntMod::<P>::from((i == idx_i) as u64)).scale_up_into() * inv_even;
        }

        let mut digits = Vec::with_capacity(ETA2);
        let mut idx_j_curr = idx_j;
        for _ in 0..ETA2 {
            digits.push(idx_j_curr % Z_FOLD);
            idx_j_curr /= Z_FOLD;
        }

        // Think of the odd entries of packed as [ETA2] x [Z_FOLD - 1] x [T_GSW]
        let inv_odd = IntMod::from(mod_inverse(1 << (1 + Self::GSW_EXPAND_ITERS), Q));
        for (digit_idx, digit) in digits.into_iter().rev().enumerate() {
            for which in 0..Z_FOLD - 1 {
                let mut msg = IntMod::from((digit == which + 1) as u64);
                for gsw_pow in 0..T_GSW {
                    let pack_idx = T_GSW * ((Z_FOLD - 1) * digit_idx + which) + gsw_pow;
                    packed_vec[2 * pack_idx + 1] = msg * inv_odd;
                    msg *= IntMod::from(Z_GSW);
                }
            }
        }

        let mu: IntModCyclo<D, Q> = packed_vec.into();
        Self::encode_scalar_regev(s_scalar, &mu)
    }

    fn query_expand(
        (auto_keys, scal_to_mat_key, regev_to_gsw_key): &<Self as SPIRAL>::PublicParams,
        q: &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::QueryExpanded {
        let do_expand_iter = |i: usize,
                              cts: &Vec<<Self as SPIRAL>::ScalarRegevCiphertext>|
         -> Vec<<Self as SPIRAL>::ScalarRegevCiphertext> {
            let auto_key = &auto_keys[i];
            let new_len = 1 << i;
            let mut cts_new = Vec::with_capacity(new_len);
            cts_new.resize(new_len, Matrix::zero());
            for j in 0..new_len / 2 {
                let shifted = Self::scalar_regev_mul_x_pow(&cts[j], 2 * D - (1 << i));
                cts_new[j] = &cts[j] + &Self::auto_hom(auto_key, &cts[j]);
                cts_new[j + new_len / 2] = &shifted + &Self::auto_hom(auto_key, &shifted);
            }
            cts_new
        };

        let regev_base = q + &Self::auto_hom(&auto_keys[0], &q);
        let mut regevs: Vec<Self::ScalarRegevCiphertext> = vec![regev_base];
        for i in 1..Self::REGEV_EXPAND_ITERS + 1 {
            regevs = do_expand_iter(i, &regevs);
        }
        regevs.truncate(1 << Self::ETA1);

        let regevs: Vec<Self::MatrixRegevCiphertext> = regevs
            .iter()
            .map(|c| Self::scal_to_mat(&scal_to_mat_key, c))
            .collect();

        let q_shifted = Self::scalar_regev_mul_x_pow(&q, 2 * D - 1);
        let gsw_base = &q_shifted + &Self::auto_hom(&auto_keys[0], &q_shifted);
        let mut gsws = vec![gsw_base];
        for i in 1..Self::GSW_EXPAND_ITERS + 1 {
            gsws = do_expand_iter(i, &gsws);
        }
        gsws.truncate(Self::ETA2 * (Z_FOLD - 1) * T_GSW);

        let gsws: Vec<Self::GSWCiphertext> = gsws
            .chunks(T_GSW)
            .map(|cs| Self::regev_to_gsw(&regev_to_gsw_key, cs))
            .collect();

        (regevs, gsws)
    }

    fn answer(
        db: &Vec<<Self as SPIRAL>::RecordPreprocessed>,
        (regevs, gsws): &<Self as SPIRAL>::QueryExpanded,
    ) -> <Self as SPIRAL>::Response {
        // First dimension processing
        let fold_size: usize = Z_FOLD.pow(Self::ETA2 as u32);
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

        // Folding
        let mut curr_size = fold_size;
        for gsw_idx in 0..ETA2 {
            curr.truncate(curr_size);
            for fold_idx in 0..curr_size / Z_FOLD {
                let c0 = curr[fold_idx].clone();
                for i in 1..Z_FOLD {
                    let c_i = &curr[i * curr_size / Z_FOLD + fold_idx];
                    let c_i_sub_c0 = Self::regev_sub_hom(c_i, &c0);
                    let b = &gsws[gsw_idx * (Z_FOLD - 1) + i - 1];
                    let c_i_sub_c0_mul_b = Self::hybrid_mul_hom(&c_i_sub_c0, &b);
                    curr[fold_idx] += &c_i_sub_c0_mul_b;
                }
            }
            curr_size /= Z_FOLD;
        }
        curr.remove(0)
    }

    fn extract(
        (_, s_mat): &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
    ) -> <Self as SPIRAL>::Record {
        Self::decode_matrix_regev(s_mat, r).into_ring(|x| x.round_down_into())
    }

    fn response_error(
        (_, s_mat): &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
        actual: &<Self as SPIRAL>::Record,
    ) -> f64 {
        let actual_scaled = actual.into_ring(|x| x.scale_up_into());
        let decoded = Self::decode_matrix_regev(&s_mat, r);
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
        const Z_GSW: u64,
        const T_GSW: usize,
        const Z_COEFF: u64,
        const T_COEFF: usize,
        const Z_CONV: u64,
        const T_CONV: usize,
        const T_CONV_TIMES_TWO: usize,
        const M_CONV: usize,
        const M: usize,
        const NOISE_WIDTH_MILLIONTHS: u64,
        const P: u64,
        const ETA1: usize,
        const ETA2: usize,
        const Z_FOLD: usize,
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
        Z_GSW,
        T_GSW,
        Z_COEFF,
        T_COEFF,
        Z_CONV,
        T_CONV,
        T_CONV_TIMES_TWO,
        M_CONV,
        M,
        NOISE_WIDTH_MILLIONTHS,
        P,
        ETA1,
        ETA2,
        Z_FOLD,
    >
{
    fn scalar_regev_setup() -> <Self as SPIRAL>::RingQFast {
        let mut rng = ChaCha20Rng::from_entropy();
        <Self as SPIRAL>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng)
    }

    fn matrix_regev_setup() -> Matrix<N, 1, <Self as SPIRAL>::RingQFast> {
        let mut rng = ChaCha20Rng::from_entropy();
        Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng)
    }

    fn encode_scalar_regev(
        s_scalar: &<Self as SPIRAL>::ScalarKey,
        mu: &<Self as SPIRAL>::RingQ,
    ) -> <Self as SPIRAL>::ScalarRegevCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut c = Matrix::zero();
        c[(0, 0)] = <Self as SPIRAL>::RingQFast::rand_uniform(&mut rng);
        let e = <Self as SPIRAL>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(
            &mut rng,
        );
        let mut c1 = &c[(0, 0)] * s_scalar;
        c1 += &e;
        c1 += &<Self as SPIRAL>::RingQFast::from(mu);
        c[(1, 0)] = c1;
        c
    }

    fn decode_scalar_regev(
        s_scalar: &<Self as SPIRAL>::ScalarKey,
        c: &<Self as SPIRAL>::ScalarRegevCiphertext,
    ) -> <Self as SPIRAL>::RingQ {
        <Self as SPIRAL>::RingQ::from(&(&c[(1, 0)] - &(&c[(0, 0)] * s_scalar)))
    }

    fn scalar_regev_mul_x_pow(
        c: &<Self as SPIRAL>::ScalarRegevCiphertext,
        k: usize,
    ) -> <Self as SPIRAL>::ScalarRegevCiphertext {
        let mut result = Matrix::zero();
        result[(0, 0)] = c[(0, 0)].mul_x_pow(k);
        result[(1, 0)] = c[(1, 0)].mul_x_pow(k);
        result
    }

    fn encode_gsw(
        s_mat: &<Self as SPIRAL>::MatrixKey,
        mu: &<Self as SPIRAL>::RingQ,
    ) -> <Self as SPIRAL>::GSWCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, M, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<N_PLUS_ONE, M, <Self as SPIRAL>::RingQFast> =
            &Matrix::stack(&a_t, &(&(s_mat * &a_t) + &e_mat))
                + &(&build_gadget::<<Self as SPIRAL>::RingQFast, N_PLUS_ONE, M, Z_GSW, T_GSW>()
                    * &<Self as SPIRAL>::RingQFast::from(mu));
        c_mat
    }

    fn decode_gsw_scaled(
        s_mat: &<Self as SPIRAL>::MatrixKey,
        c: &<Self as SPIRAL>::GSWCiphertext,
        scale: &<Self as SPIRAL>::RingQFast,
    ) -> <Self as SPIRAL>::RingQ {
        let ident = Matrix::<N, N, <Self as SPIRAL>::RingQFast>::identity();
        let scaled_ident =
            &Matrix::<N_PLUS_ONE, N_PLUS_ONE, <Self as SPIRAL>::RingQFast>::identity() * scale;
        let s_t = Matrix::append(&(-s_mat), &ident);
        let result_q_fast_mat = &(&s_t * c)
            * &gadget_inverse::<<Self as SPIRAL>::RingQFast, N_PLUS_ONE, M, N_PLUS_ONE, Z_GSW, T_GSW>(
                &scaled_ident,
            );
        let result_q = <Self as SPIRAL>::RingQ::from(&result_q_fast_mat[(0, 1)]);
        <Self as SPIRAL>::RingQ::from(result_q)
    }

    fn regev_sub_hom(
        lhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
        rhs: &<Self as SPIRAL>::MatrixRegevCiphertext,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
        lhs - rhs
    }

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
        gsw * &gadget_inverse::<<Self as SPIRAL>::RingQFast, N_PLUS_ONE, M, N, Z_GSW, T_GSW>(regev)
    }

    fn decode_matrix_regev(
        s_mat: &<Self as SPIRAL>::MatrixKey,
        c: &<Self as SPIRAL>::MatrixRegevCiphertext,
    ) -> <Self as SPIRAL>::MatrixQ {
        (&Matrix::append(&-s_mat, &Matrix::<N, N, _>::identity()) * c)
            .into_ring(|x| <Self as SPIRAL>::RingQ::from(x))
    }

    fn auto_setup(
        tau_power: usize,
        s_scalar: &<Self as SPIRAL>::RingQFast,
    ) -> <Self as SPIRAL>::AutoKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, T_COEFF, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_t: Matrix<1, T_COEFF, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let mut bottom = &a_t * s_scalar;
        bottom += &e_t;
        bottom -= &(&build_gadget::<<Self as SPIRAL>::RingQFast, 1, T_COEFF, Z_COEFF, T_COEFF>()
            * &s_scalar.auto(tau_power));
        (Matrix::stack(&a_t, &bottom), tau_power)
    }

    fn auto_hom(
        (w_mat, tau_power): &<Self as SPIRAL>::AutoKey,
        c: &<Self as SPIRAL>::ScalarRegevCiphertext,
    ) -> <Self as SPIRAL>::ScalarRegevCiphertext {
        let c0 = &c[(0, 0)];
        let c1 = &c[(1, 0)];
        let mut tau_c0_mat: Matrix<1, 1, _> = Matrix::zero();
        tau_c0_mat[(0, 0)] = c0.auto(*tau_power);
        let g_inv_tau_c0 = gadget_inverse::<_, 1, T_COEFF, 1, Z_COEFF, T_COEFF>(&tau_c0_mat);
        let mut result = w_mat * &g_inv_tau_c0;
        result[(1, 0)] += &c1.auto(*tau_power);
        result
    }

    fn scal_to_mat_setup(
        s_scalar: &<Self as SPIRAL>::ScalarKey,
        s_matrix: &<Self as SPIRAL>::MatrixKey,
    ) -> <Self as SPIRAL>::ScalToMatKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M_CONV, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<N, M_CONV, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let mut bottom = s_matrix * &a_t;
        bottom += &e_mat;
        bottom -=
            &(&build_gadget::<<Self as SPIRAL>::RingQFast, N, M_CONV, Z_CONV, T_CONV>() * s_scalar);
        let mut result: Matrix<N_PLUS_ONE, M_CONV, <Self as SPIRAL>::RingQFast> = Matrix::zero();
        result.copy_into(&a_t, 0, 0);
        result.copy_into(&bottom, 1, 0);
        result
    }

    fn scal_to_mat(
        scal_to_mat_key: &<Self as SPIRAL>::ScalToMatKey,
        c: &<Self as SPIRAL>::ScalarRegevCiphertext,
    ) -> <Self as SPIRAL>::MatrixRegevCiphertext {
        let c0 = &c[(0, 0)];
        let c1 = &c[(1, 0)];

        let mut g_inv_c0_ident = Matrix::<M_CONV, N, <Self as SPIRAL>::RingQFast>::zero();
        <<Self as SPIRAL>::RingQFast as RingElementDecomposable<Z_CONV, T_CONV>>::decompose_into_mat(c0,&mut g_inv_c0_ident, 0, 0);
        for i in 1..N {
            for k in 0..T_CONV {
                g_inv_c0_ident[(i * T_CONV + k, i)] = g_inv_c0_ident[(k, 0)].clone();
            }
        }

        let mut result = scal_to_mat_key * &g_inv_c0_ident;
        for i in 0..N {
            result[(i + 1, i)] += &c1;
        }
        result
    }

    fn regev_to_gsw_setup(
        s_scalar: &<Self as SPIRAL>::ScalarKey,
        s_matrix: &<Self as SPIRAL>::MatrixKey,
    ) -> <Self as SPIRAL>::RegevToGSWKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t =
            Matrix::<1, T_CONV_TIMES_TWO, <Self as SPIRAL>::RingQFast>::rand_uniform(&mut rng);
        let e_mat =
            Matrix::<N, T_CONV_TIMES_TWO, <Self as SPIRAL>::RingQFast>::rand_discrete_gaussian::<
                _,
                NOISE_WIDTH_MILLIONTHS,
            >(&mut rng);
        let mut bottom = s_matrix * &a_t;
        bottom += &e_mat;
        let g_vec = build_gadget::<<Self as SPIRAL>::RingQFast, 1, T_CONV, Z_CONV, T_CONV>();
        let mut s_scalar_tensor_g =
            Matrix::<1, T_CONV_TIMES_TWO, <Self as SPIRAL>::RingQFast>::zero();
        s_scalar_tensor_g.copy_into(&g_vec, 0, T_CONV);
        s_scalar_tensor_g.copy_into(&(&g_vec * &(-s_scalar)), 0, 0);
        bottom -= &(s_matrix * &s_scalar_tensor_g);

        let result = Matrix::stack(&a_t, &bottom);
        let scal_to_mat_key = Self::scal_to_mat_setup(&s_scalar, &s_matrix);
        (result, scal_to_mat_key)
    }

    fn regev_to_gsw(
        (v_mat, scal_to_mat_key): &<Self as SPIRAL>::RegevToGSWKey,
        cs: &[<Self as SPIRAL>::ScalarRegevCiphertext],
    ) -> <Self as SPIRAL>::GSWCiphertext {
        let mut result = Matrix::<N_PLUS_ONE, M, <Self as SPIRAL>::RingQFast>::zero();
        let mut c_hat = Matrix::<2, T_GSW, <Self as SPIRAL>::RingQFast>::zero();
        for i in 0..T_GSW {
            c_hat.copy_into(&cs[i], 0, i);
        }
        let v_g_inv_c_hat = v_mat
            * &gadget_inverse::<
                <Self as SPIRAL>::RingQFast,
                2,
                T_CONV_TIMES_TWO,
                T_GSW,
                Z_CONV,
                T_CONV,
            >(&c_hat);
        result.copy_into(&v_g_inv_c_hat, 0, 0);
        for i in 0..T_GSW {
            let c_i_mat = Self::scal_to_mat(&scal_to_mat_key, &cs[i]);
            result.copy_into(&c_i_mat, 0, T_GSW + N * i);
        }

        let mut result_perm = Matrix::<N_PLUS_ONE, M, <Self as SPIRAL>::RingQFast>::zero();

        // Copy the leftmost T_GSW columns
        for c in 0..T_GSW {
            for r in 0..N_PLUS_ONE {
                result_perm[(r, c)] = result[(r, c)].clone();
            }
        }

        // Think of the columns of the bottom right minor of `result` as a N x T_GSW matrix.
        // We are transposing this matrix. The additional `T_GSW` is to shift to the minor.
        for i in 0..N {
            for j in 0..T_GSW {
                let from_c = N * j + i + T_GSW;
                let to_c = T_GSW * i + j + T_GSW;
                for r in 0..N_PLUS_ONE {
                    result_perm[(r, to_c)] = result[(r, from_c)].clone();
                }
            }
        }

        result_perm
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use crate::math::int_mod_poly::IntModPoly;
    use rand::Rng;

    use super::*;

    const SPIRAL_TEST_PARAMS: SPIRALParams = SPIRALParamsRaw {
        N: 2,
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        Z_GSW: 75,
        Z_COEFF: 128,
        Z_CONV: 16384,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 1 << 8,
        ETA1: 9,
        ETA2: 6,
        Z_FOLD: 2,
    }
    .expand();

    type SPIRALTest = spiral!(SPIRAL_TEST_PARAMS);

    #[test]
    fn test_scalar_regev() {
        let s = SPIRALTest::scalar_regev_setup();
        let mu = <SPIRALTest as SPIRAL>::RingP::from(12_u64);
        let encoded = SPIRALTest::encode_scalar_regev(&s, &mu.scale_up_into());
        let decoded: <SPIRALTest as SPIRAL>::RingP =
            SPIRALTest::decode_scalar_regev(&s, &encoded).round_down_into();
        assert_eq!(mu, decoded);
    }

    #[test]
    fn test_gsw() {
        let s = SPIRALTest::matrix_regev_setup();
        type RingPP = IntModCyclo<{ SPIRAL_TEST_PARAMS.D }, 1024>;
        let mu = RingPP::from(111_u64);
        let encrypt = SPIRALTest::encode_gsw(&s, &mu.include_into());

        let scale = <SPIRALTest as SPIRAL>::RingQFast::from(SPIRAL_TEST_PARAMS.Q / 1024);
        let decrypt = SPIRALTest::decode_gsw_scaled(&s, &encrypt, &scale);
        assert_eq!(decrypt.round_down_into(), mu);
    }

    #[test]
    fn test_auto_hom() {
        let s = SPIRALTest::scalar_regev_setup();
        let auto_key = SPIRALTest::auto_setup(3, &s);
        let x = <SPIRALTest as SPIRAL>::RingP::from(IntModPoly::x());
        let encrypt = SPIRALTest::encode_scalar_regev(&s, &x.scale_up_into());
        let encrypt_auto = SPIRALTest::auto_hom(&auto_key, &encrypt);
        let decrypt: <SPIRALTest as SPIRAL>::RingP =
            SPIRALTest::decode_scalar_regev(&s, &encrypt_auto).round_down_into();
        assert_eq!(decrypt, &(&x * &x) * &x);
    }

    #[test]
    fn test_scal_to_mat() {
        let s_scal = SPIRALTest::scalar_regev_setup();
        let s_mat = SPIRALTest::matrix_regev_setup();
        let s_scal_to_mat = SPIRALTest::scal_to_mat_setup(&s_scal, &s_mat);
        let mu = <SPIRALTest as SPIRAL>::RingP::from(1234_u64);
        let encrypt_scal = SPIRALTest::encode_scalar_regev(&s_scal, &mu.scale_up_into());
        let encrypt_mat = SPIRALTest::scal_to_mat(&s_scal_to_mat, &encrypt_scal);
        let decrypt = SPIRALTest::decode_matrix_regev(&s_mat, &encrypt_mat)
            .into_ring(|x| -> <SPIRALTest as SPIRAL>::RingP { x.round_down_into() });
        assert_eq!(
            decrypt,
            &Matrix::<
                { SPIRAL_TEST_PARAMS.N },
                { SPIRAL_TEST_PARAMS.N },
                <SPIRALTest as SPIRAL>::RingP,
            >::identity()
                * &<SPIRALTest as SPIRAL>::RingP::from(1234_u64)
        );
    }

    #[test]
    fn test_regev_to_gsw() {
        let s_scal = SPIRALTest::scalar_regev_setup();
        let s_mat = SPIRALTest::matrix_regev_setup();
        let s_regev_to_gsw = SPIRALTest::regev_to_gsw_setup(&s_scal, &s_mat);
        type RingPP = IntModCyclo<{ SPIRAL_TEST_PARAMS.D }, 1024>;
        let mu = RingPP::from(567_u64);
        let mut mu_curr = mu.include_into();
        let mut encrypt_vec = Vec::with_capacity(SPIRAL_TEST_PARAMS.T_GSW);
        for _ in 0..SPIRAL_TEST_PARAMS.T_GSW {
            encrypt_vec.push(SPIRALTest::encode_scalar_regev(&s_scal, &mu_curr));
            mu_curr *= IntMod::from(SPIRAL_TEST_PARAMS.Z_GSW);
        }
        let encrypt_gsw = SPIRALTest::regev_to_gsw(&s_regev_to_gsw, encrypt_vec.as_slice());

        let scale = <SPIRALTest as SPIRAL>::RingQFast::from(SPIRAL_TEST_PARAMS.Q / 1024);
        let decrypted = SPIRALTest::decode_gsw_scaled(&s_mat, &encrypt_gsw, &scale);
        assert_eq!(decrypted.round_down_into(), mu);
    }

    #[test]
    fn test_spiral_one() {
        run_spiral::<SPIRALTest, _>([11111].into_iter());
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
        let (qk, pp) = TheSPIRAL::setup();
        let setup_end = Instant::now();
        eprintln!("{:?} to setup", setup_end - setup_start);

        let check = |idx: usize| {
            eprintln!("Running with idx = {}", idx);
            let query_start = Instant::now();
            let q = TheSPIRAL::query(&qk, idx);
            let query_end = Instant::now();
            let query_total = query_end - query_start;

            let query_expand_start = Instant::now();
            let q_expanded = TheSPIRAL::query_expand(&pp, &q);
            let query_expand_end = Instant::now();
            let query_expand_total = query_expand_end - query_expand_start;

            let answer_start = Instant::now();
            let result = TheSPIRAL::answer(&db_pre, &q_expanded);
            let answer_end = Instant::now();
            let answer_total = answer_end - answer_start;

            let extract_start = Instant::now();
            let extracted = TheSPIRAL::extract(&qk, &result);
            let extract_end = Instant::now();
            let extract_total = extract_end - extract_start;

            if &extracted != &db[idx] {
                eprintln!("  **** **** **** **** ERROR **** **** **** ****");
                eprintln!("  protocol failed");
            }
            eprintln!(
                "  {:?} total",
                query_total + query_expand_total + answer_total + extract_total
            );
            eprintln!("    {:?} to query", query_total);
            eprintln!("    {:?} to query expand", query_expand_total);
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
