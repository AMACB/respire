use bitvec::prelude::*;
use itertools::Itertools;
use std::cmp::max;
use std::f64::consts::PI;
use std::time::Instant;

use crate::math::discrete_gaussian::NUM_WIDTHS;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::math::gadget::{
    base_from_len, build_gadget, gadget_inverse, gadget_inverse_scalar, RingElementDecomposable,
};
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::matrix::Matrix;
use crate::math::number_theory::mod_pow;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::ring_elem::{NormedRingElement, RingElement};
use crate::math::utils::{ceil_log, floor_log, mod_inverse, reverse_bits, reverse_bits_fast};

use crate::math::simd_utils::*;
use crate::pir::noise::{BoundedNoise, Independent, SubGaussianNoise};

pub struct RespireImpl<
    const Q: u64,
    const Q_A: u64,
    const Q_B: u64,
    const D: usize,
    const Z_GSW: u64,
    const T_GSW: usize,
    const Z_COEFF_REGEV: u64,
    const T_COEFF_REGEV: usize,
    const Z_COEFF_GSW: u64,
    const T_COEFF_GSW: usize,
    const Z_CONV: u64,
    const T_CONV: usize,
    const M_CONV: usize,
    const M_GSW: usize,
    const BATCH_SIZE: usize,
    const N_VEC: usize,
    const Z_SCAL_TO_VEC: u64,
    const T_SCAL_TO_VEC: usize,
    const NOISE_WIDTH_MILLIONTHS: u64,
    const P: u64,
    const D_RECORD: usize,
    const ETA1: usize,
    const ETA2: usize,
    const Z_FOLD: usize,
    const Q_SWITCH1: u64,
    const Q_SWITCH2: u64,
    const D_SWITCH: usize,
    const T_SWITCH: usize,
    const Z_SWITCH: u64,
    const BYTES_PER_RECORD: usize,
> {}

#[allow(non_snake_case)]
pub struct RespireParamsRaw {
    pub Q_A: u64,
    pub Q_B: u64,
    pub D: usize,
    pub T_GSW: usize,
    pub T_COEFF_REGEV: usize,
    pub T_COEFF_GSW: usize,
    pub T_CONV: usize,
    pub BATCH_SIZE: usize,
    pub N_VEC: usize,
    pub T_SCAL_TO_VEC: usize,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub D_RECORD: usize,
    pub ETA1: usize,
    pub ETA2: usize,
    pub Z_FOLD: usize,
    pub Q_SWITCH1: u64,
    pub Q_SWITCH2: u64,
    pub D_SWITCH: usize,
    pub T_SWITCH: usize,
}

impl RespireParamsRaw {
    pub const fn expand(&self) -> RespireParams {
        let q = self.Q_A * self.Q_B;
        let z_gsw = base_from_len(self.T_GSW, q);
        let z_coeff_regev = base_from_len(self.T_COEFF_REGEV, q);
        let z_coeff_gsw = base_from_len(self.T_COEFF_GSW, q);
        let z_conv = base_from_len(self.T_CONV, q);
        let z_scal_to_vec = base_from_len(self.T_SCAL_TO_VEC, q);
        let z_switch = base_from_len(self.T_SWITCH, self.Q_SWITCH2);
        RespireParams {
            Q: q,
            Q_A: self.Q_A,
            Q_B: self.Q_B,
            D: self.D,
            Z_GSW: z_gsw,
            T_GSW: self.T_GSW,
            Z_COEFF_REGEV: z_coeff_regev,
            T_COEFF_REGEV: self.T_COEFF_REGEV,
            Z_COEFF_GSW: z_coeff_gsw,
            T_COEFF_GSW: self.T_COEFF_GSW,
            Z_CONV: z_conv,
            T_CONV: self.T_CONV,
            BATCH_SIZE: self.BATCH_SIZE,
            N_VEC: self.N_VEC,
            T_SCAL_TO_VEC: self.T_SCAL_TO_VEC,
            Z_SCAL_TO_VEC: z_scal_to_vec,
            M_CONV: 2 * self.T_CONV,
            M_GSW: 2 * self.T_GSW,
            NOISE_WIDTH_MILLIONTHS: self.NOISE_WIDTH_MILLIONTHS,
            P: self.P,
            D_RECORD: self.D_RECORD,
            ETA1: self.ETA1,
            ETA2: self.ETA2,
            Z_FOLD: self.Z_FOLD,
            Q_SWITCH1: self.Q_SWITCH1,
            Q_SWITCH2: self.Q_SWITCH2,
            D_SWITCH: self.D_SWITCH,
            T_SWITCH: self.T_SWITCH,
            Z_SWITCH: z_switch,
            BYTES_PER_RECORD: (self.D_RECORD * floor_log(2, self.P)) / 8,
        }
    }
}

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct RespireParams {
    pub Q: u64,
    pub Q_A: u64,
    pub Q_B: u64,
    pub D: usize,
    pub Z_GSW: u64,
    pub T_GSW: usize,
    pub Z_COEFF_REGEV: u64,
    pub T_COEFF_REGEV: usize,
    pub Z_COEFF_GSW: u64,
    pub T_COEFF_GSW: usize,
    pub Z_CONV: u64,
    pub T_CONV: usize,
    pub M_CONV: usize,
    pub M_GSW: usize,
    pub BATCH_SIZE: usize,
    pub N_VEC: usize,
    pub Z_SCAL_TO_VEC: u64,
    pub T_SCAL_TO_VEC: usize,
    pub NOISE_WIDTH_MILLIONTHS: u64,
    pub P: u64,
    pub D_RECORD: usize,
    pub ETA1: usize,
    pub ETA2: usize,
    pub Z_FOLD: usize,
    pub Q_SWITCH1: u64,
    pub Q_SWITCH2: u64,
    pub D_SWITCH: usize,
    pub T_SWITCH: usize,
    pub Z_SWITCH: u64,
    pub BYTES_PER_RECORD: usize,
}

impl RespireParams {
    pub fn correctness_param(&self) -> f64 {
        // 2 d n^2 * exp(-pi * correctness^2) <= 2^(-40)
        (-1_f64 / PI * (2_f64.powi(-40) / 2_f64 / self.D as f64).ln()).sqrt()
    }

    pub fn relative_noise_threshold(&self) -> f64 {
        1_f64 / (2_f64 * self.P as f64) / self.correctness_param()
    }

    pub fn noise_estimate(&self) -> f64 {
        // TODO: this is very out of date
        let chi = SubGaussianNoise::new(
            (self.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64).powi(2),
            self.D as u64,
        );

        let chi_bounded = BoundedNoise::new(
            (self.NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64 * NUM_WIDTHS as f64).ceil(),
            self.D as u64,
        );

        let db_record_noise = BoundedNoise::new((self.P - 1) as f64, self.D as u64);

        let gadget_inverse_noise =
            |base: u64, len: usize, rows: usize, cols: usize| -> BoundedNoise {
                // Note: we use base / 4 since G inverse of uniform is uniform in [-z/2, z/2]. So the expected magnitude is z/4.
                BoundedNoise::new_matrix((base / 2) as f64 / 2_f64, self.D as u64, rows * len, cols)
            };

        let automorph_noise = |e: SubGaussianNoise, base: u64, len: usize| -> SubGaussianNoise {
            let e_t_g_inv = chi.with_dimension(1, len) * gadget_inverse_noise(base, len, 1, 1);
            e + e_t_g_inv
        };

        let expand_iter_noise = |e: SubGaussianNoise, base: u64, len: usize| -> SubGaussianNoise {
            e + automorph_noise(e, base, len)
        };

        let regev_to_gsw_noise = |e_t: SubGaussianNoise| -> SubGaussianNoise {
            let e_conv = chi.with_dimension(1, 2 * self.T_CONV);
            let g_inv_z_conv = gadget_inverse_noise(self.Z_CONV, self.T_CONV, 2, self.T_GSW);
            let s_gsw = chi_bounded.with_dimension(1, 1);
            let lhs = e_conv * g_inv_z_conv + s_gsw * e_t.with_dimension(1, self.T_GSW);

            let lhs_variance = lhs.variance();
            let rhs_variance = e_t.variance();
            // Average the noise of the left T_GSW and right T_GSW entries. Since this will get
            // multiplied by a BoundedNoise after this estimate is accurate.
            SubGaussianNoise::new((lhs_variance + rhs_variance) / 2_f64, self.D as u64)
                .with_dimension(1, 2 * self.T_GSW)
        };

        let regev_count = 1 << self.ETA1;
        let regev_expand_iters = self.ETA1;
        let gsw_count = self.ETA2 * (self.Z_FOLD - 1) * self.T_GSW;
        let gsw_expand_iters = ceil_log(2, gsw_count as u64);

        // Original query
        let query_noise = chi;

        // Query expansion
        let (regev_noise, gsw_noise) = {
            let mut regev_expand_noise =
                expand_iter_noise(query_noise, self.Z_COEFF_GSW, self.T_COEFF_GSW);
            let mut gsw_expand_noise = regev_expand_noise;
            for _ in 0..regev_expand_iters {
                regev_expand_noise =
                    expand_iter_noise(regev_expand_noise, self.Z_COEFF_REGEV, self.T_COEFF_REGEV);
            }
            for _ in 0..gsw_expand_iters {
                gsw_expand_noise =
                    expand_iter_noise(gsw_expand_noise, self.Z_COEFF_GSW, self.T_COEFF_GSW);
            }
            (regev_expand_noise, regev_to_gsw_noise(gsw_expand_noise))
        };

        // First dimension
        let first_dim_noise = (regev_noise * db_record_noise) * Independent(regev_count as f64);

        // Folding
        let mut fold_noise = first_dim_noise;
        for _ in 0..self.ETA2 {
            let ci_minus_c0_noise = gsw_noise * gadget_inverse_noise(self.Z_GSW, self.T_GSW, 2, 1);
            // Second fold_noise term is for E_regev
            fold_noise =
                fold_noise + ci_minus_c0_noise * Independent((self.Z_FOLD - 1) as f64) + fold_noise;
        }

        fold_noise.variance().sqrt() / self.Q as f64
    }

    pub fn public_param_size(&self) -> usize {
        let automorph_elems = floor_log(2, self.D as u64) * (self.T_COEFF_REGEV + self.T_COEFF_GSW);
        let reg_to_gsw_elems = 2 * self.T_CONV;
        let scal_to_vec_elems = self.N_VEC * self.T_SCAL_TO_VEC;
        let q_elem_size = self.D * ceil_log(2, self.Q) / 8;

        let compress_elems = self.N_VEC * self.T_SWITCH;
        let q2_elem_size = self.D * ceil_log(2, self.Q_SWITCH2) / 8;
        return (automorph_elems + reg_to_gsw_elems + scal_to_vec_elems) * q_elem_size
            + compress_elems * q2_elem_size;
    }

    pub fn query_size(&self) -> usize {
        let n_regev = 1usize << self.ETA1;
        let n_gsw = self.T_GSW * (self.ETA2 + (self.D / self.D_RECORD));
        (n_regev + n_gsw) * ceil_log(2, self.Q) / 8
    }

    pub fn record_size(&self) -> usize {
        let log_p = floor_log(2, self.P);
        self.BATCH_SIZE * self.D_RECORD * log_p / 8
    }
    pub fn response_size(&self) -> usize {
        let num_elems = self
            .BATCH_SIZE
            .div_ceil(self.N_VEC * (self.D_SWITCH / self.D_RECORD));
        let log_q1 = (self.Q_SWITCH1 as f64).log2();
        let log_q2 = (self.Q_SWITCH2 as f64).log2();
        let one_elem = ((self.D_SWITCH as f64) * (log_q2 + (self.N_VEC as f64) * log_q1) / 8_f64)
            .ceil() as usize;
        num_elems * one_elem
    }

    pub fn rate(&self) -> f64 {
        (self.record_size() as f64) / (self.response_size() as f64)
    }
}

#[macro_export]
macro_rules! respire {
    ($params: expr) => {
        RespireImpl<
            {$params.Q},
            {$params.Q_A},
            {$params.Q_B},
            {$params.D},
            {$params.Z_GSW},
            {$params.T_GSW},
            {$params.Z_COEFF_REGEV},
            {$params.T_COEFF_REGEV},
            {$params.Z_COEFF_GSW},
            {$params.T_COEFF_GSW},
            {$params.Z_CONV},
            {$params.T_CONV},
            {$params.M_CONV},
            {$params.M_GSW},
            {$params.BATCH_SIZE},
            {$params.N_VEC},
            {$params.Z_SCAL_TO_VEC},
            {$params.T_SCAL_TO_VEC},
            {$params.NOISE_WIDTH_MILLIONTHS},
            {$params.P},
            {$params.D_RECORD},
            {$params.ETA1},
            {$params.ETA2},
            {$params.Z_FOLD},
            {$params.Q_SWITCH1},
            {$params.Q_SWITCH2},
            {$params.D_SWITCH},
            {$params.T_SWITCH},
            {$params.Z_SWITCH},
            {$params.BYTES_PER_RECORD},
        >
    }
}

macro_rules! respire_impl {
    ($impl_for: ident, $body: tt) => {
        impl<
                const Q: u64,
                const Q_A: u64,
                const Q_B: u64,
                const D: usize,
                const Z_GSW: u64,
                const T_GSW: usize,
                const Z_COEFF_REGEV: u64,
                const T_COEFF_REGEV: usize,
                const Z_COEFF_GSW: u64,
                const T_COEFF_GSW: usize,
                const Z_CONV: u64,
                const T_CONV: usize,
                const M_CONV: usize,
                const M_GSW: usize,
                const BATCH_SIZE: usize,
                const N_VEC: usize,
                const Z_SCAL_TO_VEC: u64,
                const T_SCAL_TO_VEC: usize,
                const NOISE_WIDTH_MILLIONTHS: u64,
                const P: u64,
                const D_RECORD: usize,
                const ETA1: usize,
                const ETA2: usize,
                const Z_FOLD: usize,
                const Q_SWITCH1: u64,
                const Q_SWITCH2: u64,
                const D_SWITCH: usize,
                const T_SWITCH: usize,
                const Z_SWITCH: u64,
                const BYTES_PER_RECORD: usize,
            > $impl_for
            for RespireImpl<
                Q,
                Q_A,
                Q_B,
                D,
                Z_GSW,
                T_GSW,
                Z_COEFF_REGEV,
                T_COEFF_REGEV,
                Z_COEFF_GSW,
                T_COEFF_GSW,
                Z_CONV,
                T_CONV,
                M_CONV,
                M_GSW,
                BATCH_SIZE,
                N_VEC,
                Z_SCAL_TO_VEC,
                T_SCAL_TO_VEC,
                NOISE_WIDTH_MILLIONTHS,
                P,
                D_RECORD,
                ETA1,
                ETA2,
                Z_FOLD,
                Q_SWITCH1,
                Q_SWITCH2,
                D_SWITCH,
                T_SWITCH,
                Z_SWITCH,
                BYTES_PER_RECORD,
            >
        $body
    };
    ($body: tt) => {
        impl<
                const Q: u64,
                const Q_A: u64,
                const Q_B: u64,
                const D: usize,
                const Z_GSW: u64,
                const T_GSW: usize,
                const Z_COEFF_REGEV: u64,
                const T_COEFF_REGEV: usize,
                const Z_COEFF_GSW: u64,
                const T_COEFF_GSW: usize,
                const Z_CONV: u64,
                const T_CONV: usize,
                const M_CONV: usize,
                const M_GSW: usize,
                const BATCH_SIZE: usize,
                const N_VEC: usize,
                const Z_SCAL_TO_VEC: u64,
                const T_SCAL_TO_VEC: usize,
                const NOISE_WIDTH_MILLIONTHS: u64,
                const P: u64,
                const D_RECORD: usize,
                const ETA1: usize,
                const ETA2: usize,
                const Z_FOLD: usize,
                const Q_SWITCH1: u64,
                const Q_SWITCH2: u64,
                const D_SWITCH: usize,
                const T_SWITCH: usize,
                const Z_SWITCH: u64,
                const BYTES_PER_RECORD: usize,
            > RespireImpl<
                Q,
                Q_A,
                Q_B,
                D,
                Z_GSW,
                T_GSW,
                Z_COEFF_REGEV,
                T_COEFF_REGEV,
                Z_COEFF_GSW,
                T_COEFF_GSW,
                Z_CONV,
                T_CONV,
                M_CONV,
                M_GSW,
                BATCH_SIZE,
                N_VEC,
                Z_SCAL_TO_VEC,
                T_SCAL_TO_VEC,
                NOISE_WIDTH_MILLIONTHS,
                P,
                D_RECORD,
                ETA1,
                ETA2,
                Z_FOLD,
                Q_SWITCH1,
                Q_SWITCH2,
                D_SWITCH,
                T_SWITCH,
                Z_SWITCH,
                BYTES_PER_RECORD,
            >
        $body
    };
}

pub trait Respire: PIR {
    // Type aliases
    type RingP;
    type RingQ;
    type RingQFast;
    type RegevCiphertext;
    type RegevSeeded;
    type RegevCompressed;
    type GSWCiphertext;
    type EncodingKey;
    type VecEncodingKey;
    type VecEncodingKeyQ2;
    type VecEncodingKeyQ2Small;
    type AutoKey<const T: usize>;
    type AutoKeyRegev;
    type AutoKeyGSW;
    type RegevToGSWKey;
    type KeySwitchKey;
    type ScalToVecKey;
    type VecRegevCiphertext;
    type VecRegevSmall;

    // A single record
    type Record;
    // Packed records from a single response, after compression
    type RecordPackedSmall;
    // Packed records from a single response, before compression
    type RecordPacked;
    type QueryOne;
    type QueryOneExpanded;
    type ResponseOne;
    type ResponseOneCompressed;

    // Constants
    const PACKED_DIM1_SIZE: usize;
    const PACKED_DIM2_SIZE: usize;
    const PACKED_DB_SIZE: usize;
    const DB_SIZE: usize;
    const PACK_RATIO_DB: usize;
    const PACK_RATIO_RESPONSE: usize;
    const ETA1: usize;
    const ETA2: usize;
    const REGEV_COUNT: usize;
    const REGEV_EXPAND_ITERS: usize;
    const GSW_FOLD_COUNT: usize;
    const GSW_PROJ_COUNT: usize;
    const GSW_COUNT: usize;
    const GSW_EXPAND_ITERS: usize;

    fn query_one(qk: &<Self as PIR>::QueryKey, idx: usize) -> <Self as Respire>::QueryOne;
    fn answer_one(
        pp: &<Self as PIR>::PublicParams,
        db: &<Self as PIR>::Database,
        q: &<Self as Respire>::QueryOne,
    ) -> <Self as Respire>::ResponseOne;
    fn answer_compress_chunk(
        pp: &<Self as PIR>::PublicParams,
        chunk: &[<Self as Respire>::ResponseOne],
        qk: Option<&<Self as PIR>::QueryKey>,
    ) -> <Self as Respire>::ResponseOneCompressed;
    fn answer_compress_vec(
        pp: &<Self as PIR>::PublicParams,
        vec: &<Self as Respire>::VecRegevCiphertext,
    ) -> <Self as Respire>::ResponseOneCompressed;
    fn extract_one(
        qk: &<Self as PIR>::QueryKey,
        r: &<Self as Respire>::ResponseOneCompressed,
    ) -> Vec<<Self as PIR>::RecordBytes>;
}

pub trait PIR {
    // Associated types
    type QueryKey;
    type PublicParams;
    type Query;
    type Response;
    type Database;

    // A single raw record
    type RecordBytes: Clone + Default;

    const NUM_RECORDS: usize;
    const BATCH_SIZE: usize;

    fn encode_db<I: ExactSizeIterator<Item = Self::RecordBytes>>(records_iter: I)
        -> Self::Database;
    fn setup() -> (Self::QueryKey, Self::PublicParams);
    fn query(qk: &Self::QueryKey, idx: &[usize]) -> Self::Query;
    fn answer(
        pp: &Self::PublicParams,
        db: &Self::Database,
        q: &Self::Query,
        qk: Option<&Self::QueryKey>,
    ) -> Self::Response;
    fn extract(qk: &Self::QueryKey, r: &Self::Response) -> Vec<Self::RecordBytes>;
}

#[repr(transparent)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordBytesImpl<const LEN: usize> {
    pub it: [u8; LEN],
}

impl<const LEN: usize> Default for RecordBytesImpl<LEN> {
    fn default() -> Self {
        Self { it: [0u8; LEN] }
    }
}

respire_impl!(PIR, {
    // Associated types
    type QueryKey = (
        <Self as Respire>::EncodingKey,
        <Self as Respire>::VecEncodingKey,
        <Self as Respire>::VecEncodingKeyQ2Small,
    );
    type PublicParams = (
        (
            Vec<<Self as Respire>::AutoKeyRegev>,
            Vec<<Self as Respire>::AutoKeyGSW>,
        ),
        <Self as Respire>::RegevToGSWKey,
        <Self as Respire>::KeySwitchKey,
        <Self as Respire>::ScalToVecKey,
    );

    type Query = Vec<<Self as Respire>::QueryOne>;
    type Response = Vec<<Self as Respire>::ResponseOneCompressed>;

    /// We structure the database as `[2] x [D / S] x [DIM2_SIZE] x [DIM1_SIZE] x [S]` for optimal first dimension
    /// processing. The outermost pair is the first resp. second CRT projections, packed as two u32 into one u64;
    /// `S` is the SIMD lane count that we can use, i.e. 4 for AVX2.
    type Database = Vec<SimdVec>;

    // Public types & constants
    type RecordBytes = RecordBytesImpl<BYTES_PER_RECORD>;
    const NUM_RECORDS: usize = Self::DB_SIZE;
    const BATCH_SIZE: usize = BATCH_SIZE;

    fn encode_db<I: ExactSizeIterator<Item = Self::RecordBytes>>(
        records_iter: I,
    ) -> Self::Database {
        assert_eq!(records_iter.len(), Self::DB_SIZE);
        let records_encoded_iter = records_iter.map(|r| Self::encode_record(&r));

        let proj_inv =
            IntMod::<P>::from(mod_pow(mod_inverse(2, P), Self::GSW_PROJ_COUNT as u64, P));
        let records_eval: Vec<<Self as Respire>::RingQFast> = records_encoded_iter
            .chunks(Self::PACK_RATIO_DB)
            .into_iter()
            .map(|chunk| {
                let mut record_packed = IntModCyclo::<D, P>::zero();
                for (record_in_chunk, record) in chunk.enumerate() {
                    // Transpose so projection is more significant
                    let packed_offset = reverse_bits(Self::PACK_RATIO_DB, record_in_chunk);
                    for (coeff_idx, coeff) in record.coeff.iter().enumerate() {
                        record_packed.coeff[Self::PACK_RATIO_DB * coeff_idx + packed_offset] =
                            *coeff;
                    }
                }
                record_packed *= proj_inv;
                <Self as Respire>::RingQFast::from(&record_packed.include_into::<Q>())
            })
            .collect();

        #[cfg(not(target_feature = "avx2"))]
        {
            let mut db: Vec<SimdVec> = (0..(D * Self::PACKED_DB_SIZE)).map(|_| 0_u64).collect();
            for db_idx in 0..Self::PACKED_DB_SIZE {
                for eval_vec_idx in 0..D {
                    // Transpose the index
                    let (db_i, db_j) = (
                        db_idx / Self::PACKED_DIM2_SIZE,
                        db_idx % Self::PACKED_DIM2_SIZE,
                    );
                    let db_idx_t = db_j * Self::PACKED_DIM1_SIZE + db_i;

                    let to_idx = eval_vec_idx * Self::PACKED_DB_SIZE + db_idx_t;
                    let from_idx = eval_vec_idx;
                    let lo = u64::from(records_eval[db_idx].proj1.evals[from_idx]);
                    let hi = u64::from(records_eval[db_idx].proj2.evals[from_idx]);
                    db[to_idx] = (hi << 32) | lo;
                }
            }

            db
        }

        #[cfg(target_feature = "avx2")]
        {
            let mut db: Vec<SimdVec> = (0..((D / SIMD_LANES) * Self::PACKED_DB_SIZE))
                .map(|_| Aligned32([0_u64; 4]))
                .collect();

            for db_idx in 0..Self::PACKED_DB_SIZE {
                for eval_vec_idx in 0..(D / SIMD_LANES) {
                    // Transpose the index
                    let (db_i, db_j) = (
                        db_idx / Self::PACKED_DIM2_SIZE,
                        db_idx % Self::PACKED_DIM2_SIZE,
                    );
                    let db_idx_t = db_j * Self::PACKED_DIM1_SIZE + db_i;

                    let mut db_vec: SimdVec = Aligned32([0_u64; 4]);
                    for lane in 0..SIMD_LANES {
                        let from_idx = eval_vec_idx * SIMD_LANES + lane;
                        let lo = u64::from(records_eval[db_idx].proj1.evals[from_idx]);
                        let hi = u64::from(records_eval[db_idx].proj2.evals[from_idx]);
                        db_vec.0[lane] = (hi << 32) | lo;
                    }

                    let to_idx = eval_vec_idx * Self::PACKED_DB_SIZE + db_idx_t;
                    db[to_idx] = db_vec;
                }
            }

            db
        }
    }

    fn setup() -> (<Self as PIR>::QueryKey, <Self as PIR>::PublicParams) {
        // Regev/GSW secret key
        let s_encode = Self::encode_setup();

        // Vector regev secret key
        let s_vec: <Self as Respire>::VecEncodingKey = Self::encode_vec_setup();

        // Small ring key
        let s_small: <Self as Respire>::VecEncodingKeyQ2Small = {
            let mut rng = ChaCha20Rng::from_entropy();
            let mut result = Matrix::zero();
            for i in 0..N_VEC {
                result[(i, 0)] =
                    IntModCycloEval::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
            }
            result
        };

        // Key switching key
        let s_vec_q2 = s_vec.map_ring(|r| {
            IntModCycloEval::from(IntModCyclo::from(
                IntModCyclo::<D, Q>::from(r)
                    .coeff
                    .map(|x| IntMod::from(i64::from(x))),
            ))
        });
        let s_small_q2 =
            s_small.map_ring(|r| IntModCycloEval::from(IntModCyclo::from(r).include_dim()));
        let s_switch = Self::key_switch_setup(&s_vec_q2, &s_small_q2);

        // Automorphism keys
        let mut auto_keys_regev: Vec<<Self as Respire>::AutoKeyRegev> =
            Vec::with_capacity(Self::REGEV_EXPAND_ITERS);
        for i in 0..floor_log(2, D as u64) {
            let tau_power = (D >> i) + 1;
            auto_keys_regev.push(Self::auto_setup::<T_COEFF_REGEV, Z_COEFF_REGEV>(
                tau_power, &s_encode,
            ));
        }
        let mut auto_keys_gsw: Vec<<Self as Respire>::AutoKeyGSW> =
            Vec::with_capacity(Self::GSW_EXPAND_ITERS);
        for i in 0..floor_log(2, D as u64) {
            let tau_power = (D >> i) + 1;
            auto_keys_gsw.push(Self::auto_setup::<T_COEFF_GSW, Z_COEFF_GSW>(
                tau_power, &s_encode,
            ));
        }

        // Regev to GSW key
        let regev_to_gsw_key = Self::regev_to_gsw_setup(&s_encode);

        // Scalar to vector key
        let scal_to_vec_key = Self::scal_to_vec_setup(&s_encode, &s_vec);

        (
            (s_encode, s_vec, s_small),
            (
                (auto_keys_regev, auto_keys_gsw),
                regev_to_gsw_key,
                s_switch,
                scal_to_vec_key,
            ),
        )
    }

    fn query(qk: &<Self as PIR>::QueryKey, indices: &[usize]) -> <Self as PIR>::Query {
        assert_eq!(indices.len(), Self::BATCH_SIZE);
        indices
            .iter()
            .copied()
            .map(|idx| Self::query_one(qk, idx))
            .collect_vec()
    }

    fn answer(
        pp: &<Self as PIR>::PublicParams,
        db: &<Self as PIR>::Database,
        qs: &<Self as PIR>::Query,
        qk: Option<&<Self as PIR>::QueryKey>,
    ) -> <Self as PIR>::Response {
        assert_eq!(qs.len(), Self::BATCH_SIZE);
        let answers = qs.iter().map(|q| Self::answer_one(pp, db, q)).collect_vec();
        let answers_compressed = answers
            .chunks(N_VEC * Self::PACK_RATIO_RESPONSE)
            .map(|chunk| Self::answer_compress_chunk(pp, chunk, qk))
            .collect_vec();
        answers_compressed
    }

    fn extract(qk: &Self::QueryKey, r: &Self::Response) -> Vec<Self::RecordBytes> {
        let mut result = Vec::with_capacity(Self::BATCH_SIZE);
        for r_one in r {
            let extracted = Self::extract_one(qk, r_one);
            for record in extracted {
                if result.len() < Self::BATCH_SIZE {
                    result.push(record);
                }
            }
        }
        result
    }
});

respire_impl!(Respire, {
    type RingP = IntModCyclo<D, P>;
    type RingQ = IntModCyclo<D, Q>;
    type RingQFast = IntModCycloCRTEval<D, Q_A, Q_B>;
    type RegevCiphertext = Matrix<2, 1, Self::RingQFast>;
    type RegevSeeded = ([u8; 32], Self::RingQFast);
    type RegevCompressed = ([u8; 32], Vec<IntMod<Q>>);
    type GSWCiphertext = Matrix<2, M_GSW, Self::RingQFast>;

    type EncodingKey = Self::RingQFast;
    type VecEncodingKey = Matrix<N_VEC, 1, Self::RingQFast>;
    type VecEncodingKeyQ2 = Matrix<N_VEC, 1, IntModCycloEval<D, Q_SWITCH2>>;
    type VecEncodingKeyQ2Small = Matrix<N_VEC, 1, IntModCycloEval<D_SWITCH, Q_SWITCH2>>;
    type AutoKey<const T: usize> = (Matrix<2, T, Self::RingQFast>, usize);
    type AutoKeyRegev = Self::AutoKey<T_COEFF_REGEV>;
    type AutoKeyGSW = Self::AutoKey<T_COEFF_GSW>;
    type RegevToGSWKey = Matrix<2, M_CONV, Self::RingQFast>;
    type KeySwitchKey = (
        Matrix<1, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>,
        Matrix<N_VEC, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>,
    );
    type ScalToVecKey = Vec<(
        Matrix<1, T_SCAL_TO_VEC, Self::RingQFast>,
        Matrix<N_VEC, T_SCAL_TO_VEC, Self::RingQFast>,
    )>;
    type VecRegevCiphertext = (Self::RingQFast, Matrix<N_VEC, 1, Self::RingQFast>);
    type VecRegevSmall = (
        IntModCyclo<D_SWITCH, Q_SWITCH2>,
        Matrix<N_VEC, 1, IntModCyclo<D_SWITCH, Q_SWITCH1>>,
    );

    type Record = IntModCyclo<D_RECORD, P>;
    type RecordPackedSmall = Matrix<N_VEC, 1, IntModCyclo<D_SWITCH, P>>;
    type RecordPacked = IntModCyclo<D, P>;
    type QueryOne = (
        <Self as Respire>::RegevCompressed,
        <Self as Respire>::RegevCompressed,
    );
    type QueryOneExpanded = (
        Vec<<Self as Respire>::RegevCiphertext>,
        Vec<<Self as Respire>::GSWCiphertext>,
        Vec<<Self as Respire>::GSWCiphertext>,
    );
    type ResponseOne = <Self as Respire>::RegevCiphertext;
    type ResponseOneCompressed = <Self as Respire>::VecRegevSmall;

    const PACKED_DIM1_SIZE: usize = 2_usize.pow(ETA1 as u32);
    const PACKED_DIM2_SIZE: usize = Z_FOLD.pow(ETA2 as u32);
    const PACKED_DB_SIZE: usize = Self::PACKED_DIM1_SIZE * Self::PACKED_DIM2_SIZE;
    const DB_SIZE: usize = Self::PACKED_DB_SIZE * Self::PACK_RATIO_DB;
    const PACK_RATIO_DB: usize = D / D_RECORD;
    const PACK_RATIO_RESPONSE: usize = D_SWITCH / D_RECORD;
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;

    const REGEV_COUNT: usize = 1 << ETA1;
    const REGEV_EXPAND_ITERS: usize = ETA1;
    const GSW_FOLD_COUNT: usize = ETA2 * (Z_FOLD - 1);

    // TODO add param to reduce this when we don't care about garbage being in the other slots
    const GSW_PROJ_COUNT: usize = floor_log(2, Self::PACK_RATIO_DB as u64);
    const GSW_COUNT: usize = (Self::GSW_FOLD_COUNT + Self::GSW_PROJ_COUNT) * T_GSW;
    const GSW_EXPAND_ITERS: usize = ceil_log(2, Self::GSW_COUNT as u64);

    fn query_one(
        (s_encode, _, _): &<Self as PIR>::QueryKey,
        idx: usize,
    ) -> <Self as Respire>::QueryOne {
        assert!(idx < Self::DB_SIZE);
        let (idx, proj_idx) = (idx / Self::PACK_RATIO_DB, idx % Self::PACK_RATIO_DB);
        let (idx_i, idx_j) = (idx / Self::PACKED_DIM2_SIZE, idx % Self::PACKED_DIM2_SIZE);

        let mut mu_regev = <Self as Respire>::RingQ::zero();
        for i in 0..Self::REGEV_COUNT {
            mu_regev.coeff[reverse_bits_fast::<D>(i)] =
                IntMod::<P>::from((i == idx_i) as u64).scale_up_into();
        }

        // Think of these entries as [ETA2] x [Z_FOLD - 1] x [T_GSW] + [GSW_PROJ_COUNT] x [T_GSW]
        let mut mu_gsw = <Self as Respire>::RingQ::zero();

        // [ETA2] x [Z_FOLD - 1] x [T_GSW] part
        let mut digits = Vec::with_capacity(ETA2);
        let mut idx_j_curr = idx_j;
        for _ in 0..ETA2 {
            digits.push(idx_j_curr % Z_FOLD);
            idx_j_curr /= Z_FOLD;
        }

        for (digit_idx, digit) in digits.into_iter().rev().enumerate() {
            for which in 0..Z_FOLD - 1 {
                let mut msg = IntMod::from((digit == which + 1) as u64);
                for gsw_pow in 0..T_GSW {
                    let pack_idx = T_GSW * ((Z_FOLD - 1) * digit_idx + which) + gsw_pow;
                    mu_gsw.coeff[reverse_bits_fast::<D>(pack_idx)] = msg;
                    msg *= IntMod::from(Z_GSW);
                }
            }
        }

        // [GSW_PROJ_COUNT] x [T_GSW] part
        let mut proj_bits = Vec::with_capacity(Self::GSW_PROJ_COUNT);
        let mut proj_idx_curr = proj_idx;
        for _ in 0..Self::GSW_PROJ_COUNT {
            proj_bits.push(proj_idx_curr % 2);
            proj_idx_curr /= 2;
        }

        let gsw_proj_offset = ETA2 * (Z_FOLD - 1) * T_GSW;
        for (proj_idx, proj_bit) in proj_bits.into_iter().rev().enumerate() {
            let mut msg = IntMod::from(proj_bit as u64);
            for gsw_pow in 0..T_GSW {
                let pack_idx = gsw_proj_offset + T_GSW * proj_idx + gsw_pow;
                mu_gsw.coeff[reverse_bits_fast::<D>(pack_idx)] = msg;
                msg *= IntMod::from(Z_GSW);
            }
        }

        let (seed_regev, ct1_regev) = Self::encode_regev_seeded(s_encode, &mu_regev);
        let ct1_regev_coeff = <Self as Respire>::RingQ::from(&ct1_regev).coeff;
        let (seed_gsw, ct1_gsw) = Self::encode_regev_seeded(s_encode, &mu_gsw);
        let ct1_gsw_coeff = <Self as Respire>::RingQ::from(&ct1_gsw).coeff;
        let compressed_regev = (
            seed_regev,
            (0..Self::REGEV_COUNT)
                .map(|i| ct1_regev_coeff[reverse_bits_fast::<D>(i)])
                .collect_vec(),
        );
        let compressed_gsw = (
            seed_gsw,
            (0..Self::GSW_COUNT)
                .map(|i| ct1_gsw_coeff[reverse_bits_fast::<D>(i)])
                .collect_vec(),
        );
        (compressed_regev, compressed_gsw)
    }

    fn answer_one(
        pp: &<Self as PIR>::PublicParams,
        db: &<Self as PIR>::Database,
        q: &<Self as Respire>::QueryOne,
    ) -> <Self as Respire>::ResponseOne {
        let i0 = Instant::now();
        // Query expansion
        let (regevs, gsws_fold, gsws_proj) = Self::answer_query_expand(pp, q);
        let i1 = Instant::now();

        // First dimension
        let first_dim_folded = Self::answer_first_dim(db, &regevs);
        let i2 = Instant::now();

        // Folding
        let result = Self::answer_fold(first_dim_folded, gsws_fold.as_slice());
        let i3 = Instant::now();

        // Projecting
        let result_projected = Self::answer_project(pp, result, gsws_proj.as_slice());
        let i4 = Instant::now();

        eprintln!("(*) answer query expand: {:?}", i1 - i0);
        eprintln!("(*) answer first dim: {:?}", i2 - i1);
        eprintln!("(*) answer fold: {:?}", i3 - i2);
        eprintln!("(*) answer project: {:?}", i4 - i3);

        result_projected
    }

    fn answer_compress_chunk(
        pp: &<Self as PIR>::PublicParams,
        chunk: &[<Self as Respire>::ResponseOne],
        qk: Option<&<Self as PIR>::QueryKey>,
    ) -> <Self as Respire>::ResponseOneCompressed {
        let mut scalar_cts = Vec::with_capacity(N_VEC * Self::PACK_RATIO_RESPONSE);
        let (_, _, _, scal_to_vec_key) = pp;
        for vec_idx in 0..N_VEC {
            let mut scalar_ct = Matrix::zero();
            for pack_idx in 0..Self::PACK_RATIO_RESPONSE {
                let idx = vec_idx * Self::PACK_RATIO_RESPONSE + pack_idx;
                if idx < chunk.len() {
                    scalar_ct += &Self::regev_mul_x_pow(&chunk[idx], pack_idx * (D / D_SWITCH));
                }
            }
            scalar_cts.push(scalar_ct)
        }

        let ii0 = Instant::now();
        let vec = Self::scal_to_vec(scal_to_vec_key, scalar_cts.as_slice().try_into().unwrap());
        let ii1 = Instant::now();
        eprintln!("(**) answer scal to vec: {:?}", ii1 - ii0);

        if let Some((_, s_vec, _)) = qk {
            eprintln!(
                "  pre compression noise (subgaussian widths): 2^({})",
                Self::noise_subgaussian_bits_vec(s_vec, &vec)
            );
        }
        let compressed = Self::answer_compress_vec(pp, &vec);
        compressed
    }

    fn answer_compress_vec(
        (_, _, (a_t, b_mat), _): &<Self as PIR>::PublicParams,
        (c_r, c_m): &<Self as Respire>::VecRegevCiphertext,
    ) -> <Self as Respire>::ResponseOneCompressed {
        let c_r = IntModCyclo::<D, Q>::from(c_r);
        let c_m = c_m.map_ring(|r| IntModCyclo::<D, Q>::from(r));
        let mut cr_scaled = IntModCyclo::zero();
        for (cr_scaled_coeff, c0_coeff) in cr_scaled.coeff.iter_mut().zip(c_r.coeff) {
            let numer = Q_SWITCH2 as u128 * u64::from(c0_coeff) as u128;
            let denom = Q as u128;
            let div = (numer + denom / 2) / denom;
            *cr_scaled_coeff = IntMod::from(div as u64);
        }
        let g_inv_cr_scaled = gadget_inverse_scalar::<_, Z_SWITCH, T_SWITCH>(&cr_scaled)
            .map_ring(|x| IntModCycloEval::from(x));
        let c_r_hat: IntModCyclo<D_SWITCH, Q_SWITCH2> =
            IntModCyclo::from(&(a_t * &g_inv_cr_scaled)[(0, 0)]).project_dim();
        let c_m_hat: Matrix<N_VEC, 1, IntModCyclo<D_SWITCH, Q_SWITCH1>> = {
            let b_g_inv = (b_mat * &g_inv_cr_scaled).map_ring(|r| IntModCyclo::from(r));
            let mut result = Matrix::<N_VEC, 1, IntModCyclo<D, Q_SWITCH1>>::zero();
            for i in 0..N_VEC {
                for (result_coeff, (c1_coeff, b_t_g_inv_coeff)) in result[(i, 0)]
                    .coeff
                    .iter_mut()
                    .zip(c_m[(i, 0)].coeff.iter().copied().zip(b_g_inv[(i, 0)].coeff))
                {
                    let numer = Q_SWITCH1 as u128 * Q_SWITCH2 as u128 * u64::from(c1_coeff) as u128
                        + Q as u128 * Q_SWITCH1 as u128 * u64::from(b_t_g_inv_coeff) as u128;
                    let denom = Q as u128 * Q_SWITCH2 as u128;
                    let div = (numer + denom / 2) / denom;
                    *result_coeff = IntMod::from(div as u64);
                }
            }
            result.map_ring(|x| x.project_dim())
        };
        (c_r_hat, c_m_hat)
    }

    fn extract_one(
        qk: &<Self as PIR>::QueryKey,
        r: &<Self as Respire>::ResponseOneCompressed,
    ) -> Vec<<Self as PIR>::RecordBytes> {
        Self::extract_bytes_one(&Self::extract_ring_one(qk, r))
    }
});

respire_impl!({
    pub fn extract_ring_one(
        (_, _, s_small): &<Self as PIR>::QueryKey,
        (c_r_hat, c_m_hat): &<Self as Respire>::ResponseOneCompressed,
    ) -> <Self as Respire>::RecordPackedSmall {
        let neg_s_small_cr =
            (-&(s_small * &IntModCycloEval::from(c_r_hat))).map_ring(|r| IntModCyclo::from(r));
        let mut result = Matrix::<N_VEC, 1, IntModCyclo<D_SWITCH, Q_SWITCH1>>::zero();
        for i in 0..N_VEC {
            for (result_coeff, neg_s_small_c0_coeff) in result[(i, 0)]
                .coeff
                .iter_mut()
                .zip(neg_s_small_cr[(i, 0)].coeff)
            {
                let numer = Q_SWITCH1 as u128 * u64::from(neg_s_small_c0_coeff) as u128;
                let denom = Q_SWITCH2 as u128;
                let div = (numer + denom / 2) / denom;
                *result_coeff = IntMod::from(div as u64);
            }
            result[(i, 0)] += &c_m_hat[(i, 0)];
        }
        result.map_ring(|r| r.round_down_into())
    }

    pub fn extract_bytes_one(
        r: &<Self as Respire>::RecordPackedSmall,
    ) -> Vec<<Self as PIR>::RecordBytes> {
        let mut result = Vec::with_capacity(N_VEC * Self::PACK_RATIO_RESPONSE);
        for i in 0..N_VEC {
            for j in 0..Self::PACK_RATIO_RESPONSE {
                let record_coeffs: [IntMod<P>; D_RECORD] = r[(i, 0)]
                    .coeff
                    .iter()
                    .copied()
                    .skip(j)
                    .step_by(D_SWITCH / D_RECORD)
                    .collect_vec()
                    .try_into()
                    .unwrap();
                result.push(RecordBytesImpl {
                    it: Self::decode_record(&IntModCyclo::from(record_coeffs)),
                });
            }
        }
        result
    }

    pub fn answer_query_expand(
        ((auto_keys_regev, auto_keys_gsw), regev_to_gsw_key, _, _): &<Self as PIR>::PublicParams,
        ((seed_reg, vec_reg), (seed_gsw, vec_gsw)): &<Self as Respire>::QueryOne,
    ) -> <Self as Respire>::QueryOneExpanded {
        let inv = <Self as Respire>::RingQFast::from(mod_inverse(D as u64, Q));
        let mut c_regevs = {
            let mut c1_reg = IntModCyclo::zero();
            for (i, coeff) in vec_reg.iter().copied().enumerate() {
                c1_reg.coeff[reverse_bits_fast::<D>(i)] = coeff;
            }
            let mut c_reg = Self::regev_recover_from_seeded((
                *seed_reg,
                <Self as Respire>::RingQFast::from(&c1_reg),
            ));
            c_reg[(0, 0)] *= &inv;
            c_reg[(1, 0)] *= &inv;
            vec![c_reg]
        };

        let mut c_gsws = {
            let mut c1_gsw = IntModCyclo::zero();
            for (i, coeff) in vec_gsw.iter().copied().enumerate() {
                c1_gsw.coeff[reverse_bits_fast::<D>(i)] = coeff;
            }
            let mut c_gsw = Self::regev_recover_from_seeded((
                *seed_gsw,
                <Self as Respire>::RingQFast::from(&c1_gsw),
            ));
            c_gsw[(0, 0)] *= &inv;
            c_gsw[(1, 0)] *= &inv;
            vec![c_gsw]
        };

        assert_eq!(1 << auto_keys_regev.len(), D);
        assert_eq!(1 << auto_keys_gsw.len(), D);

        let i0 = Instant::now();
        for (i, auto_key_regev) in auto_keys_regev.iter().enumerate() {
            c_regevs = Self::do_coeff_expand_iter::<T_COEFF_REGEV, Z_COEFF_REGEV>(
                i,
                c_regevs.as_slice(),
                auto_key_regev,
            );
            let denom = D >> (i + 1);
            c_regevs.truncate((Self::REGEV_COUNT + denom - 1) / denom);
        }
        assert_eq!(c_regevs.len(), Self::REGEV_COUNT);

        let i1 = Instant::now();
        for (i, auto_key_gsw) in auto_keys_gsw.iter().enumerate() {
            c_gsws = Self::do_coeff_expand_iter::<T_COEFF_GSW, Z_COEFF_GSW>(
                i,
                c_gsws.as_slice(),
                auto_key_gsw,
            );
            let denom = D >> (i + 1);
            c_gsws.truncate((Self::GSW_COUNT + denom - 1) / denom);
        }
        assert_eq!(c_gsws.len(), Self::GSW_COUNT);

        let i2 = Instant::now();
        let mut c_gsws_iter = c_gsws
            .chunks_exact(T_GSW)
            .map(|cs| Self::regev_to_gsw(regev_to_gsw_key, cs));

        let c_gsws_fold = (0..Self::GSW_FOLD_COUNT)
            .map(|_| c_gsws_iter.next().unwrap())
            .collect();
        let c_gsws_proj = (0..Self::GSW_PROJ_COUNT)
            .map(|_| c_gsws_iter.next().unwrap())
            .collect();
        assert_eq!(c_gsws_iter.next(), None);

        let i3 = Instant::now();
        eprintln!("(**) answer query expand (reg): {:?}", i1 - i0);
        eprintln!("(**) answer query expand (gsw): {:?}", i2 - i1);
        eprintln!("(**) answer query expand (reg_to_gsw): {:?}", i3 - i2);

        (c_regevs, c_gsws_fold, c_gsws_proj)
    }

    pub fn answer_first_dim(
        db: &<Self as PIR>::Database,
        regevs: &[<Self as Respire>::RegevCiphertext],
    ) -> Vec<<Self as Respire>::RegevCiphertext> {
        assert_eq!(regevs.len(), Self::PACKED_DIM1_SIZE);

        // Flatten + transpose the ciphertexts
        let mut c0s: Vec<SimdVec> = Vec::with_capacity((D / SIMD_LANES) * Self::PACKED_DIM1_SIZE);
        let mut c1s: Vec<SimdVec> = Vec::with_capacity((D / SIMD_LANES) * Self::PACKED_DIM1_SIZE);

        #[cfg(not(target_feature = "avx2"))]
        for eval_idx in 0..D {
            for c in regevs.iter() {
                let c0_lo = u64::from(c[(0, 0)].proj1.evals[eval_idx]);
                let c0_hi = u64::from(c[(0, 0)].proj2.evals[eval_idx]);
                c0s.push((c0_hi << 32) | c0_lo);

                let c1_lo = u64::from(c[(1, 0)].proj1.evals[eval_idx]);
                let c1_hi = u64::from(c[(1, 0)].proj2.evals[eval_idx]);
                c1s.push((c1_hi << 32) | c1_lo);
            }
        }

        #[cfg(target_feature = "avx2")]
        for eval_vec_idx in 0..(D / SIMD_LANES) {
            for c in regevs.iter() {
                let mut c0_vec: SimdVec = Aligned32([0_u64; 4]);
                let mut c1_vec: SimdVec = Aligned32([0_u64; 4]);
                for lane_idx in 0..SIMD_LANES {
                    let from_idx = eval_vec_idx * SIMD_LANES + lane_idx;

                    let c0_lo = u64::from(c[(0, 0)].proj1.evals[from_idx]);
                    let c0_hi = u64::from(c[(0, 0)].proj2.evals[from_idx]);
                    c0_vec.0[lane_idx] = (c0_hi << 32) | c0_lo;

                    let c1_lo = u64::from(c[(1, 0)].proj1.evals[from_idx]);
                    let c1_hi = u64::from(c[(1, 0)].proj2.evals[from_idx]);
                    c1_vec.0[lane_idx] = (c1_hi << 32) | c1_lo;
                }
                c0s.push(c0_vec);
                c1s.push(c1_vec);
            }
        }

        // First dimension processing
        let mut result: Vec<<Self as Respire>::RegevCiphertext> = (0..Self::PACKED_DIM2_SIZE)
            .map(|_| <Self as Respire>::RegevCiphertext::zero())
            .collect();

        // Norm is at most max(Q_A, Q_B)^2 for each term
        // Add one for margin
        let reduce_every = 1 << (64 - 2 * ceil_log(2, max(Q_A, Q_B)) - 1);

        // We want to compute the sum over i of ct_i * db_(i, j).
        // Here db_(i, j) are scalars; ct_i are 2 x 1 matrices.

        #[cfg(not(target_feature = "avx2"))]
        for eval_idx in 0..D {
            for j in 0..Self::PACKED_DIM2_SIZE {
                let mut sum0_proj1 = 0_u64;
                let mut sum0_proj2 = 0_u64;
                let mut sum1_proj1 = 0_u64;
                let mut sum1_proj2 = 0_u64;

                for i in 0..Self::PACKED_DIM1_SIZE {
                    let lhs0 = c0s[eval_idx * Self::PACKED_DIM1_SIZE + i];
                    let lhs0_proj1 = lhs0 as u32 as u64;
                    let lhs0_proj2 = lhs0 >> 32;

                    let lhs1 = c1s[eval_idx * Self::PACKED_DIM1_SIZE + i];
                    let lhs1_proj1 = lhs1 as u32 as u64;
                    let lhs1_proj2 = lhs1 >> 32;

                    let rhs = db[eval_idx * Self::PACKED_DB_SIZE + j * Self::PACKED_DIM1_SIZE + i];
                    let rhs_proj1 = rhs as u32 as u64;
                    let rhs_proj2 = rhs >> 32;

                    sum0_proj1 += lhs0_proj1 * rhs_proj1;
                    sum0_proj2 += lhs0_proj2 * rhs_proj2;
                    sum1_proj1 += lhs1_proj1 * rhs_proj1;
                    sum1_proj2 += lhs1_proj2 * rhs_proj2;

                    if i % reduce_every == 0 || i == Self::PACKED_DIM1_SIZE - 1 {
                        sum0_proj1 %= Q_A;
                        sum0_proj2 %= Q_B;
                        sum1_proj1 %= Q_A;
                        sum1_proj2 %= Q_B;
                    }
                }

                result[j][(0, 0)].proj1.evals[eval_idx] = IntMod::from(sum0_proj1);
                result[j][(0, 0)].proj2.evals[eval_idx] = IntMod::from(sum0_proj2);
                result[j][(1, 0)].proj1.evals[eval_idx] = IntMod::from(sum1_proj1);
                result[j][(1, 0)].proj2.evals[eval_idx] = IntMod::from(sum1_proj2);
            }
        }

        #[cfg(target_feature = "avx2")]
        for eval_vec_idx in 0..(D / SIMD_LANES) {
            use std::arch::x86_64::*;
            unsafe {
                for j in 0..Self::PACKED_DIM2_SIZE {
                    let mut sum0_proj1 = _mm256_setzero_si256();
                    let mut sum0_proj2 = _mm256_setzero_si256();
                    let mut sum1_proj1 = _mm256_setzero_si256();
                    let mut sum1_proj2 = _mm256_setzero_si256();

                    for i in 0..Self::PACKED_DIM1_SIZE {
                        let lhs0_ptr = c0s.get_unchecked(eval_vec_idx * Self::PACKED_DIM1_SIZE + i)
                            as *const SimdVec
                            as *const __m256i;
                        let lhs1_ptr = c1s.get_unchecked(eval_vec_idx * Self::PACKED_DIM1_SIZE + i)
                            as *const SimdVec
                            as *const __m256i;
                        let rhs_ptr = db.get_unchecked(
                            eval_vec_idx * Self::PACKED_DB_SIZE + j * Self::PACKED_DIM1_SIZE + i,
                        ) as *const SimdVec as *const __m256i;

                        let lhs0_proj1 = _mm256_load_si256(lhs0_ptr);
                        let lhs0_proj2 = _mm256_srli_epi64::<32>(lhs0_proj1);
                        let lhs1_proj1 = _mm256_load_si256(lhs1_ptr);
                        let lhs1_proj2 = _mm256_srli_epi64::<32>(lhs1_proj1);
                        let rhs_proj1 = _mm256_load_si256(rhs_ptr);
                        let rhs_proj2 = _mm256_srli_epi64::<32>(rhs_proj1);

                        sum0_proj1 =
                            _mm256_add_epi64(sum0_proj1, _mm256_mul_epu32(lhs0_proj1, rhs_proj1));
                        sum0_proj2 =
                            _mm256_add_epi64(sum0_proj2, _mm256_mul_epu32(lhs0_proj2, rhs_proj2));
                        sum1_proj1 =
                            _mm256_add_epi64(sum1_proj1, _mm256_mul_epu32(lhs1_proj1, rhs_proj1));
                        sum1_proj2 =
                            _mm256_add_epi64(sum1_proj2, _mm256_mul_epu32(lhs1_proj2, rhs_proj2));

                        if i % reduce_every == 0 || i == Self::PACKED_DIM1_SIZE - 1 {
                            let mut tmp0_proj1: SimdVec = Aligned32([0_u64; 4]);
                            let mut tmp0_proj2: SimdVec = Aligned32([0_u64; 4]);
                            let mut tmp1_proj1: SimdVec = Aligned32([0_u64; 4]);
                            let mut tmp1_proj2: SimdVec = Aligned32([0_u64; 4]);
                            _mm256_store_si256(
                                &mut tmp0_proj1 as *mut SimdVec as *mut __m256i,
                                sum0_proj1,
                            );
                            _mm256_store_si256(
                                &mut tmp0_proj2 as *mut SimdVec as *mut __m256i,
                                sum0_proj2,
                            );
                            _mm256_store_si256(
                                &mut tmp1_proj1 as *mut SimdVec as *mut __m256i,
                                sum1_proj1,
                            );
                            _mm256_store_si256(
                                &mut tmp1_proj2 as *mut SimdVec as *mut __m256i,
                                sum1_proj2,
                            );
                            for lane in 0..SIMD_LANES {
                                tmp0_proj1.0[lane] %= Q_A;
                                tmp0_proj2.0[lane] %= Q_B;
                                tmp1_proj1.0[lane] %= Q_A;
                                tmp1_proj2.0[lane] %= Q_B;
                            }
                            sum0_proj1 =
                                _mm256_load_si256(&tmp0_proj1 as *const SimdVec as *const __m256i);
                            sum0_proj2 =
                                _mm256_load_si256(&tmp0_proj2 as *const SimdVec as *const __m256i);
                            sum1_proj1 =
                                _mm256_load_si256(&tmp1_proj1 as *const SimdVec as *const __m256i);
                            sum1_proj2 =
                                _mm256_load_si256(&tmp1_proj2 as *const SimdVec as *const __m256i);
                        }
                    }

                    let sum0_proj1_ptr = result[j][(0, 0)]
                        .proj1
                        .evals
                        .get_unchecked_mut(eval_vec_idx * SIMD_LANES)
                        as *mut IntMod<Q_A>
                        as *mut __m256i;
                    let sum0_proj2_ptr = result[j][(0, 0)]
                        .proj2
                        .evals
                        .get_unchecked_mut(eval_vec_idx * SIMD_LANES)
                        as *mut IntMod<Q_B>
                        as *mut __m256i;
                    let sum1_proj1_ptr = result[j][(1, 0)]
                        .proj1
                        .evals
                        .get_unchecked_mut(eval_vec_idx * SIMD_LANES)
                        as *mut IntMod<Q_A>
                        as *mut __m256i;
                    let sum1_proj2_ptr = result[j][(1, 0)]
                        .proj2
                        .evals
                        .get_unchecked_mut(eval_vec_idx * SIMD_LANES)
                        as *mut IntMod<Q_B>
                        as *mut __m256i;
                    _mm256_store_si256(sum0_proj1_ptr, sum0_proj1);
                    _mm256_store_si256(sum0_proj2_ptr, sum0_proj2);
                    _mm256_store_si256(sum1_proj1_ptr, sum1_proj1);
                    _mm256_store_si256(sum1_proj2_ptr, sum1_proj2);
                }
            }
        }

        result
    }

    pub fn answer_fold(
        first_dim_folded: Vec<<Self as Respire>::RegevCiphertext>,
        gsws: &[<Self as Respire>::GSWCiphertext],
    ) -> <Self as Respire>::RegevCiphertext {
        assert_eq!(gsws.len(), Self::ETA2 * (Z_FOLD - 1));
        let fold_size: usize = Z_FOLD.pow(Self::ETA2 as u32);

        let mut curr = first_dim_folded;
        let mut curr_size = fold_size;
        for gsw_idx in 0..ETA2 {
            curr.truncate(curr_size);
            for fold_idx in 0..curr_size / Z_FOLD {
                let c0 = curr[fold_idx].clone();
                for i in 1..Z_FOLD {
                    let c_i = &curr[i * curr_size / Z_FOLD + fold_idx];
                    let c_i_sub_c0 = Self::regev_sub_hom(c_i, &c0);
                    let b = &gsws[gsw_idx * (Z_FOLD - 1) + i - 1];
                    let c_i_sub_c0_mul_b = Self::hybrid_mul_hom(&c_i_sub_c0, b);
                    curr[fold_idx] += &c_i_sub_c0_mul_b;
                }
            }
            curr_size /= Z_FOLD;
        }
        curr.remove(0)
    }

    pub fn answer_project(
        ((_, auto_key_gsws), _, _, _): &<Self as PIR>::PublicParams,
        mut ct: <Self as Respire>::RegevCiphertext,
        gsws: &[<Self as Respire>::GSWCiphertext],
    ) -> <Self as Respire>::RegevCiphertext {
        // TODO use different T/Z/auto keys
        for (iter, (gsw, auto_key)) in gsws.iter().zip(auto_key_gsws).enumerate() {
            ct = Self::project_hom::<T_COEFF_GSW, Z_COEFF_GSW>(iter, &ct, gsw, auto_key);
        }
        ct
    }

    pub fn encode_setup() -> <Self as Respire>::RingQFast {
        let mut rng = ChaCha20Rng::from_entropy();
        <Self as Respire>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng)
    }

    pub fn encode_vec_setup() -> <Self as Respire>::VecEncodingKey {
        let mut result = Matrix::zero();
        for i in 0..N_VEC {
            result[(i, 0)] = Self::encode_setup();
        }
        result
    }

    pub fn encode_regev(
        s_encode: &<Self as Respire>::EncodingKey,
        mu: &<Self as Respire>::RingQ,
    ) -> <Self as Respire>::RegevCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut c = Matrix::zero();
        c[(0, 0)] = <Self as Respire>::RingQFast::rand_uniform(&mut rng);
        let e = <Self as Respire>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(
            &mut rng,
        );
        let mut c1 = &c[(0, 0)] * s_encode;
        c1 += &e;
        c1 += &<Self as Respire>::RingQFast::from(mu);
        c[(1, 0)] = c1;
        c
    }

    pub fn encode_regev_seeded(
        s_encode: &<Self as Respire>::EncodingKey,
        mu: &<Self as Respire>::RingQ,
    ) -> <Self as Respire>::RegevSeeded {
        let mut rng = ChaCha20Rng::from_entropy();
        let seed = rng.gen();
        let c0 = {
            let mut seeded_rng = ChaCha20Rng::from_seed(seed);
            <Self as Respire>::RingQFast::rand_uniform(&mut seeded_rng)
        };
        let e = <Self as Respire>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(
            &mut rng,
        );
        let mut c1 = &c0 * s_encode;
        c1 += &e;
        c1 += &<Self as Respire>::RingQFast::from(mu);
        (seed, c1)
    }

    pub fn regev_recover_from_seeded(
        (seed, c1): <Self as Respire>::RegevSeeded,
    ) -> <Self as Respire>::RegevCiphertext {
        let c0 = {
            let mut seeded_rng = ChaCha20Rng::from_seed(seed);
            <Self as Respire>::RingQFast::rand_uniform(&mut seeded_rng)
        };
        let mut result = Matrix::zero();
        result[(0, 0)] = c0;
        result[(1, 0)] = c1;
        result
    }

    pub fn encode_vec_regev(
        s_vec: &<Self as Respire>::VecEncodingKey,
        mu: &Matrix<N_VEC, 1, <Self as Respire>::RingQ>,
    ) -> <Self as Respire>::VecRegevCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let c_r = <Self as Respire>::RingQFast::rand_uniform(&mut rng);
        let e = Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let mut c_m = s_vec * &c_r;
        c_m += &e;
        c_m += &mu.map_ring(|r| <Self as Respire>::RingQFast::from(r));
        (c_r, c_m)
    }

    pub fn decode_regev(
        s_encode: &<Self as Respire>::EncodingKey,
        c: &<Self as Respire>::RegevCiphertext,
    ) -> <Self as Respire>::RingQ {
        <Self as Respire>::RingQ::from(&(&c[(1, 0)] - &(&c[(0, 0)] * s_encode)))
    }

    pub fn decode_vec_regev(
        s_vec: &<Self as Respire>::VecEncodingKey,
        (c_r, c_m): &<Self as Respire>::VecRegevCiphertext,
    ) -> Matrix<N_VEC, 1, <Self as Respire>::RingQ> {
        (c_m - &(s_vec * c_r)).map_ring(|r| <Self as Respire>::RingQ::from(r))
    }

    pub fn encode_gsw(
        s_encode: &<Self as Respire>::EncodingKey,
        mu: &<Self as Respire>::RingQ,
    ) -> <Self as Respire>::GSWCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M_GSW, <Self as Respire>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<1, M_GSW, <Self as Respire>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<2, M_GSW, <Self as Respire>::RingQFast> =
            &Matrix::stack(&a_t, &(&(&a_t * s_encode) + &e_mat))
                + &(&build_gadget::<<Self as Respire>::RingQFast, 2, M_GSW, Z_GSW, T_GSW>()
                    * &<Self as Respire>::RingQFast::from(mu));
        c_mat
    }

    pub fn decode_gsw_scaled(
        s_encode: &<Self as Respire>::EncodingKey,
        c: &<Self as Respire>::GSWCiphertext,
        scale: &<Self as Respire>::RingQFast,
    ) -> <Self as Respire>::RingQ {
        let scaled_ident = &Matrix::<2, 2, <Self as Respire>::RingQFast>::identity() * scale;
        let mut s_t = Matrix::<1, 2, <Self as Respire>::RingQFast>::zero();
        s_t[(0, 0)] = (-s_encode).clone();
        s_t[(0, 1)] = <Self as Respire>::RingQFast::one();
        let result_q_fast_mat = &(&s_t * c)
            * &gadget_inverse::<<Self as Respire>::RingQFast, 2, M_GSW, 2, Z_GSW, T_GSW>(
                &scaled_ident,
            );
        let result_q = <Self as Respire>::RingQ::from(&result_q_fast_mat[(0, 1)]);
        <Self as Respire>::RingQ::from(result_q)
    }

    pub fn regev_sub_hom(
        lhs: &<Self as Respire>::RegevCiphertext,
        rhs: &<Self as Respire>::RegevCiphertext,
    ) -> <Self as Respire>::RegevCiphertext {
        lhs - rhs
    }

    pub fn hybrid_mul_hom(
        regev: &<Self as Respire>::RegevCiphertext,
        gsw: &<Self as Respire>::GSWCiphertext,
    ) -> <Self as Respire>::RegevCiphertext {
        gsw * &gadget_inverse::<<Self as Respire>::RingQFast, 2, M_GSW, 1, Z_GSW, T_GSW>(regev)
    }

    pub fn regev_mul_x_pow(
        c: &<Self as Respire>::RegevCiphertext,
        k: usize,
    ) -> <Self as Respire>::RegevCiphertext {
        c.map_ring(|x| x.mul_x_pow(k))
    }

    pub fn gsw_mul_x_pow(
        c: &<Self as Respire>::GSWCiphertext,
        k: usize,
    ) -> <Self as Respire>::GSWCiphertext {
        c.map_ring(|x| x.mul_x_pow(k))
    }

    pub fn auto_setup<const LEN: usize, const BASE: u64>(
        tau_power: usize,
        s_encode: &<Self as Respire>::RingQFast,
    ) -> <Self as Respire>::AutoKey<LEN> {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, LEN, <Self as Respire>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_t: Matrix<1, LEN, <Self as Respire>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let mut bottom = &a_t * s_encode;
        bottom += &e_t;
        bottom -= &(&build_gadget::<<Self as Respire>::RingQFast, 1, LEN, BASE, LEN>()
            * &s_encode.auto(tau_power));
        (Matrix::stack(&a_t, &bottom), tau_power)
    }

    pub fn auto_hom<const LEN: usize, const BASE: u64>(
        (w_mat, tau_power): &<Self as Respire>::AutoKey<LEN>,
        c: &<Self as Respire>::RegevCiphertext,
    ) -> <Self as Respire>::RegevCiphertext {
        let c0 = &c[(0, 0)];
        let c1 = &c[(1, 0)];
        let mut g_inv_tau_c0 = Matrix::<LEN, 1, <Self as Respire>::RingQFast>::zero();

        <<Self as Respire>::RingQFast as RingElementDecomposable<BASE, LEN>>::decompose_into_mat(
            &c0.auto(*tau_power),
            &mut g_inv_tau_c0,
            0,
            0,
        );
        let mut result = w_mat * &g_inv_tau_c0;

        result[(1, 0)] += &c1.auto(*tau_power);
        result
    }

    ///
    /// Perform a single iteration of the coefficient expansion algorithm, doubling the length of the
    /// input ciphertexts.
    ///
    /// # Parameters
    /// * `which_iter`: the zero-indexed iteration number; the ciphertexts are assumed to be
    ///   encryptions of plaintexts that only have coefficients of degree divisible `2^which_iter`.
    /// * `cts`: the input ciphertexts
    /// * `auto_key`: the automorphism key, which should have power equal to `D / 2^which_iter + 1`
    ///
    pub fn do_coeff_expand_iter<const LEN: usize, const BASE: u64>(
        which_iter: usize,
        cts: &[<Self as Respire>::RegevCiphertext],
        auto_key: &<Self as Respire>::AutoKey<LEN>,
    ) -> Vec<<Self as Respire>::RegevCiphertext> {
        debug_assert_eq!(auto_key.1, D / (1 << which_iter) + 1);
        let len = cts.len();
        let mut cts_new = Vec::with_capacity(2 * len);
        cts_new.resize(2 * len, Matrix::zero());
        for (j, ct) in cts.iter().enumerate() {
            let shift_exp = 1 << which_iter;
            let shift_auto_exp = (shift_exp * auto_key.1) % (2 * D);

            let ct_shifted = Self::regev_mul_x_pow(ct, 2 * D - shift_exp);
            let ct_auto = Self::auto_hom::<LEN, BASE>(auto_key, ct);
            let ct_auto_shifted = Self::regev_mul_x_pow(&ct_auto, 2 * D - shift_auto_exp);

            cts_new[2 * j] = ct + &ct_auto;
            cts_new[2 * j + 1] = &ct_shifted + &ct_auto_shifted;
        }
        cts_new
    }

    pub fn project_hom<const LEN: usize, const BASE: u64>(
        which_iter: usize,
        ct: &<Self as Respire>::RegevCiphertext,
        gsw: &<Self as Respire>::GSWCiphertext,
        auto_key: &<Self as Respire>::AutoKey<LEN>,
    ) -> <Self as Respire>::RegevCiphertext {
        debug_assert_eq!(auto_key.1, D / (1 << which_iter) + 1);
        let shift_exp = 1 << which_iter;
        let diff = &Self::gsw_mul_x_pow(gsw, 2 * D - shift_exp) - gsw;
        let mul = Self::hybrid_mul_hom(ct, &diff);
        let shifted = &mul + ct;
        let auto = Self::auto_hom::<LEN, BASE>(auto_key, &shifted);
        &shifted + &auto
    }

    pub fn regev_to_gsw_setup(
        s_encode: &<Self as Respire>::EncodingKey,
    ) -> <Self as Respire>::RegevToGSWKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t = Matrix::<1, M_CONV, <Self as Respire>::RingQFast>::rand_uniform(&mut rng);
        let e_mat = Matrix::<1, M_CONV, <Self as Respire>::RingQFast>::rand_discrete_gaussian::<
            _,
            NOISE_WIDTH_MILLIONTHS,
        >(&mut rng);
        let mut bottom = &a_t * s_encode;
        bottom += &e_mat;
        let g_vec = build_gadget::<<Self as Respire>::RingQFast, 1, T_CONV, Z_CONV, T_CONV>();
        let mut s_encode_tensor_g = Matrix::<1, M_CONV, <Self as Respire>::RingQFast>::zero();
        s_encode_tensor_g.copy_into(&g_vec, 0, T_CONV);
        s_encode_tensor_g.copy_into(&(&g_vec * &(-s_encode)), 0, 0);
        bottom -= &(&s_encode_tensor_g * s_encode);

        Matrix::stack(&a_t, &bottom)
    }

    pub fn regev_to_gsw(
        v_mat: &<Self as Respire>::RegevToGSWKey,
        cs: &[<Self as Respire>::RegevCiphertext],
    ) -> <Self as Respire>::GSWCiphertext {
        let mut result = Matrix::<2, M_GSW, <Self as Respire>::RingQFast>::zero();
        let mut c_hat = Matrix::<2, T_GSW, <Self as Respire>::RingQFast>::zero();
        for (i, ci) in cs.iter().enumerate() {
            c_hat.copy_into(ci, 0, i);
        }
        let g_inv_c_hat =
            gadget_inverse::<<Self as Respire>::RingQFast, 2, M_CONV, T_GSW, Z_CONV, T_CONV>(
                &c_hat,
            );
        let v_g_inv_c_hat = v_mat * &g_inv_c_hat;
        result.copy_into(&v_g_inv_c_hat, 0, 0);
        for (i, ci) in cs.iter().enumerate() {
            result.copy_into(ci, 0, T_GSW + i);
        }

        // No permutation needed for scalar regev
        result
    }

    pub fn key_switch_setup(
        s_from: &<Self as Respire>::VecEncodingKeyQ2,
        s_to: &<Self as Respire>::VecEncodingKeyQ2,
    ) -> <Self as Respire>::KeySwitchKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t = Matrix::<1, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>::rand_uniform(&mut rng);
        let e_mat =
            Matrix::<N_VEC, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>::rand_discrete_gaussian::<
                _,
                NOISE_WIDTH_MILLIONTHS,
            >(&mut rng);
        let mut b_mat = &(-s_from)
            * &build_gadget::<IntModCycloEval<D, Q_SWITCH2>, 1, T_SWITCH, Z_SWITCH, T_SWITCH>();
        b_mat += &(s_to * &a_t);
        b_mat += &e_mat;
        (a_t, b_mat)
    }

    pub fn scal_to_vec_setup(
        s_scal: &<Self as Respire>::EncodingKey,
        s_vec: &<Self as Respire>::VecEncodingKey,
    ) -> <Self as Respire>::ScalToVecKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut result = Vec::with_capacity(N_VEC);
        for i in 0..N_VEC {
            let mut unit = Matrix::<N_VEC, 1, <Self as Respire>::RingQFast>::zero();
            unit[(i, 0)] = <Self as Respire>::RingQFast::one();
            let unit = unit;

            let a_t =
                Matrix::<1, T_SCAL_TO_VEC, <Self as Respire>::RingQFast>::rand_uniform(&mut rng);
            let e_mat =
                Matrix::<N_VEC, T_SCAL_TO_VEC, <Self as Respire>::RingQFast>::rand_discrete_gaussian::<
                    _,
                    NOISE_WIDTH_MILLIONTHS,
                >(&mut rng);
            let mut bottom = s_vec * &a_t;
            bottom += &e_mat;
            let embedding = &(&unit * s_scal)
                * &build_gadget::<_, 1, T_SCAL_TO_VEC, Z_SCAL_TO_VEC, T_SCAL_TO_VEC>();
            bottom -= &embedding;
            result.push((a_t, bottom));
        }
        result
    }

    pub fn scal_to_vec(
        s_scal_to_vec: &<Self as Respire>::ScalToVecKey,
        cs: &[<Self as Respire>::RegevCiphertext; N_VEC],
    ) -> <Self as Respire>::VecRegevCiphertext {
        let mut result_rand = <Self as Respire>::RingQFast::zero();
        let mut result_embed = Matrix::<N_VEC, 1, <Self as Respire>::RingQFast>::zero();
        for (i, c) in cs.iter().enumerate() {
            let c0 = &c[(0, 0)];
            let c1 = &c[(1, 0)];
            let g_inv = gadget_inverse_scalar::<_, Z_SCAL_TO_VEC, T_SCAL_TO_VEC>(c0);
            result_rand += &(&s_scal_to_vec[i].0 * &g_inv)[(0, 0)];
            result_embed += &(&s_scal_to_vec[i].1 * &g_inv);
            result_embed[(i, 0)] += c1;
        }
        (result_rand, result_embed)
    }

    pub fn encode_record(bytes: &RecordBytesImpl<BYTES_PER_RECORD>) -> IntModCyclo<D_RECORD, P> {
        let bit_iter = BitSlice::<u8, Msb0>::from_slice(&bytes.it);
        let p_bits = floor_log(2, P);
        let coeff = bit_iter
            .chunks(p_bits)
            .map(|c| IntMod::<P>::from(c.iter().fold(0, |acc, b| 2 * acc + *b as u64)))
            .collect_vec();
        let coeff_slice: [IntMod<P>; D_RECORD] = coeff.try_into().unwrap();
        IntModCyclo::from(coeff_slice)
    }

    pub fn decode_record(record: &IntModCyclo<D_RECORD, P>) -> [u8; BYTES_PER_RECORD] {
        let p_bits = floor_log(2, P);
        let bit_iter = record.coeff.iter().flat_map(|x| {
            u64::from(*x)
                .into_bitarray::<Msb0>()
                .into_iter()
                .skip(64 - p_bits)
                .take(p_bits)
        });
        let bytes = bit_iter
            .chunks(8)
            .into_iter()
            .map(|c| c.fold(0, |acc, b| 2 * acc + b as u8))
            .collect_vec();
        bytes.try_into().unwrap()
    }

    pub fn noise_variance(
        s_scal: &<Self as Respire>::EncodingKey,
        c: &<Self as Respire>::RegevCiphertext,
    ) -> f64 {
        let decoded: <Self as Respire>::RingQ = Self::decode_regev(&s_scal, c);
        let message: <Self as Respire>::RingP = decoded.round_down_into();
        let noise: <Self as Respire>::RingQ = &decoded - &message.scale_up_into();

        let mut sum = 0_f64;
        let mut samples = 0_usize;

        for e in noise.coeff.iter() {
            let e_sq = (e.norm() as f64) * (e.norm() as f64);
            sum += e_sq;
            samples += 1;
        }

        sum / samples as f64
    }

    fn variance_to_subgaussian_bits(x: f64) -> f64 {
        // Subgaussian widths = sqrt(2*pi) * (standard deviation)
        (x * (2f64 * PI)).log2() / 2f64
    }

    pub fn noise_subgaussian_bits(
        s_scal: &<Self as Respire>::EncodingKey,
        c: &<Self as Respire>::RegevCiphertext,
    ) -> f64 {
        Self::variance_to_subgaussian_bits(Self::noise_variance(s_scal, c))
    }

    pub fn noise_subgaussian_bits_vec(
        s_vec: &<Self as Respire>::VecEncodingKey,
        (cr, cm): &<Self as Respire>::VecRegevCiphertext,
    ) -> f64 {
        let mut total = 0_f64;
        for i in 0..N_VEC {
            let mut fake_ct = Matrix::zero();
            fake_ct[(0, 0)] = cr.clone();
            fake_ct[(1, 0)] = cm[(i, 0)].clone();
            let fake_s = s_vec[(i, 0)].clone();
            total += Self::noise_variance(&fake_s, &fake_ct);
        }
        Self::variance_to_subgaussian_bits(total / N_VEC as f64)
    }
});
