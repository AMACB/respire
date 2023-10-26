use itertools::Itertools;
use std::cmp::max;
use std::f64::consts::PI;
use std::time::Instant;
use std::{iter, slice};

use crate::math::discrete_gaussian::NUM_WIDTHS;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::math::gadget::{
    base_from_len, build_gadget, gadget_inverse, gadget_inverse_scalar, RingElementDecomposable,
};
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::matrix::Matrix;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::ring_elem::{NormedRingElement, RingElement};
use crate::math::utils::{ceil_log, floor_log, mod_inverse, reverse_bits};

use crate::math::simd_utils::*;
use crate::pir::noise::{BoundedNoise, Independent, SubGaussianNoise};

pub struct SPIRALImpl<
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
> {}

#[allow(non_snake_case)]
pub struct SPIRALParamsRaw {
    pub Q_A: u64,
    pub Q_B: u64,
    pub D: usize,
    pub T_GSW: usize,
    pub T_COEFF_REGEV: usize,
    pub T_COEFF_GSW: usize,
    pub T_CONV: usize,
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

impl SPIRALParamsRaw {
    pub const fn expand(&self) -> SPIRALParams {
        let q = self.Q_A * self.Q_B;
        let z_gsw = base_from_len(self.T_GSW, q);
        let z_coeff_regev = base_from_len(self.T_COEFF_REGEV, q);
        let z_coeff_gsw = base_from_len(self.T_COEFF_GSW, q);
        let z_conv = base_from_len(self.T_CONV, q);
        let z_switch = base_from_len(self.T_SWITCH, self.Q_SWITCH2);
        SPIRALParams {
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
        }
    }
}

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct SPIRALParams {
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
}

impl SPIRALParams {
    pub fn correctness_param(&self) -> f64 {
        // 2 d n^2 * exp(-pi * correctness^2) <= 2^(-40)
        (-1_f64 / PI * (2_f64.powi(-40) / 2_f64 / self.D as f64).ln()).sqrt()
    }

    pub fn relative_noise_threshold(&self) -> f64 {
        1_f64 / (2_f64 * self.P as f64) / self.correctness_param()
    }

    pub fn noise_estimate(&self) -> f64 {
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
}

#[macro_export]
macro_rules! spiral {
    ($params: expr) => {
        SPIRALImpl<
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
        >
    }
}

pub trait SPIRAL {
    // Type aliases
    type RingP;
    type RingQ;
    type RingQFast;
    type RegevCiphertext;
    type RegevSmall;
    type GSWCiphertext;
    type EncodingKey;
    type AutoKey<const T: usize>;
    type AutoKeyRegev;
    type AutoKeyGSW;
    type RegevToGSWKey;
    type KeySwitchKey;

    // Associated types
    type QueryKey;
    type PublicParams;
    type Query;
    type QueryExpanded;
    type ResponseRaw;
    type Response;
    type Record;
    type RecordPackedSmall;
    type RecordPacked;
    type Database;

    // Constants
    const PACKED_DIM1_SIZE: usize;
    const PACKED_DIM2_SIZE: usize;
    const PACKED_DB_SIZE: usize;
    const DB_SIZE: usize;
    const PACK_RATIO: usize;
    const PACK_RATIO_SMALL: usize;
    const ETA1: usize;
    const ETA2: usize;
    const REGEV_COUNT: usize;
    const REGEV_EXPAND_ITERS: usize;
    const GSW_FOLD_COUNT: usize;
    const GSW_PROJ_COUNT: usize;
    const GSW_COUNT: usize;
    const GSW_EXPAND_ITERS: usize;

    fn preprocess<I: ExactSizeIterator<Item = Self::Record>>(records_iter: I) -> Self::Database;
    fn setup() -> (Self::QueryKey, Self::PublicParams);
    fn query(qk: &Self::QueryKey, idx: usize) -> Self::Query;
    fn answer(pp: &Self::PublicParams, db: &Self::Database, q: &Self::Query) -> Self::ResponseRaw;
    fn response_compress(pp: &Self::PublicParams, r: &Self::ResponseRaw) -> Self::Response;
    fn response_extract(qk: &Self::QueryKey, r: &Self::Response) -> Self::RecordPackedSmall;
    fn response_raw_stats(
        qk: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::ResponseRaw,
        actual: &<Self as SPIRAL>::RecordPackedSmall,
    ) -> f64;
}

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
    > SPIRAL
    for SPIRALImpl<
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
    >
{
    // Type aliases
    type RingP = IntModCyclo<D, P>;
    type RingQ = IntModCyclo<D, Q>;
    type RingQFast = IntModCycloCRTEval<D, Q_A, Q_B>;
    type RegevCiphertext = Matrix<2, 1, Self::RingQFast>;
    type RegevSmall = (
        IntModCyclo<D_SWITCH, Q_SWITCH2>,
        IntModCyclo<D_SWITCH, Q_SWITCH1>,
    );
    type GSWCiphertext = Matrix<2, M_GSW, Self::RingQFast>;

    type EncodingKey = Self::RingQFast;
    type AutoKey<const T: usize> = (Matrix<2, T, Self::RingQFast>, usize);
    type AutoKeyRegev = Self::AutoKey<T_COEFF_REGEV>;
    type AutoKeyGSW = Self::AutoKey<T_COEFF_GSW>;
    type RegevToGSWKey = Matrix<2, M_CONV, Self::RingQFast>;
    type KeySwitchKey = (
        Matrix<1, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>,
        Matrix<1, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>,
    );

    // Associated types
    type QueryKey = (
        Self::EncodingKey,                    // main key
        IntModCycloEval<D_SWITCH, Q_SWITCH2>, // small ring key
    );
    type PublicParams = (
        (
            Self::AutoKeyGSW,
            Vec<Self::AutoKeyRegev>,
            Vec<Self::AutoKeyGSW>,
        ),
        Self::RegevToGSWKey,
        Self::KeySwitchKey,
    );
    type Query = Self::RegevCiphertext;
    type QueryExpanded = (
        Vec<Self::RegevCiphertext>,
        Vec<Self::GSWCiphertext>,
        Vec<Self::GSWCiphertext>,
    );
    type ResponseRaw = Self::RegevCiphertext;
    type Response = Self::RegevSmall;
    type Record = IntModCyclo<D_RECORD, P>;
    type RecordPackedSmall = IntModCyclo<D_SWITCH, P>;
    type RecordPacked = IntModCyclo<D, P>;

    /// We structure the database as `[2] x [D / S] x [DIM2_SIZE] x [DIM1_SIZE] x [S]` for optimal first dimension
    /// processing. The outermost pair is the first resp. second CRT projections, packed as two u32 into one u64;
    /// `S` is the SIMD lane count that we can use, i.e. 4 for AVX2.
    type Database = Vec<SimdVec>;

    // Constants
    const PACKED_DIM1_SIZE: usize = 2_usize.pow(ETA1 as u32);
    const PACKED_DIM2_SIZE: usize = Z_FOLD.pow(ETA2 as u32);
    const PACKED_DB_SIZE: usize = Self::PACKED_DIM1_SIZE * Self::PACKED_DIM2_SIZE;
    const DB_SIZE: usize = Self::PACKED_DB_SIZE * Self::PACK_RATIO;
    const PACK_RATIO: usize = D / D_RECORD;
    const PACK_RATIO_SMALL: usize = D_SWITCH / D_RECORD;
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;

    const REGEV_COUNT: usize = 1 << ETA1;
    const REGEV_EXPAND_ITERS: usize = ETA1;
    const GSW_FOLD_COUNT: usize = ETA2 * (Z_FOLD - 1);

    // TODO add param to reduce this when we don't care about garbage being in the other slots
    const GSW_PROJ_COUNT: usize = floor_log(2, Self::PACK_RATIO as u64);
    const GSW_COUNT: usize = (Self::GSW_FOLD_COUNT + Self::GSW_PROJ_COUNT) * T_GSW;
    const GSW_EXPAND_ITERS: usize = ceil_log(2, Self::GSW_COUNT as u64);

    fn preprocess<'a, I: ExactSizeIterator<Item = Self::Record>>(
        records_iter: I,
    ) -> Self::Database {
        assert_eq!(records_iter.len(), Self::DB_SIZE);

        let records_eval: Vec<<Self as SPIRAL>::RingQFast> = records_iter
            .chunks(Self::PACK_RATIO)
            .into_iter()
            .map(|chunk| {
                let mut record_packed = IntModCyclo::<D, P>::zero();
                for (record_in_chunk, record) in chunk.enumerate() {
                    // Transpose so projection is more significant
                    let packed_offset = reverse_bits(Self::PACK_RATIO, record_in_chunk);
                    for (coeff_idx, coeff) in record.coeff.iter().enumerate() {
                        record_packed.coeff[Self::PACK_RATIO * coeff_idx + packed_offset] = *coeff;
                    }
                }
                <Self as SPIRAL>::RingQFast::from(&record_packed.include_into::<Q>())
            })
            .collect();

        #[cfg(not(target_feature = "avx2"))]
        {
            let mut db: Vec<SimdVec> = (0..(D * Self::PACKED_DB_SIZE)).map(|_| 0_u64).collect();
            for eval_vec_idx in 0..D {
                for db_idx in 0..Self::PACKED_DB_SIZE {
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

            for eval_vec_idx in 0..(D / SIMD_LANES) {
                for db_idx in 0..Self::PACKED_DB_SIZE {
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

    fn setup() -> (<Self as SPIRAL>::QueryKey, <Self as SPIRAL>::PublicParams) {
        // Regev/GSW secret key
        let s_encode = Self::encode_setup();

        // Small ring key
        let s_small = {
            let mut rng = ChaCha20Rng::from_entropy();
            IntModCycloEval::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng)
        };

        // Key switching key
        let s_encode_q2 = IntModCycloEval::from(IntModCyclo::from(
            IntModCyclo::<D, Q>::from(&s_encode)
                .coeff
                .map(|x| IntMod::from(i64::from(x))),
        ));
        let s_small_q2 = IntModCycloEval::from(IntModCyclo::from(&s_small).include_dim());
        let s_switch = Self::key_switch_setup(&s_encode_q2, &s_small_q2);

        // Automorphism keys
        let auto_key_first = Self::auto_setup::<T_COEFF_GSW, Z_COEFF_GSW>(D + 1, &s_encode);

        let mut auto_keys_regev: Vec<<Self as SPIRAL>::AutoKeyRegev> =
            Vec::with_capacity(Self::REGEV_EXPAND_ITERS);
        for i in 1..Self::REGEV_EXPAND_ITERS + 1 {
            let tau_power = (D >> i) + 1;
            auto_keys_regev.push(Self::auto_setup::<T_COEFF_REGEV, Z_COEFF_REGEV>(
                tau_power, &s_encode,
            ));
        }
        let mut auto_keys_gsw: Vec<<Self as SPIRAL>::AutoKeyGSW> =
            Vec::with_capacity(Self::GSW_EXPAND_ITERS);
        for i in 1..Self::GSW_EXPAND_ITERS + 1 {
            let tau_power = (D >> i) + 1;
            auto_keys_gsw.push(Self::auto_setup::<T_COEFF_GSW, Z_COEFF_GSW>(
                tau_power, &s_encode,
            ));
        }

        // Regev to GSW key
        let regev_to_gsw_key = Self::regev_to_gsw_setup(&s_encode);

        (
            (s_encode, s_small),
            (
                (auto_key_first, auto_keys_regev, auto_keys_gsw),
                regev_to_gsw_key,
                s_switch,
            ),
        )
    }

    fn query(
        (s_encode, _s_small): &<Self as SPIRAL>::QueryKey,
        idx: usize,
    ) -> <Self as SPIRAL>::Query {
        assert!(idx < Self::DB_SIZE);
        let (idx, proj_idx) = (idx / Self::PACK_RATIO, idx % Self::PACK_RATIO);
        let (idx_i, idx_j) = (idx / Self::PACKED_DIM2_SIZE, idx % Self::PACKED_DIM2_SIZE);

        let mut packed_vec: Vec<IntMod<Q>> = iter::repeat(IntMod::zero()).take(D).collect();

        let inv_even = IntMod::from(mod_inverse(1 << (1 + Self::REGEV_EXPAND_ITERS), Q));
        for i in 0_usize..Self::PACKED_DIM1_SIZE {
            packed_vec[2 * i] = (IntMod::<P>::from((i == idx_i) as u64)).scale_up_into() * inv_even;
        }

        // Think of the odd entries of packed as [ETA2] x [Z_FOLD - 1] x [T_GSW] + [GSW_PROJ_COUNT] x [T_GSW]
        let inv_odd = IntMod::from(mod_inverse(1 << (1 + Self::GSW_EXPAND_ITERS), Q));

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
                    packed_vec[2 * pack_idx + 1] = msg * inv_odd;
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
                packed_vec[2 * pack_idx + 1] = msg * inv_odd;
                msg *= IntMod::from(Z_GSW);
            }
        }

        let mu: IntModCyclo<D, Q> = packed_vec.into();
        Self::encode_regev(s_encode, &mu)
    }

    fn answer(
        pp: &<Self as SPIRAL>::PublicParams,
        db: &<Self as SPIRAL>::Database,
        q: &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::ResponseRaw {
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
        let result_projected = Self::answer_project(result, gsws_proj.as_slice());
        let i4 = Instant::now();

        eprintln!("(*) answer query expand: {:?}", i1 - i0);
        eprintln!("(*) answer first dim: {:?}", i2 - i1);
        eprintln!("(*) answer fold: {:?}", i3 - i2);
        eprintln!("(*) answer project: {:?}", i4 - i3);
        result_projected
    }

    fn response_compress(
        (_auto_keys, _regev_to_gsw_key, (a_t, b_t)): &Self::PublicParams,
        r: &Self::ResponseRaw,
    ) -> Self::Response {
        let c0 = IntModCyclo::<D, Q>::from(&r[(0, 0)]);
        let c1 = IntModCyclo::<D, Q>::from(&r[(1, 0)]);
        let mut c0_scaled = IntModCyclo::zero();
        for (c0_scaled_coeff, c0_coeff) in c0_scaled.coeff.iter_mut().zip(c0.coeff) {
            let numer = Q_SWITCH2 as u128 * u64::from(c0_coeff) as u128;
            let denom = Q as u128;
            let div = (numer + denom / 2) / denom;
            *c0_scaled_coeff = IntMod::from(div as u64);
        }
        let g_inv_c0_scaled = gadget_inverse_scalar::<_, Z_SWITCH, T_SWITCH>(&c0_scaled)
            .map_ring(|x| IntModCycloEval::from(x));
        let c0_hat: IntModCyclo<D_SWITCH, Q_SWITCH2> =
            IntModCyclo::from(&(a_t * &g_inv_c0_scaled)[(0, 0)]).project_dim();
        let c1_hat: IntModCyclo<D_SWITCH, Q_SWITCH1> = {
            let b_t_g_inv = IntModCyclo::from(&(b_t * &g_inv_c0_scaled)[(0, 0)]);
            let mut result = IntModCyclo::<D, Q_SWITCH1>::zero();
            for (result_coeff, (c1_coeff, b_t_g_inv_coeff)) in result
                .coeff
                .iter_mut()
                .zip(c1.coeff.iter().copied().zip(b_t_g_inv.coeff))
            {
                let numer = Q_SWITCH1 as u128 * Q_SWITCH2 as u128 * u64::from(c1_coeff) as u128
                    + Q as u128 * Q_SWITCH1 as u128 * u64::from(b_t_g_inv_coeff) as u128;
                let denom = Q as u128 * Q_SWITCH2 as u128;
                let div = (numer + denom / 2) / denom;
                *result_coeff = IntMod::from(div as u64);
            }
            result.project_dim()
        };
        (c0_hat, c1_hat)
    }

    fn response_extract(
        (_s_encode, s_small): &<Self as SPIRAL>::QueryKey,
        (c0_hat, c1_hat): &<Self as SPIRAL>::Response,
    ) -> <Self as SPIRAL>::RecordPackedSmall {
        let neg_s_small_c0 = IntModCyclo::from(-&(s_small * &IntModCycloEval::from(c0_hat)));
        let mut result = IntModCyclo::zero();
        for (result_coeff, neg_s_small_c0_coeff) in
            result.coeff.iter_mut().zip(neg_s_small_c0.coeff)
        {
            let numer = Q_SWITCH1 as u128 * u64::from(neg_s_small_c0_coeff) as u128;
            let denom = Q_SWITCH2 as u128;
            let div = (numer + denom / 2) / denom;
            *result_coeff = IntMod::from(div as u64);
        }
        result += c1_hat;
        result.round_down_into()
    }

    fn response_raw_stats(
        (s_encode, _s_small): &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::ResponseRaw,
        actual: &<Self as SPIRAL>::RecordPackedSmall,
    ) -> f64 {
        let actual_scaled = actual.scale_up_into();
        let decoded = Self::decode_regev(s_encode, r).project_dim();
        let diff = &actual_scaled - &decoded;

        let mut sum = 0_f64;
        let mut samples = 0_usize;

        for e in diff.coeff.iter() {
            let e_sq = (e.norm() as f64) * (e.norm() as f64);
            sum += e_sq;
            samples += 1;
        }

        // sigma^2 is the variance of the relative noise (noise divided by Q) of the coefficients. Return sigma.
        (sum / samples as f64).sqrt() / (Q as f64)
    }
}

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
    >
    SPIRALImpl<
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
    >
{
    pub fn answer_query_expand(
        ((auto_key_first, auto_keys_regev, auto_keys_gsw), regev_to_gsw_key, _key_switch_key): &<Self as SPIRAL>::PublicParams,
        q: &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::QueryExpanded {
        assert_eq!(auto_keys_regev.len(), Self::REGEV_EXPAND_ITERS);
        assert_eq!(auto_keys_gsw.len(), Self::GSW_EXPAND_ITERS);

        let first = Self::do_coeff_expand_iter::<T_COEFF_GSW, Z_COEFF_GSW>(
            0,
            slice::from_ref(q),
            auto_key_first,
        );
        let [regev_base, gsw_base]: [_; 2] = first.try_into().unwrap();

        let mut regevs: Vec<<Self as SPIRAL>::RegevCiphertext> = vec![regev_base];
        for (i, auto_key_regev) in auto_keys_regev.iter().enumerate() {
            regevs = Self::do_coeff_expand_iter::<T_COEFF_REGEV, Z_COEFF_REGEV>(
                i + 1,
                regevs.as_slice(),
                auto_key_regev,
            );
        }
        assert_eq!(regevs.len(), Self::PACKED_DIM1_SIZE);

        let mut gsws = vec![gsw_base];
        for (i, auto_key_gsw) in auto_keys_gsw.iter().enumerate() {
            gsws = Self::do_coeff_expand_iter::<T_COEFF_GSW, Z_COEFF_GSW>(
                i + 1,
                gsws.as_slice(),
                auto_key_gsw,
            );
        }
        gsws.truncate(Self::GSW_COUNT);

        let mut gsws_iter = gsws
            .chunks_exact(T_GSW)
            .map(|cs| Self::regev_to_gsw(regev_to_gsw_key, cs));

        let gsws_fold = (0..Self::GSW_FOLD_COUNT)
            .map(|_| gsws_iter.next().unwrap())
            .collect();
        let gsws_proj = (0..Self::GSW_PROJ_COUNT)
            .map(|_| gsws_iter.next().unwrap())
            .collect();
        assert_eq!(gsws_iter.next(), None);

        (regevs, gsws_fold, gsws_proj)
    }

    pub fn answer_first_dim(
        db: &<Self as SPIRAL>::Database,
        regevs: &[<Self as SPIRAL>::RegevCiphertext],
    ) -> Vec<<Self as SPIRAL>::RegevCiphertext> {
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
        let mut result: Vec<<Self as SPIRAL>::RegevCiphertext> = (0..Self::PACKED_DIM2_SIZE)
            .map(|_| <Self as SPIRAL>::RegevCiphertext::zero())
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
        first_dim_folded: Vec<<Self as SPIRAL>::RegevCiphertext>,
        gsws: &[<Self as SPIRAL>::GSWCiphertext],
    ) -> <Self as SPIRAL>::RegevCiphertext {
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
        ct: <Self as SPIRAL>::RegevCiphertext,
        gsws: &[<Self as SPIRAL>::GSWCiphertext],
    ) -> <Self as SPIRAL>::RegevCiphertext {
        todo!()
    }

    pub fn encode_setup() -> <Self as SPIRAL>::RingQFast {
        let mut rng = ChaCha20Rng::from_entropy();
        <Self as SPIRAL>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng)
    }

    pub fn encode_regev(
        s_encode: &<Self as SPIRAL>::EncodingKey,
        mu: &<Self as SPIRAL>::RingQ,
    ) -> <Self as SPIRAL>::RegevCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut c = Matrix::zero();
        c[(0, 0)] = <Self as SPIRAL>::RingQFast::rand_uniform(&mut rng);
        let e = <Self as SPIRAL>::RingQFast::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(
            &mut rng,
        );
        let mut c1 = &c[(0, 0)] * s_encode;
        c1 += &e;
        c1 += &<Self as SPIRAL>::RingQFast::from(mu);
        c[(1, 0)] = c1;
        c
    }

    pub fn decode_regev(
        s_encode: &<Self as SPIRAL>::EncodingKey,
        c: &<Self as SPIRAL>::RegevCiphertext,
    ) -> <Self as SPIRAL>::RingQ {
        <Self as SPIRAL>::RingQ::from(&(&c[(1, 0)] - &(&c[(0, 0)] * s_encode)))
    }

    pub fn encode_gsw(
        s_encode: &<Self as SPIRAL>::EncodingKey,
        mu: &<Self as SPIRAL>::RingQ,
    ) -> <Self as SPIRAL>::GSWCiphertext {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, M_GSW, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_mat: Matrix<1, M_GSW, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let c_mat: Matrix<2, M_GSW, <Self as SPIRAL>::RingQFast> =
            &Matrix::stack(&a_t, &(&(&a_t * s_encode) + &e_mat))
                + &(&build_gadget::<<Self as SPIRAL>::RingQFast, 2, M_GSW, Z_GSW, T_GSW>()
                    * &<Self as SPIRAL>::RingQFast::from(mu));
        c_mat
    }

    pub fn decode_gsw_scaled(
        s_encode: &<Self as SPIRAL>::EncodingKey,
        c: &<Self as SPIRAL>::GSWCiphertext,
        scale: &<Self as SPIRAL>::RingQFast,
    ) -> <Self as SPIRAL>::RingQ {
        let scaled_ident = &Matrix::<2, 2, <Self as SPIRAL>::RingQFast>::identity() * scale;
        let mut s_t = Matrix::<1, 2, <Self as SPIRAL>::RingQFast>::zero();
        s_t[(0, 0)] = (-s_encode).clone();
        s_t[(0, 1)] = <Self as SPIRAL>::RingQFast::one();
        let result_q_fast_mat = &(&s_t * c)
            * &gadget_inverse::<<Self as SPIRAL>::RingQFast, 2, M_GSW, 2, Z_GSW, T_GSW>(
                &scaled_ident,
            );
        let result_q = <Self as SPIRAL>::RingQ::from(&result_q_fast_mat[(0, 1)]);
        <Self as SPIRAL>::RingQ::from(result_q)
    }

    pub fn regev_sub_hom(
        lhs: &<Self as SPIRAL>::RegevCiphertext,
        rhs: &<Self as SPIRAL>::RegevCiphertext,
    ) -> <Self as SPIRAL>::RegevCiphertext {
        lhs - rhs
    }

    pub fn hybrid_mul_hom(
        regev: &<Self as SPIRAL>::RegevCiphertext,
        gsw: &<Self as SPIRAL>::GSWCiphertext,
    ) -> <Self as SPIRAL>::RegevCiphertext {
        gsw * &gadget_inverse::<<Self as SPIRAL>::RingQFast, 2, M_GSW, 1, Z_GSW, T_GSW>(regev)
    }

    pub fn regev_mul_x_pow(
        c: &<Self as SPIRAL>::RegevCiphertext,
        k: usize,
    ) -> <Self as SPIRAL>::RegevCiphertext {
        let mut result = Matrix::zero();
        result[(0, 0)] = c[(0, 0)].mul_x_pow(k);
        result[(1, 0)] = c[(1, 0)].mul_x_pow(k);
        result
    }

    pub fn auto_setup<const LEN: usize, const BASE: u64>(
        tau_power: usize,
        s_encode: &<Self as SPIRAL>::RingQFast,
    ) -> <Self as SPIRAL>::AutoKey<LEN> {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, LEN, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_t: Matrix<1, LEN, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let mut bottom = &a_t * s_encode;
        bottom += &e_t;
        bottom -= &(&build_gadget::<<Self as SPIRAL>::RingQFast, 1, LEN, BASE, LEN>()
            * &s_encode.auto(tau_power));
        (Matrix::stack(&a_t, &bottom), tau_power)
    }

    pub fn auto_hom<const LEN: usize, const BASE: u64>(
        (w_mat, tau_power): &<Self as SPIRAL>::AutoKey<LEN>,
        c: &<Self as SPIRAL>::RegevCiphertext,
    ) -> <Self as SPIRAL>::RegevCiphertext {
        let c0 = &c[(0, 0)];
        let c1 = &c[(1, 0)];
        let mut g_inv_tau_c0 = Matrix::<LEN, 1, <Self as SPIRAL>::RingQFast>::zero();

        // 80% + 1%
        <<Self as SPIRAL>::RingQFast as RingElementDecomposable<BASE, LEN>>::decompose_into_mat(
            &c0.auto(*tau_power),
            &mut g_inv_tau_c0,
            0,
            0,
        );

        // 8%
        let mut result = w_mat * &g_inv_tau_c0;

        // 1%
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
        cts: &[<Self as SPIRAL>::RegevCiphertext],
        auto_key: &<Self as SPIRAL>::AutoKey<LEN>,
    ) -> Vec<<Self as SPIRAL>::RegevCiphertext> {
        debug_assert_eq!(auto_key.1, D / (1 << which_iter) + 1);
        let len = cts.len();
        let mut cts_new = Vec::with_capacity(2 * len);
        cts_new.resize(2 * len, Matrix::zero());
        for (j, ct) in cts.iter().enumerate() {
            let shift_exp = (1 << which_iter);
            let shift_auto_exp = (shift_exp * auto_key.1) % (2 * D);

            let ct_shifted = Self::regev_mul_x_pow(ct, 2 * D - shift_exp);
            let ct_auto = Self::auto_hom::<LEN, BASE>(auto_key, ct);
            let ct_auto_shifted = Self::regev_mul_x_pow(&ct_auto, 2 * D - shift_auto_exp);

            cts_new[j] = ct + &ct_auto;
            cts_new[j + len] = &ct_shifted + &ct_auto_shifted;
        }
        cts_new
    }

    pub fn regev_to_gsw_setup(
        s_encode: &<Self as SPIRAL>::EncodingKey,
    ) -> <Self as SPIRAL>::RegevToGSWKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t = Matrix::<1, M_CONV, <Self as SPIRAL>::RingQFast>::rand_uniform(&mut rng);
        let e_mat = Matrix::<1, M_CONV, <Self as SPIRAL>::RingQFast>::rand_discrete_gaussian::<
            _,
            NOISE_WIDTH_MILLIONTHS,
        >(&mut rng);
        let mut bottom = &a_t * s_encode;
        bottom += &e_mat;
        let g_vec = build_gadget::<<Self as SPIRAL>::RingQFast, 1, T_CONV, Z_CONV, T_CONV>();
        let mut s_encode_tensor_g = Matrix::<1, M_CONV, <Self as SPIRAL>::RingQFast>::zero();
        s_encode_tensor_g.copy_into(&g_vec, 0, T_CONV);
        s_encode_tensor_g.copy_into(&(&g_vec * &(-s_encode)), 0, 0);
        bottom -= &(&s_encode_tensor_g * s_encode);

        Matrix::stack(&a_t, &bottom)
    }

    pub fn regev_to_gsw(
        v_mat: &<Self as SPIRAL>::RegevToGSWKey,
        cs: &[<Self as SPIRAL>::RegevCiphertext],
    ) -> <Self as SPIRAL>::GSWCiphertext {
        let mut result = Matrix::<2, M_GSW, <Self as SPIRAL>::RingQFast>::zero();
        let mut c_hat = Matrix::<2, T_GSW, <Self as SPIRAL>::RingQFast>::zero();
        for (i, ci) in cs.iter().enumerate() {
            c_hat.copy_into(ci, 0, i);
        }
        let g_inv_c_hat =
            gadget_inverse::<<Self as SPIRAL>::RingQFast, 2, M_CONV, T_GSW, Z_CONV, T_CONV>(&c_hat);
        let v_g_inv_c_hat = v_mat * &g_inv_c_hat;
        result.copy_into(&v_g_inv_c_hat, 0, 0);
        for (i, ci) in cs.iter().enumerate() {
            result.copy_into(ci, 0, T_GSW + i);
        }

        // No permutation needed for scalar regev
        result
    }

    pub fn key_switch_setup(
        s_from: &IntModCycloEval<D, Q_SWITCH2>,
        s_to: &IntModCycloEval<D, Q_SWITCH2>,
    ) -> <Self as SPIRAL>::KeySwitchKey {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t = Matrix::<1, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>::rand_uniform(&mut rng);
        let e_t = Matrix::<1, T_SWITCH, IntModCycloEval<D, Q_SWITCH2>>::rand_discrete_gaussian::<
            _,
            NOISE_WIDTH_MILLIONTHS,
        >(&mut rng);
        let mut b_t =
            &build_gadget::<IntModCycloEval<D, Q_SWITCH2>, 1, T_SWITCH, Z_SWITCH, T_SWITCH>()
                * &(-s_from);
        b_t += &(&a_t * s_to);
        b_t += &e_t;
        (a_t, b_t)
    }
}
