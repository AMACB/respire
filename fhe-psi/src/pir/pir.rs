use libm::erfc;
use std::cmp::max;
use std::slice;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::math::gadget::{base_from_len, build_gadget, gadget_inverse, RingElementDecomposable};
use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::int_mod_cyclo_crt_eval::IntModCycloCRTEval;
use crate::math::matrix::Matrix;
use crate::math::number_theory::find_sqrt_primitive_root;
use crate::math::rand_sampled::{RandDiscreteGaussianSampled, RandUniformSampled};
use crate::math::ring_elem::{NormedRingElement, RingElement};
use crate::math::utils::{ceil_log, mod_inverse};

use crate::math::simd_utils::*;

pub struct SPIRALImpl<
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
    const ETA1: usize,
    const ETA2: usize,
    const Z_FOLD: usize,
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
    pub ETA1: usize,
    pub ETA2: usize,
    pub Z_FOLD: usize,
}

impl SPIRALParamsRaw {
    pub const fn expand(&self) -> SPIRALParams {
        let q = self.Q_A * self.Q_B;
        let z_gsw = base_from_len(self.T_GSW, q);
        let z_coeff_regev = base_from_len(self.T_COEFF_REGEV, q);
        let z_coeff_gsw = base_from_len(self.T_COEFF_GSW, q);
        let z_conv = base_from_len(self.T_CONV, q);
        SPIRALParams {
            Q: q,
            Q_A: self.Q_A,
            Q_B: self.Q_B,
            Q_A_INV: mod_inverse(self.Q_A, self.Q_B),
            Q_B_INV: mod_inverse(self.Q_B, self.Q_A),
            D: self.D,
            W_A: find_sqrt_primitive_root(self.D, self.Q_A),
            W_B: find_sqrt_primitive_root(self.D, self.Q_B),
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
            ETA1: self.ETA1,
            ETA2: self.ETA2,
            Z_FOLD: self.Z_FOLD,
        }
    }
}

#[allow(non_snake_case)]
#[derive(Debug)]
pub struct SPIRALParams {
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
    pub ETA1: usize,
    pub ETA2: usize,
    pub Z_FOLD: usize,
}

impl SPIRALParams {
    pub fn relative_noise_threshold(&self) -> f64 {
        // erfc_inverse(2^-40 / D)
        assert_eq!(self.D, 2048);
        let erfc_inv = 5.745_872_392_191_18_f64;
        1_f64 / (2_f64 * (self.P as f64) * 2_f64.sqrt() * erfc_inv)
    }
}

#[macro_export]
macro_rules! spiral {
    ($params: expr) => {
        SPIRALImpl<
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
    type RegevCiphertext;
    type RegevCiphertext0;
    type GSWCiphertext;
    type EncodingKey;
    type AutoKey<const T: usize>;
    type AutoKeyRegev;
    type AutoKeyGSW;
    type RegevToGSWKey;

    // Associated types
    type QueryKey;
    type PublicParams;
    type Query;
    type QueryExpanded;
    type Response;
    type Record;
    type Database;

    // Constants
    const DB_DIM1_SIZE: usize;
    const DB_DIM2_SIZE: usize;
    const DB_SIZE: usize;
    const ETA1: usize;
    const ETA2: usize;
    const REGEV_COUNT: usize;
    const REGEV_EXPAND_ITERS: usize;
    const GSW_COUNT: usize;
    const GSW_EXPAND_ITERS: usize;

    fn preprocess<'a, I: ExactSizeIterator<Item = &'a Self::Record>>(
        records_iter: I,
    ) -> Self::Database
    where
        Self::Record: 'a;
    fn setup() -> (Self::QueryKey, Self::PublicParams);
    fn query(qk: &Self::QueryKey, idx: usize) -> Self::Query;
    fn answer(pp: &Self::PublicParams, db: &Self::Database, q: &Self::Query) -> Self::Response;
    fn extract(qk: &Self::QueryKey, r: &Self::Response) -> Self::Record;
    fn response_stats(
        qk: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
        actual: &<Self as SPIRAL>::Record,
    ) -> (f64, f64);
}

impl<
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
        const ETA1: usize,
        const ETA2: usize,
        const Z_FOLD: usize,
    > SPIRAL
    for SPIRALImpl<
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
    type RegevCiphertext = Matrix<2, 1, Self::RingQFast>;
    type RegevCiphertext0 = Matrix<2, 1, Self::Ring0Fast>;
    type GSWCiphertext = Matrix<2, M_GSW, Self::RingQFast>;

    type EncodingKey = Self::RingQFast;
    type AutoKey<const T: usize> = (Matrix<2, T, Self::RingQFast>, usize);
    type AutoKeyRegev = Self::AutoKey<T_COEFF_REGEV>;
    type AutoKeyGSW = Self::AutoKey<T_COEFF_GSW>;
    type RegevToGSWKey = Matrix<2, M_CONV, Self::RingQFast>;

    // Associated types
    type QueryKey = Self::EncodingKey;
    type PublicParams = (
        (
            Self::AutoKeyGSW,
            Vec<Self::AutoKeyRegev>,
            Vec<Self::AutoKeyGSW>,
        ),
        Self::RegevToGSWKey,
    );
    type Query = Self::RegevCiphertext;
    type QueryExpanded = (Vec<Self::RegevCiphertext>, Vec<Self::GSWCiphertext>);
    type Response = Self::RegevCiphertext;
    type Record = Self::RingP;

    /// We structure the database as `[2] x [D / S] x [DIM2_SIZE] x [DIM1_SIZE] x [S]` for optimal first dimension
    /// processing. The outermost pair is the first resp. second CRT projections, packed as two u32 into one u64;
    /// `S` is the SIMD lane count that we can use, i.e. 4 for AVX2.
    type Database = Vec<SimdVec>;

    // Constants
    const DB_DIM1_SIZE: usize = 2_usize.pow(ETA1 as u32);
    const DB_DIM2_SIZE: usize = Z_FOLD.pow(ETA2 as u32);
    const DB_SIZE: usize = Self::DB_DIM1_SIZE * Self::DB_DIM2_SIZE;
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;

    const REGEV_COUNT: usize = 1 << ETA1;
    const REGEV_EXPAND_ITERS: usize = ETA1;
    const GSW_COUNT: usize = ETA2 * (Z_FOLD - 1) * T_GSW;
    const GSW_EXPAND_ITERS: usize = ceil_log(2, Self::GSW_COUNT as u64);

    fn preprocess<'a, I: ExactSizeIterator<Item = &'a Self::Record>>(
        records_iter: I,
    ) -> Self::Database
    where
        Self::Record: 'a,
    {
        assert_eq!(records_iter.len(), Self::DB_SIZE);

        let records_eval: Vec<<Self as SPIRAL>::RingQFast> = records_iter
            .map(|r| <Self as SPIRAL>::RingQFast::from(&r.include_into::<Q>()))
            .collect();

        #[cfg(not(target_feature = "avx2"))]
        {
            let mut db: Vec<SimdVec> = (0..(D * Self::DB_SIZE)).map(|_| 0_u64).collect();
            for eval_vec_idx in 0..D {
                for db_idx in 0..Self::DB_SIZE {
                    // Transpose the index
                    let (db_i, db_j) = (db_idx / Self::DB_DIM2_SIZE, db_idx % Self::DB_DIM2_SIZE);
                    let db_idx_t = db_j * Self::DB_DIM1_SIZE + db_i;

                    let to_idx = eval_vec_idx * Self::DB_SIZE + db_idx_t;
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
            let mut db: Vec<SimdVec> = (0..((D / SIMD_LANES) * Self::DB_SIZE))
                .map(|_| Aligned32([0_u64; 4]))
                .collect();

            for eval_vec_idx in 0..(D / SIMD_LANES) {
                for db_idx in 0..Self::DB_SIZE {
                    // Transpose the index
                    let (db_i, db_j) = (db_idx / Self::DB_DIM2_SIZE, db_idx % Self::DB_DIM2_SIZE);
                    let db_idx_t = db_j * Self::DB_DIM1_SIZE + db_i;

                    let mut db_vec: SimdVec = Aligned32([0_u64; 4]);
                    for lane in 0..SIMD_LANES {
                        let from_idx = eval_vec_idx * SIMD_LANES + lane;
                        let lo = u64::from(records_eval[db_idx].proj1.evals[from_idx]);
                        let hi = u64::from(records_eval[db_idx].proj2.evals[from_idx]);
                        db_vec.0[lane] = (hi << 32) | lo;
                    }

                    let to_idx = eval_vec_idx * Self::DB_SIZE + db_idx_t;
                    db[to_idx] = db_vec;
                }
            }

            db
        }
    }

    fn setup() -> (<Self as SPIRAL>::QueryKey, <Self as SPIRAL>::PublicParams) {
        // Regev/GSW secret key
        let s_encode = Self::encode_setup();

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
            s_encode,
            (
                (auto_key_first, auto_keys_regev, auto_keys_gsw),
                regev_to_gsw_key,
            ),
        )
    }

    fn query(s_encode: &<Self as SPIRAL>::QueryKey, idx: usize) -> <Self as SPIRAL>::Query {
        assert!(idx < Self::DB_SIZE);
        let (idx_i, idx_j) = (idx / Self::DB_DIM2_SIZE, idx % Self::DB_DIM2_SIZE);

        let mut packed_vec: Vec<IntMod<Q>> = Vec::with_capacity(D);
        for _ in 0..D {
            packed_vec.push(IntMod::zero());
        }

        let inv_even = IntMod::from(mod_inverse(1 << (1 + Self::REGEV_EXPAND_ITERS), Q));
        for i in 0_usize..Self::DB_DIM1_SIZE {
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
        Self::encode_regev(s_encode, &mu)
    }

    fn answer(
        pp: &<Self as SPIRAL>::PublicParams,
        db: &<Self as SPIRAL>::Database,
        q: &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::Response {
        let i0 = Instant::now();

        // Query expansion
        let (regevs, gsws) = Self::answer_query_expand(pp, q);

        let i1 = Instant::now();

        // First dimension
        let first_dim_folded = Self::answer_first_dim(db, &regevs);

        let i2 = Instant::now();

        // Folding
        let result = Self::answer_fold(first_dim_folded, gsws.as_slice());

        let i3 = Instant::now();

        eprintln!("(*) answer query expand: {:?}", i1 - i0);
        eprintln!("(*) answer first dim: {:?}", i2 - i1);
        eprintln!("(*) answer fold: {:?}", i3 - i2);
        result
    }

    fn extract(
        s_encode: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
    ) -> <Self as SPIRAL>::Record {
        Self::decode_regev(s_encode, r).round_down_into()
    }

    fn response_stats(
        s_encode: &<Self as SPIRAL>::QueryKey,
        r: &<Self as SPIRAL>::Response,
        actual: &<Self as SPIRAL>::Record,
    ) -> (f64, f64) {
        let actual_scaled = actual.scale_up_into();
        let decoded = Self::decode_regev(s_encode, r);
        let diff = &actual_scaled - &decoded;

        let mut sum = 0_f64;
        let mut samples = 0_usize;

        for e in diff.coeff.iter() {
            let e_sq = (e.norm() as f64) * (e.norm() as f64);
            sum += e_sq;
            samples += 1;
        }

        // sigma^2 is the variance of the relative noise (noise divided by Q) of the coefficients
        let sigma = (sum / samples as f64).sqrt() / (Q as f64);

        // We want to bound the probability that a single sample of relative noise has magnitude
        // >= k. For decoding correctness, we use k = 1 / 2P. Assuming the relative noise is
        // Gaussian, this is 1 - erf( k / (sigma * sqrt(2))).
        let num_sigmas = 1_f64 / (sigma * 2_f64 * (P as f64));
        let correctness_error = erfc(num_sigmas / 2_f64.sqrt());

        // Multiply by the number of samples to union bound over all coefficients
        (sigma, (correctness_error * samples as f64).min(1_f64))
    }
}

impl<
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
        const ETA1: usize,
        const ETA2: usize,
        const Z_FOLD: usize,
    >
    SPIRALImpl<
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
        ETA1,
        ETA2,
        Z_FOLD,
    >
{
    pub fn answer_query_expand(
        ((auto_key_first, auto_keys_regev, auto_keys_gsw), regev_to_gsw_key): &<Self as SPIRAL>::PublicParams,
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
        assert_eq!(regevs.len(), Self::DB_DIM1_SIZE);

        let mut gsws = vec![gsw_base];
        for (i, auto_key_gsw) in auto_keys_gsw.iter().enumerate() {
            gsws = Self::do_coeff_expand_iter::<T_COEFF_GSW, Z_COEFF_GSW>(
                i + 1,
                gsws.as_slice(),
                auto_key_gsw,
            );
        }
        gsws.truncate(Self::ETA2 * (Z_FOLD - 1) * T_GSW);

        let gsws: Vec<<Self as SPIRAL>::GSWCiphertext> = gsws
            .chunks_exact(T_GSW)
            .map(|cs| Self::regev_to_gsw(regev_to_gsw_key, cs))
            .collect();
        assert_eq!(gsws.len(), Self::ETA2);

        (regevs, gsws)
    }

    pub fn answer_first_dim(
        db: &<Self as SPIRAL>::Database,
        regevs: &[<Self as SPIRAL>::RegevCiphertext],
    ) -> Vec<<Self as SPIRAL>::RegevCiphertext> {
        assert_eq!(regevs.len(), Self::DB_DIM1_SIZE);

        // Flatten + transpose the ciphertexts
        let mut c0s: Vec<SimdVec> = Vec::with_capacity((D / SIMD_LANES) * Self::DB_DIM1_SIZE);
        let mut c1s: Vec<SimdVec> = Vec::with_capacity((D / SIMD_LANES) * Self::DB_DIM1_SIZE);

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
        let mut result: Vec<<Self as SPIRAL>::RegevCiphertext> = (0..Self::DB_DIM2_SIZE)
            .map(|_| <Self as SPIRAL>::RegevCiphertext::zero())
            .collect();

        // Norm is at most max(Q_A, Q_B)^2 for each term
        // Add one for margin
        let reduce_every = 1 << (64 - 2 * ceil_log(2, max(Q_A, Q_B)) - 1);

        // We want to compute the sum over i of ct_i * db_(i, j).
        // Here db_(i, j) are scalars; ct_i are 2 x 1 matrices.

        #[cfg(not(target_feature = "avx2"))]
        for eval_idx in 0..D {
            for j in 0..Self::DB_DIM2_SIZE {
                let mut sum0_proj1 = 0_u64;
                let mut sum0_proj2 = 0_u64;
                let mut sum1_proj1 = 0_u64;
                let mut sum1_proj2 = 0_u64;

                for i in 0..Self::DB_DIM1_SIZE {
                    let lhs0 = c0s[eval_idx * Self::DB_DIM1_SIZE + i];
                    let lhs0_proj1 = lhs0 as u32 as u64;
                    let lhs0_proj2 = lhs0 >> 32;

                    let lhs1 = c1s[eval_idx * Self::DB_DIM1_SIZE + i];
                    let lhs1_proj1 = lhs1 as u32 as u64;
                    let lhs1_proj2 = lhs1 >> 32;

                    let rhs = db[eval_idx * Self::DB_SIZE + j * Self::DB_DIM1_SIZE + i];
                    let rhs_proj1 = rhs as u32 as u64;
                    let rhs_proj2 = rhs >> 32;

                    sum0_proj1 += lhs0_proj1 * rhs_proj1;
                    sum0_proj2 += lhs0_proj2 * rhs_proj2;
                    sum1_proj1 += lhs1_proj1 * rhs_proj1;
                    sum1_proj2 += lhs1_proj2 * rhs_proj2;

                    if i % reduce_every == 0 || i == Self::DB_DIM1_SIZE - 1 {
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
                for j in 0..Self::DB_DIM2_SIZE {
                    let mut sum0_proj1 = _mm256_setzero_si256();
                    let mut sum0_proj2 = _mm256_setzero_si256();
                    let mut sum1_proj1 = _mm256_setzero_si256();
                    let mut sum1_proj2 = _mm256_setzero_si256();

                    for i in 0..Self::DB_DIM1_SIZE {
                        let lhs0_ptr = c0s.get_unchecked(eval_vec_idx * Self::DB_DIM1_SIZE + i)
                            as *const SimdVec
                            as *const __m256i;
                        let lhs1_ptr = c1s.get_unchecked(eval_vec_idx * Self::DB_DIM1_SIZE + i)
                            as *const SimdVec
                            as *const __m256i;
                        let rhs_ptr = db.get_unchecked(
                            eval_vec_idx * Self::DB_SIZE + j * Self::DB_DIM1_SIZE + i,
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

                        if i % reduce_every == 0 || i == Self::DB_DIM1_SIZE - 1 {
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
        <<Self as SPIRAL>::RingQFast as RingElementDecomposable<BASE, LEN>>::decompose_into_mat(
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
        cts: &[<Self as SPIRAL>::RegevCiphertext],
        auto_key: &<Self as SPIRAL>::AutoKey<LEN>,
    ) -> Vec<<Self as SPIRAL>::RegevCiphertext> {
        debug_assert_eq!(auto_key.1, D / (1 << which_iter) + 1);
        let len = cts.len();
        let mut cts_new = Vec::with_capacity(2 * len);
        cts_new.resize(2 * len, Matrix::zero());
        for (j, ct) in cts.iter().enumerate() {
            let shifted = Self::regev_mul_x_pow(ct, 2 * D - (1 << which_iter));
            cts_new[j] = ct + &Self::auto_hom::<LEN, BASE>(auto_key, ct);
            cts_new[j + len] = &shifted + &Self::auto_hom::<LEN, BASE>(auto_key, &shifted);
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
}
