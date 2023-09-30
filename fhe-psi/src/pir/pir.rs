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
use crate::math::ring_elem::{NormedRingElement, RingCompatible, RingElement};
use crate::math::utils::{ceil_log, mod_inverse};

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
        let erfc_inv = 5.7458723921911804703334813888123646139061858840406015270555555247_f64;
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
    fn answer(
        pp: &Self::PublicParams,
        db: &[Self::RecordPreprocessed],
        q: &Self::Query,
    ) -> Self::Response;
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
    type RecordPreprocessed = Self::RingQFast;

    // Constants
    const DB_SIZE: usize = 2_usize.pow(ETA1 as u32) * Z_FOLD.pow(ETA2 as u32);
    const ETA1: usize = ETA1;
    const ETA2: usize = ETA2;

    const REGEV_COUNT: usize = 1 << ETA1;
    const REGEV_EXPAND_ITERS: usize = ETA1;
    const GSW_COUNT: usize = ETA2 * (Z_FOLD - 1) * T_GSW;
    const GSW_EXPAND_ITERS: usize = ceil_log(2, Self::GSW_COUNT as u64);

    fn preprocess(record: &<Self as SPIRAL>::Record) -> <Self as SPIRAL>::RecordPreprocessed {
        <Self as SPIRAL>::RingQFast::from(&record.include_into::<Q>())
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
        Self::encode_regev(s_encode, &mu)
    }

    fn answer(
        pp: &<Self as SPIRAL>::PublicParams,
        db: &[<Self as SPIRAL>::RecordPreprocessed],
        q: &<Self as SPIRAL>::Query,
    ) -> <Self as SPIRAL>::Response {
        // Query expansion
        let (regevs, gsws) = Self::answer_query_expand(pp, q);

        let start = Instant::now();
        // First dimension
        let first_dim_folded = Self::answer_first_dim(db, &regevs);
        let mid = Instant::now();

        // Folding
        let result = Self::answer_fold(first_dim_folded, gsws.as_slice());

        let end = Instant::now();
        eprintln!("(*) answer first dim: {:?}", mid - start);
        eprintln!("(*) answer fold: {:?}", end - mid);
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

        for e in diff.coeff_iter() {
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
        regevs.truncate(1 << Self::ETA1);

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

        (regevs, gsws)
    }

    pub fn answer_first_dim(
        db: &[<Self as SPIRAL>::RecordPreprocessed],
        regevs: &[<Self as SPIRAL>::RegevCiphertext],
    ) -> Vec<<Self as SPIRAL>::RegevCiphertext> {
        assert_eq!(regevs.len(), 1 << ETA1);
        let fold_size: usize = Z_FOLD.pow(Self::ETA2 as u32);

        // First dimension processing
        let db_at = |i: usize, j: usize| &db[i * fold_size + j];
        let mut curr: Vec<<Self as SPIRAL>::RegevCiphertext> = Vec::with_capacity(fold_size);
        for j in 0..fold_size {
            // Norm is at most max(Q_A, Q_B)^2 for each term
            // Add one for margin
            let reduce_every = 1 << (64 - 2 * ceil_log(2, max(Q_A, Q_B)) - 1);
            let mut sum = Matrix::zero();
            for i in 0..(1 << ETA1) {
                Self::regev_add_eq_mul_scalar_no_reduce(&mut sum, &regevs[i], db_at(i, j));
                if i % reduce_every == 0 {
                    sum.iter_do(|r| <Self as SPIRAL>::RingQFast::reduce_mod(r));
                }
            }
            sum.iter_do(|r| <Self as SPIRAL>::RingQFast::reduce_mod(r));
            curr.push(sum.convert_ring());
        }

        curr
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

    pub fn regev_add_eq_mul_scalar_no_reduce(
        lhs: &mut <Self as SPIRAL>::RegevCiphertext0,
        rhs_a: &<Self as SPIRAL>::RegevCiphertext,
        rhs_b: &<Self as SPIRAL>::RingQFast,
    ) {
        lhs[(0, 0)].add_eq_mul(rhs_a[(0, 0)].convert_ref(), rhs_b.convert_ref());
        lhs[(1, 0)].add_eq_mul(rhs_a[(1, 0)].convert_ref(), rhs_b.convert_ref());
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
        s_scalar: &<Self as SPIRAL>::RingQFast,
    ) -> <Self as SPIRAL>::AutoKey<LEN> {
        let mut rng = ChaCha20Rng::from_entropy();
        let a_t: Matrix<1, LEN, <Self as SPIRAL>::RingQFast> = Matrix::rand_uniform(&mut rng);
        let e_t: Matrix<1, LEN, <Self as SPIRAL>::RingQFast> =
            Matrix::rand_discrete_gaussian::<_, NOISE_WIDTH_MILLIONTHS>(&mut rng);
        let mut bottom = &a_t * s_scalar;
        bottom += &e_t;
        bottom -= &(&build_gadget::<<Self as SPIRAL>::RingQFast, 1, LEN, BASE, LEN>()
            * &s_scalar.auto(tau_power));
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
        let mut s_scalar_tensor_g = Matrix::<1, M_CONV, <Self as SPIRAL>::RingQFast>::zero();
        s_scalar_tensor_g.copy_into(&g_vec, 0, T_CONV);
        s_scalar_tensor_g.copy_into(&(&g_vec * &(-s_encode)), 0, 0);
        bottom -= &(&s_scalar_tensor_g * s_encode);

        let result = Matrix::stack(&a_t, &bottom);
        result
    }

    pub fn regev_to_gsw(
        v_mat: &<Self as SPIRAL>::RegevToGSWKey,
        cs: &[<Self as SPIRAL>::RegevCiphertext],
    ) -> <Self as SPIRAL>::GSWCiphertext {
        let mut result = Matrix::<2, M_GSW, <Self as SPIRAL>::RingQFast>::zero();
        let mut c_hat = Matrix::<2, T_GSW, <Self as SPIRAL>::RingQFast>::zero();
        for i in 0..T_GSW {
            c_hat.copy_into(&cs[i], 0, i);
        }
        let g_inv_c_hat =
            gadget_inverse::<<Self as SPIRAL>::RingQFast, 2, M_CONV, T_GSW, Z_CONV, T_CONV>(&c_hat);
        let v_g_inv_c_hat = v_mat * &g_inv_c_hat;
        result.copy_into(&v_g_inv_c_hat, 0, 0);
        for i in 0..T_GSW {
            result.copy_into(&cs[i], 0, T_GSW + i);
        }

        // No permutation needed for scalar regev
        result
    }
}

#[cfg(test)]
mod test {
    use std::time::Instant;

    use crate::math::int_mod_poly::IntModPoly;
    use rand::Rng;

    use super::*;

    const SPIRAL_TEST_PARAMS: SPIRALParams = SPIRALParamsRaw {
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        T_GSW: 8,
        T_CONV: 4,
        T_COEFF_REGEV: 8,
        T_COEFF_GSW: 56,
        // Z_GSW: 75,
        // Z_COEFF_REGEV: 127,
        // Z_COEFF_GSW: 2,
        // Z_CONV: 16088,
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
        let s = SPIRALTest::encode_setup();
        let mu = <SPIRALTest as SPIRAL>::RingP::from(12_u64);
        let encoded = SPIRALTest::encode_regev(&s, &mu.scale_up_into());
        let decoded: <SPIRALTest as SPIRAL>::RingP =
            SPIRALTest::decode_regev(&s, &encoded).round_down_into();
        assert_eq!(mu, decoded);
    }

    #[test]
    fn test_gsw() {
        let s = SPIRALTest::encode_setup();
        type RingPP = IntModCyclo<{ SPIRAL_TEST_PARAMS.D }, 1024>;
        let mu = RingPP::from(111_u64);
        let encrypt = SPIRALTest::encode_gsw(&s, &mu.include_into());

        let scale = <SPIRALTest as SPIRAL>::RingQFast::from(SPIRAL_TEST_PARAMS.Q / 1024);
        let decrypt = SPIRALTest::decode_gsw_scaled(&s, &encrypt, &scale);
        assert_eq!(decrypt.round_down_into(), mu);
    }

    #[test]
    fn test_auto_hom() {
        let s = SPIRALTest::encode_setup();
        let auto_key = SPIRALTest::auto_setup::<
            { SPIRAL_TEST_PARAMS.T_COEFF_REGEV },
            { SPIRAL_TEST_PARAMS.Z_COEFF_REGEV },
        >(3, &s);
        let x = <SPIRALTest as SPIRAL>::RingP::from(IntModPoly::x());
        let encrypt = SPIRALTest::encode_regev(&s, &x.scale_up_into());
        let encrypt_auto = SPIRALTest::auto_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_REGEV },
            { SPIRAL_TEST_PARAMS.Z_COEFF_REGEV },
        >(&auto_key, &encrypt);
        let decrypt: <SPIRALTest as SPIRAL>::RingP =
            SPIRALTest::decode_regev(&s, &encrypt_auto).round_down_into();
        assert_eq!(decrypt, &(&x * &x) * &x);
    }

    #[test]
    fn test_regev_to_gsw() {
        let s = SPIRALTest::encode_setup();
        let s_regev_to_gsw = SPIRALTest::regev_to_gsw_setup(&s);
        type RingPP = IntModCyclo<{ SPIRAL_TEST_PARAMS.D }, 1024>;
        let mu = RingPP::from(567_u64);
        let mut mu_curr = mu.include_into();
        let mut encrypt_vec = Vec::with_capacity(SPIRAL_TEST_PARAMS.T_GSW);
        for _ in 0..SPIRAL_TEST_PARAMS.T_GSW {
            encrypt_vec.push(SPIRALTest::encode_regev(&s, &mu_curr));
            mu_curr *= IntMod::from(SPIRAL_TEST_PARAMS.Z_GSW);
        }
        let encrypt_gsw = SPIRALTest::regev_to_gsw(&s_regev_to_gsw, encrypt_vec.as_slice());

        let scale = <SPIRALTest as SPIRAL>::RingQFast::from(SPIRAL_TEST_PARAMS.Q / 1024);
        let decrypted = SPIRALTest::decode_gsw_scaled(&s, &encrypt_gsw, &scale);
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

    #[cfg(not(target_feature = "avx2"))]
    fn has_avx2() -> bool {
        false
    }

    #[cfg(target_feature = "avx2")]
    fn has_avx2() -> bool {
        true
    }

    fn run_spiral<TheSPIRAL: SPIRAL<Record = IntModCyclo<2048, 256>>, I: Iterator<Item = usize>>(
        iter: I,
    ) {
        eprintln!(
            "Running SPIRAL test with database size {}",
            SPIRALTest::DB_SIZE
        );
        eprintln!(
            "AVX2 is {}",
            if has_avx2() {
                "enabled"
            } else {
                "not enabled "
            }
        );
        eprintln!("Parameters: {:#?}", SPIRAL_TEST_PARAMS);
        let mut db: Vec<<TheSPIRAL as SPIRAL>::Record> = Vec::with_capacity(SPIRALTest::DB_SIZE);
        for i in 0..TheSPIRAL::DB_SIZE as u64 {
            let mut record_coeff = [IntMod::<256>::zero(); 2048];
            record_coeff[0] = (i % 256).into();
            record_coeff[1] = ((i / 256) % 256).into();
            record_coeff[2] = 42_u64.into();
            record_coeff[3] = 0_u64.into();
            record_coeff[4] = (i % 100).into();
            record_coeff[5] = ((i / 100) % 100).into();
            record_coeff[6] = ((i / 100 / 100) % 100).into();
            record_coeff[7] = ((i / 100 / 100 / 100) % 100).into();
            db.push(<TheSPIRAL as SPIRAL>::Record::from(record_coeff));
        }
        eprintln!(
            "Relative noise threshold: 2^({})",
            SPIRAL_TEST_PARAMS.relative_noise_threshold().log2()
        );

        eprintln!();

        let pre_start = Instant::now();
        let mut db_pre: Vec<<TheSPIRAL as SPIRAL>::RecordPreprocessed> =
            Vec::with_capacity(TheSPIRAL::DB_SIZE);
        for db_elem in db.iter() {
            db_pre.push(TheSPIRAL::preprocess(db_elem));
        }
        let pre_end = Instant::now();
        eprintln!("{:?} to preprocess", pre_end - pre_start);

        let setup_start = Instant::now();
        let (qk, pp) = TheSPIRAL::setup();
        let setup_end = Instant::now();
        eprintln!("{:?} to setup", setup_end - setup_start);

        let check = |idx: usize| {
            eprintln!("Testing record index {}", idx);
            let query_start = Instant::now();
            let q = TheSPIRAL::query(&qk, idx);
            let query_end = Instant::now();
            let query_total = query_end - query_start;

            let answer_start = Instant::now();
            let result = TheSPIRAL::answer(&pp, &db_pre, &q);
            let answer_end = Instant::now();
            let answer_total = answer_end - answer_start;

            let extract_start = Instant::now();
            let extracted = TheSPIRAL::extract(&qk, &result);
            let extract_end = Instant::now();
            let extract_total = extract_end - extract_start;

            if extracted != db[idx] {
                eprintln!("  **** **** **** **** ERROR **** **** **** ****");
                eprintln!("  protocol failed");
            }
            eprintln!("  {:?} total", query_total + answer_total + extract_total);
            eprintln!("    {:?} to query", query_total);
            eprintln!("    {:?} to answer", answer_total);
            eprintln!("    {:?} to extract", extract_total);
            let (rel_noise, correctness) = TheSPIRAL::response_stats(&qk, &result, &db[idx]);

            eprintln!(
                "  relative coefficient noise (sample): 2^({})",
                rel_noise.log2()
            );
            eprintln!(
                "  correctness probability (sample): 1 - 2^({})",
                correctness.log2()
            );
        };

        for i in iter {
            check(i);
        }
    }
}
