use crate::pir::cuckoo_respire::CuckooRespireImpl;
use crate::pir::pir::PIR;
use crate::pir::respire::{RecordBytesImpl, RespireImpl, RespireParams, RespireParamsExpanded};
use crate::respire;
use itertools::Itertools;
use log::info;
use std::time::Instant;

pub const fn respire_512(n_vec: usize, batch_size: usize) -> RespireParamsExpanded {
    RespireParams {
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        T_GSW: 8,
        T_CONV: 4,
        T_COEFF_REGEV: 4,
        T_COEFF_GSW: 16,
        BATCH_SIZE: batch_size,
        N_VEC: n_vec,
        T_SCAL_TO_VEC: 8,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 17,
        D_RECORD: 512,
        NU1: 9,
        NU2: 9,
        Z_FOLD: 2,
        Q_SWITCH1: 6 * 17,
        Q_SWITCH2: 163841, // 17.32 bits
        D_SWITCH: 512,
        T_SWITCH: 18,
    }
    .expand()
}

pub const fn respire_1024(n_vec: usize, batch_size: usize) -> RespireParamsExpanded {
    RespireParams {
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        T_GSW: 8,
        T_CONV: 4,
        T_COEFF_REGEV: 4,
        T_COEFF_GSW: 16,
        BATCH_SIZE: batch_size,
        N_VEC: n_vec,
        T_SCAL_TO_VEC: 8,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 257,
        D_RECORD: 256,
        NU1: 9,
        NU2: 8,
        Z_FOLD: 2,
        Q_SWITCH1: 6 * 257,
        Q_SWITCH2: 4169729, // 21.99 bits
        D_SWITCH: 1024,
        T_SWITCH: 22,
    }
    .expand()
}
// TODO: investigate noise issue here
// pub const RESPIRE_TEST_PARAMS: RespireParams = respire_1024(1, 1);
// pub const RESPIRE_TEST_PARAMS: RespireParams = respire_1024(1, 4);

pub const fn respire_1024_b32_base() -> RespireParamsExpanded {
    let mut params = respire_1024(7, 49);
    params.NU1 -= 1;
    params.NU2 -= 3;
    params
}

pub const fn respire_1024_b256_base() -> RespireParamsExpanded {
    let mut params = respire_1024(7, 398);
    params.NU1 -= 2;
    params.NU2 -= 5;
    params
}

pub const RESPIRE_TEST_PARAMS: RespireParamsExpanded = respire_512(1, 1);
pub type RespireTest = respire!(RESPIRE_TEST_PARAMS);

pub const RESPIRE_SINGLE_RECORD_PARAMS: RespireParamsExpanded = respire_512(1, 1);
pub type RespireSingleRecord = respire!(RESPIRE_SINGLE_RECORD_PARAMS);

pub const RESPIRE_BATCH32_BASE_PARAMS: RespireParamsExpanded = respire_1024_b32_base();
pub type RespireBatch32Base = respire!(RESPIRE_BATCH32_BASE_PARAMS);
pub type RespireCuckoo32 = CuckooRespireImpl<32, 49, { 2usize.pow(20) }, RespireBatch32Base>;

pub const RESPIRE_BATCH256_BASE_PARAMS: RespireParamsExpanded = respire_1024_b256_base();
pub type RespireBatch256Base = respire!(RESPIRE_BATCH256_BASE_PARAMS);
pub type RespireCuckoo256 = CuckooRespireImpl<256, 398, { 2usize.pow(20) }, RespireBatch256Base>;

#[cfg(not(target_feature = "avx2"))]
pub fn has_avx2() -> bool {
    false
}

#[cfg(target_feature = "avx2")]
pub fn has_avx2() -> bool {
    true
}

// TODO encapsulate stats into struct instead of printing directly
// struct RunResult {
//     success: bool,
//     noise: f64,
//     // preprocess_time: Duration,
//     // setup_time: Duration,
//     query_time: Duration,
//     answer_time: Duration,
//     extract_time: Duration,
// }

pub fn run_pir<ThePIR: PIR<RecordBytes = RecordBytesImpl<256>>, I: Iterator<Item = usize>>(
    iter: I,
) {
    eprintln!("Running PIR...");
    eprintln!(
        "AVX2 is {}",
        if has_avx2() {
            "enabled"
        } else {
            "not enabled "
        }
    );
    eprintln!("========");
    ThePIR::print_summary();
    eprintln!("========");

    let mut records: Vec<RecordBytesImpl<256>> = Vec::with_capacity(ThePIR::NUM_RECORDS);
    for i in 0..ThePIR::NUM_RECORDS as u64 {
        let mut record = [0_u8; 256];
        record[0] = (i % 256) as u8;
        record[1] = ((i / 256) % 256) as u8;
        record[2] = 42_u8;
        record[3] = 0_u8;
        record[4] = (i % 100) as u8;
        record[5] = ((i / 100) % 100) as u8;
        record[6] = ((i / 100 / 100) % 100) as u8;
        record[7] = ((i / 100 / 100 / 100) % 100) as u8;
        // for i in 8..256 {
        //     record[i] = random();
        // }
        records.push(RecordBytesImpl { it: record });
    }

    // FIXME: both of these are out of date. The former is replaced by the jupyter notebook though?

    // eprintln!(
    //     "Estimated relative noise: 2^({})",
    //     RESPIRE_TEST_PARAMS.noise_estimate().log2()
    // );

    // eprintln!(
    //     "Relative noise threshold: 2^({})",
    //     RESPIRE_TEST_PARAMS.relative_noise_threshold().log2()
    // );

    let pre_start = Instant::now();
    let (db, db_hint) = ThePIR::encode_db(records.iter().cloned());
    let pre_end = Instant::now();
    info!("{:?} to preprocess", pre_end - pre_start);

    let setup_start = Instant::now();
    let (qk, pp) = ThePIR::setup();
    let setup_end = Instant::now();
    info!("{:?} to setup", setup_end - setup_start);

    let check = |indices: &[usize]| {
        eprintln!("Testing record indices {:?}", &indices);
        assert_eq!(indices.len(), ThePIR::BATCH_SIZE);
        let query_start = Instant::now();
        let (q, st) = ThePIR::query(&qk, indices, &db_hint);
        let query_end = Instant::now();
        let query_total = query_end - query_start;

        let answer_start = Instant::now();
        let response = ThePIR::answer(&pp, &db, &q, Some(&qk));
        let answer_end = Instant::now();
        let answer_total = answer_end - answer_start;

        let extract_start = Instant::now();
        let extracted = ThePIR::extract(&qk, &response, &st);
        let extract_end = Instant::now();
        let extract_total = extract_end - extract_start;

        for (idx, decoded_record) in indices.iter().copied().zip(extracted) {
            if decoded_record != records[idx] {
                eprintln!("**** **** **** **** ERROR **** **** **** ****");
                eprintln!("protocol failed");
                dbg!(idx);
                dbg!(&decoded_record);
                dbg!(&records[idx]);
            }
        }

        info!("{:?} to query", query_total);
        info!("{:?} to answer", answer_total);
        info!("{:?} to extract", extract_total);

        eprintln!("{:?} total", query_total + answer_total + extract_total);
    };

    for chunk in iter.chunks(ThePIR::BATCH_SIZE).into_iter() {
        let c_vec = chunk.collect_vec();
        check(c_vec.as_slice());
    }
}

#[macro_export]
macro_rules! generate_main {
    ($name: path) => {
        type ThePIR = $name;
        fn main() {
            use rand::{Rng, SeedableRng};
            use rand_chacha::ChaCha20Rng;
            use $crate::pir::example::run_pir;
            use $crate::pir::pir::PIR;
            env_logger::init();
            let mut rng = ChaCha20Rng::from_entropy();
            run_pir::<ThePIR, _>((0..).map(|_| rng.gen_range(0_usize..ThePIR::NUM_RECORDS)));
        }
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod::IntMod;
    use crate::math::int_mod_cyclo::IntModCyclo;
    use crate::math::int_mod_poly::IntModPoly;
    use crate::math::matrix::Matrix;
    use crate::math::ring_elem::RingElement;
    use crate::pir::respire::Respire;

    #[test]
    fn test_regev() {
        let s = RespireTest::encode_setup();
        let mu = <RespireTest as Respire>::RingP::from(12_u64);
        let encoded = RespireTest::encode_regev(&s, &mu.scale_up_into());
        let decoded: <RespireTest as Respire>::RingP =
            RespireTest::decode_regev(&s, &encoded).round_down_into();
        assert_eq!(mu, decoded);
    }

    #[test]
    fn test_gsw() {
        let s = RespireTest::encode_setup();
        type RingPP = IntModCyclo<{ RESPIRE_TEST_PARAMS.D }, 1024>;
        let mu = RingPP::from(111_u64);
        let encrypt = RespireTest::encode_gsw(&s, &mu.include_into());

        let scale = <RespireTest as Respire>::RingQFast::from(RESPIRE_TEST_PARAMS.Q / 1024);
        let decrypt = RespireTest::decode_gsw_scaled(&s, &encrypt, &scale);
        assert_eq!(decrypt.round_down_into(), mu);
    }

    #[test]
    fn test_auto_hom() {
        let s = RespireTest::encode_setup();
        let auto_key = RespireTest::auto_setup::<
            { RESPIRE_TEST_PARAMS.T_COEFF_REGEV },
            { RESPIRE_TEST_PARAMS.Z_COEFF_REGEV },
        >(3, &s);
        let x = <RespireTest as Respire>::RingP::from(IntModPoly::x());
        let encrypt = RespireTest::encode_regev(&s, &x.scale_up_into());
        let encrypt_auto = RespireTest::auto_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_REGEV },
            { RESPIRE_TEST_PARAMS.Z_COEFF_REGEV },
        >(&auto_key, &encrypt);
        let decrypt: <RespireTest as Respire>::RingP =
            RespireTest::decode_regev(&s, &encrypt_auto).round_down_into();
        assert_eq!(decrypt, &(&x * &x) * &x);
    }

    #[test]
    fn test_regev_to_gsw() {
        let s = RespireTest::encode_setup();
        let s_regev_to_gsw = RespireTest::regev_to_gsw_setup(&s);
        type RingPP = IntModCyclo<{ RESPIRE_TEST_PARAMS.D }, 1024>;
        let mu = RingPP::from(567_u64);
        let mut mu_curr = mu.include_into();
        let mut encrypt_vec = Vec::with_capacity(RESPIRE_TEST_PARAMS.T_GSW);
        for _ in 0..RESPIRE_TEST_PARAMS.T_GSW {
            encrypt_vec.push(RespireTest::encode_regev(&s, &mu_curr));
            mu_curr *= IntMod::from(RESPIRE_TEST_PARAMS.Z_GSW);
        }
        let encrypt_gsw = RespireTest::regev_to_gsw(&s_regev_to_gsw, encrypt_vec.as_slice());

        let scale = <RespireTest as Respire>::RingQFast::from(RESPIRE_TEST_PARAMS.Q / 1024);
        let decrypted = RespireTest::decode_gsw_scaled(&s, &encrypt_gsw, &scale);
        assert_eq!(decrypted.round_down_into(), mu);
    }

    #[test]
    fn test_project_hom() {
        let s = RespireTest::encode_setup();
        let gsw0 = RespireTest::encode_gsw(&s, &IntModCyclo::from(0_u64));
        let gsw1 = RespireTest::encode_gsw(&s, &IntModCyclo::from(1_u64));
        let mut msg_coeff = IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::zero();
        msg_coeff.coeff[0] = IntMod::from(10_u64);
        msg_coeff.coeff[1] = IntMod::from(20_u64);
        msg_coeff.coeff[2] = IntMod::from(11_u64);
        msg_coeff.coeff[3] = IntMod::from(21_u64);
        let c = RespireTest::encode_regev(&s, &msg_coeff.scale_up_into());

        let auto_key0 = RespireTest::auto_setup::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(RESPIRE_TEST_PARAMS.D + 1, &s);
        let auto_key1 = RespireTest::auto_setup::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(RESPIRE_TEST_PARAMS.D / 2 + 1, &s);

        let c_proj0 = RespireTest::project_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(0, &c, &gsw0, &auto_key0);
        let c_proj1 = RespireTest::project_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(0, &c, &gsw1, &auto_key0);

        let c_proj00 = RespireTest::project_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj0, &gsw0, &auto_key1);
        let c_proj01 = RespireTest::project_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj0, &gsw1, &auto_key1);
        let c_proj10 = RespireTest::project_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj1, &gsw0, &auto_key1);
        let c_proj11 = RespireTest::project_hom::<
            { RESPIRE_TEST_PARAMS.T_COEFF_GSW },
            { RESPIRE_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj1, &gsw1, &auto_key1);

        // let proj0 = RespireTest::decode_regev(&s, &c_proj0).round_down_into();
        let mut proj0_expected = IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::zero();
        proj0_expected.coeff[0] = IntMod::from(10_u64 * 2);
        proj0_expected.coeff[2] = IntMod::from(11_u64 * 2);

        let mut proj1_expected = IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::zero();
        proj1_expected.coeff[0] = IntMod::from(20_u64 * 2);
        proj1_expected.coeff[2] = IntMod::from(21_u64 * 2);

        assert_eq!(
            RespireTest::decode_regev(&s, &c_proj0).round_down_into(),
            proj0_expected
        );
        assert_eq!(
            RespireTest::decode_regev(&s, &c_proj1).round_down_into(),
            proj1_expected
        );

        assert_eq!(
            RespireTest::decode_regev(&s, &c_proj00).round_down_into(),
            IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::from(10_u64 * 4)
        );
        assert_eq!(
            RespireTest::decode_regev(&s, &c_proj01).round_down_into(),
            IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::from(11_u64 * 4)
        );
        assert_eq!(
            RespireTest::decode_regev(&s, &c_proj10).round_down_into(),
            IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::from(20_u64 * 4)
        );
        assert_eq!(
            RespireTest::decode_regev(&s, &c_proj11).round_down_into(),
            IntModCyclo::<{ RESPIRE_TEST_PARAMS.D }, 256>::from(21_u64 * 4)
        );
    }

    #[test]
    fn test_scal_to_vec() {
        let s_scal = RespireTest::encode_setup();
        let s_vec = RespireTest::encode_vec_setup();
        let s_scal_to_vec = RespireTest::scal_to_vec_setup(&s_scal, &s_vec);

        let mut cs = Vec::<<RespireTest as Respire>::RegevCiphertext>::with_capacity(
            RESPIRE_TEST_PARAMS.N_VEC,
        );
        let mut expected =
            Matrix::<{ RESPIRE_TEST_PARAMS.N_VEC }, 1, <RespireTest as Respire>::RingP>::zero();
        for i in 0..RESPIRE_TEST_PARAMS.N_VEC {
            let mu = <RespireTest as Respire>::RingP::from(i as u64 + 1_u64);
            expected[(i, 0)] = mu.clone();
            cs.push(RespireTest::encode_regev(&s_scal, &mu.scale_up_into()));
        }

        let c_vec = RespireTest::scal_to_vec(&s_scal_to_vec, cs.as_slice().try_into().unwrap());
        let decoded = RespireTest::decode_vec_regev(&s_vec, &c_vec);
        let actual = decoded.map_ring(|r| r.round_down_into());
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_post_process_only() {
        let (qk, pp) = RespireTest::setup();
        let (_, s_vec, _) = &qk;
        let mut m = <RespireTest as Respire>::RecordPackedSmall::zero();
        for i in 0..RESPIRE_TEST_PARAMS.N_VEC {
            m[(i, 0)] = IntModCyclo::from(177_u64 + i as u64)
        }
        let c =
            RespireTest::encode_vec_regev(s_vec, &m.map_ring(|r| r.include_dim().scale_up_into()));
        let compressed = RespireTest::answer_compress_vec(&pp, &c, RESPIRE_TEST_PARAMS.N_VEC);
        let extracted = RespireTest::extract_ring_one(&qk, &compressed);
        assert_eq!(m, extracted);
    }

    // #[test]
    // fn test_encode_decode() {
    //     // 16 x (log2(9) = 3 bits) <=> 6 bytes
    //     let bytes = [48_u8, 47, 17, 255, 183, 0];
    //     // 00110000 00101111 00010001 11111111 10110111 00000000
    //     // 001 100 000 010 111 100 010 001 111 111 111 011 011 100 000 000
    //     let encoded = RespireTest::encode_record(&bytes);
    //     assert_eq!(
    //         encoded,
    //         IntModCyclo::from(
    //             [1_u64, 4, 0, 2, 7, 4, 2, 1, 7, 7, 7, 3, 3, 4, 0, 0].map(IntMod::from)
    //         )
    //     );
    //     let decoded = RespireTest::decode_record(&encoded);
    //     assert_eq!(bytes, decoded);
    // }

    #[test]
    fn test_respire_one() {
        run_pir::<RespireTest, _>([711_711].into_iter());
    }
}
