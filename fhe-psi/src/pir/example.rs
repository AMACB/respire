use crate::pir::pir::{SPIRALImpl, SPIRALParams, SPIRALParamsRaw, SPIRAL};
use crate::spiral;
use itertools::Itertools;
use std::time::Instant;

pub const fn spiral_16_512(n_vec: usize) -> SPIRALParams {
    SPIRALParamsRaw {
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        T_GSW: 8,
        T_CONV: 4,
        T_COEFF_REGEV: 8,
        T_COEFF_GSW: 16,
        N_PACK: 1,
        N_VEC: n_vec,
        T_SCAL_TO_VEC: 8,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 17,
        D_RECORD: 512,
        ETA1: 9,
        ETA2: 7,
        Z_FOLD: 2,
        Q_SWITCH1: 4 * 17, // 4P
        Q_SWITCH2: 114689, // 17 bit prime
        D_SWITCH: 512,
        T_SWITCH: 17,
    }
    .expand()
}

pub const fn spiral_256_256(pack: bool, n_vec: usize) -> SPIRALParams {
    SPIRALParamsRaw {
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        T_GSW: 8,
        T_CONV: 4,
        T_COEFF_REGEV: 8,
        T_COEFF_GSW: 16,
        N_PACK: if pack { 4 } else { 1 },
        N_VEC: n_vec,
        T_SCAL_TO_VEC: 8,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 257,
        D_RECORD: 256,
        ETA1: 9,
        ETA2: 6,
        Z_FOLD: 2,
        Q_SWITCH1: 4 * 257, // 4P
        Q_SWITCH2: 2056193, // 21 bit prime
        D_SWITCH: 1024,
        T_SWITCH: 21,
    }
    .expand()
}

// pub const SPIRAL_TEST_PARAMS: SPIRALParams = spiral_256_256(true, 2);
pub const SPIRAL_TEST_PARAMS: SPIRALParams = spiral_16_512(1);

pub type SPIRALTest = spiral!(SPIRAL_TEST_PARAMS);

#[cfg(not(target_feature = "avx2"))]
pub fn has_avx2() -> bool {
    false
}

#[cfg(target_feature = "avx2")]
pub fn has_avx2() -> bool {
    true
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

pub fn run_spiral<TheSPIRAL: SPIRAL<RecordBytes = [u8; 256]>, I: Iterator<Item = usize>>(iter: I) {
    eprintln!(
        "Running SPIRAL test with database size {} (packed size {})",
        TheSPIRAL::DB_SIZE,
        TheSPIRAL::PACKED_DB_SIZE,
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
    let mut records: Vec<[u8; 256]> = Vec::with_capacity(TheSPIRAL::DB_SIZE);
    for i in 0..TheSPIRAL::DB_SIZE as u64 {
        let mut record = [0_u8; 256];
        record[0] = (i % 256) as u8;
        record[1] = ((i / 256) % 256) as u8;
        record[2] = 42_u8;
        record[3] = 0_u8;
        record[4] = (i % 100) as u8;
        record[5] = ((i / 100) % 100) as u8;
        record[6] = ((i / 100 / 100) % 100) as u8;
        record[7] = ((i / 100 / 100 / 100) % 100) as u8;
        records.push(record);
    }
    eprintln!(
        "Estimated relative noise: 2^({})",
        SPIRAL_TEST_PARAMS.noise_estimate().log2()
    );
    eprintln!(
        "Relative noise threshold: 2^({})",
        SPIRAL_TEST_PARAMS.relative_noise_threshold().log2()
    );

    eprintln!();

    let pre_start = Instant::now();
    let db = TheSPIRAL::encode_db(records.iter().copied());
    let pre_end = Instant::now();
    eprintln!("{:?} to preprocess", pre_end - pre_start);

    let setup_start = Instant::now();
    let (qk, pp) = TheSPIRAL::setup();
    let setup_end = Instant::now();
    eprintln!("{:?} to setup", setup_end - setup_start);

    let check = |indices: &[usize]| {
        eprintln!("Testing record indices {:?}", &indices);
        assert_eq!(
            indices.len(),
            SPIRAL_TEST_PARAMS.N_VEC * SPIRAL_TEST_PARAMS.N_PACK
        );
        let query_start = Instant::now();
        let q = TheSPIRAL::query(&qk, indices);
        let query_end = Instant::now();
        let query_total = query_end - query_start;

        let answer_start = Instant::now();
        let response_raw = TheSPIRAL::answer(&pp, &db, &q);
        let answer_end = Instant::now();
        let answer_total = answer_end - answer_start;

        let response_compress_start = Instant::now();
        let response = TheSPIRAL::response_compress(&pp, &response_raw);
        let response_compress_end = Instant::now();
        let response_compress_total = response_compress_end - response_compress_start;

        let response_extract_start = Instant::now();
        let extracted = TheSPIRAL::response_extract(&qk, &response);
        let response_extract_end = Instant::now();
        let response_extract_total = response_extract_end - response_extract_start;

        let decoded = TheSPIRAL::response_decode(&extracted);
        for (idx, decoded_record) in indices.iter().copied().zip(decoded) {
            if decoded_record != records[idx] {
                eprintln!("  **** **** **** **** ERROR **** **** **** ****");
                eprintln!("  protocol failed");
                dbg!(idx, idx / TheSPIRAL::PACK_RATIO);
                dbg!(&decoded_record);
            }
        }

        eprintln!(
            "  {:?} total",
            query_total + answer_total + response_compress_total + response_extract_total
        );
        eprintln!("    {:?} to query", query_total);
        eprintln!("    {:?} to answer", answer_total);
        eprintln!("    {:?} to compress response", response_compress_total);
        eprintln!("    {:?} to extract response", response_extract_total);

        let rel_noise = TheSPIRAL::response_raw_stats(&qk, &response_raw);
        eprintln!(
            "  relative coefficient noise (sample): 2^({})",
            rel_noise.log2()
        );
    };

    for chunk in iter
        .chunks(SPIRAL_TEST_PARAMS.N_VEC * SPIRAL_TEST_PARAMS.N_PACK)
        .into_iter()
    {
        let c_vec = chunk.collect_vec();
        check(c_vec.as_slice());
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod::IntMod;
    use crate::math::int_mod_cyclo::IntModCyclo;
    use crate::math::int_mod_poly::IntModPoly;
    use crate::math::matrix::Matrix;
    use crate::math::ring_elem::RingElement;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_regev() {
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
    fn test_project_hom() {
        let s = SPIRALTest::encode_setup();
        let gsw0 = SPIRALTest::encode_gsw(&s, &IntModCyclo::from(0_u64));
        let gsw1 = SPIRALTest::encode_gsw(&s, &IntModCyclo::from(1_u64));
        let mut msg_coeff = IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::zero();
        msg_coeff.coeff[0] = IntMod::from(10_u64);
        msg_coeff.coeff[1] = IntMod::from(20_u64);
        msg_coeff.coeff[2] = IntMod::from(11_u64);
        msg_coeff.coeff[3] = IntMod::from(21_u64);
        let c = SPIRALTest::encode_regev(&s, &msg_coeff.scale_up_into());

        let auto_key0 = SPIRALTest::auto_setup::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(SPIRAL_TEST_PARAMS.D + 1, &s);
        let auto_key1 = SPIRALTest::auto_setup::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(SPIRAL_TEST_PARAMS.D / 2 + 1, &s);

        let c_proj0 = SPIRALTest::project_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(0, &c, &gsw0, &auto_key0);
        let c_proj1 = SPIRALTest::project_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(0, &c, &gsw1, &auto_key0);

        let c_proj00 = SPIRALTest::project_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj0, &gsw0, &auto_key1);
        let c_proj01 = SPIRALTest::project_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj0, &gsw1, &auto_key1);
        let c_proj10 = SPIRALTest::project_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj1, &gsw0, &auto_key1);
        let c_proj11 = SPIRALTest::project_hom::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(1, &c_proj1, &gsw1, &auto_key1);

        // let proj0 = SPIRALTest::decode_regev(&s, &c_proj0).round_down_into();
        let mut proj0_expected = IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::zero();
        proj0_expected.coeff[0] = IntMod::from(10_u64 * 2);
        proj0_expected.coeff[2] = IntMod::from(11_u64 * 2);

        let mut proj1_expected = IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::zero();
        proj1_expected.coeff[0] = IntMod::from(20_u64 * 2);
        proj1_expected.coeff[2] = IntMod::from(21_u64 * 2);

        assert_eq!(
            SPIRALTest::decode_regev(&s, &c_proj0).round_down_into(),
            proj0_expected
        );
        assert_eq!(
            SPIRALTest::decode_regev(&s, &c_proj1).round_down_into(),
            proj1_expected
        );

        assert_eq!(
            SPIRALTest::decode_regev(&s, &c_proj00).round_down_into(),
            IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::from(10_u64 * 4)
        );
        assert_eq!(
            SPIRALTest::decode_regev(&s, &c_proj01).round_down_into(),
            IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::from(11_u64 * 4)
        );
        assert_eq!(
            SPIRALTest::decode_regev(&s, &c_proj10).round_down_into(),
            IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::from(20_u64 * 4)
        );
        assert_eq!(
            SPIRALTest::decode_regev(&s, &c_proj11).round_down_into(),
            IntModCyclo::<{ SPIRAL_TEST_PARAMS.D }, 256>::from(21_u64 * 4)
        );
    }

    #[test]
    fn test_scal_to_vec() {
        let s_scal = SPIRALTest::encode_setup();
        let s_vec = SPIRALTest::encode_vec_setup();
        let s_scal_to_vec = SPIRALTest::scal_to_vec_setup(&s_scal, &s_vec);

        let mut cs =
            Vec::<<SPIRALTest as SPIRAL>::RegevCiphertext>::with_capacity(SPIRAL_TEST_PARAMS.N_VEC);
        let mut expected =
            Matrix::<{ SPIRAL_TEST_PARAMS.N_VEC }, 1, <SPIRALTest as SPIRAL>::RingP>::zero();
        for i in 0..SPIRAL_TEST_PARAMS.N_VEC {
            let mu = <SPIRALTest as SPIRAL>::RingP::from(i as u64 + 1_u64);
            expected[(i, 0)] = mu.clone();
            cs.push(SPIRALTest::encode_regev(&s_scal, &mu.scale_up_into()));
        }

        let c_vec = SPIRALTest::scal_to_vec(&s_scal_to_vec, cs.as_slice().try_into().unwrap());
        let decoded = SPIRALTest::decode_vec_regev(&s_vec, &c_vec);
        let actual = decoded.map_ring(|r| r.round_down_into());
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_post_process_only() {
        let (qk, pp) = SPIRALTest::setup();
        let (_, s_vec, _) = &qk;
        let mut m = <SPIRALTest as SPIRAL>::RecordPackedSmall::zero();
        for i in 0..SPIRAL_TEST_PARAMS.N_VEC {
            m[(i, 0)] = IntModCyclo::from(177_u64 + i as u64)
        }
        let c =
            SPIRALTest::encode_vec_regev(s_vec, &m.map_ring(|r| r.include_dim().scale_up_into()));
        let compressed = SPIRALTest::response_compress(&pp, &c);
        let extracted = SPIRALTest::response_extract(&qk, &compressed);
        assert_eq!(m, extracted);
    }

    // #[test]
    // fn test_encode_decode() {
    //     // 16 x (log2(9) = 3 bits) <=> 6 bytes
    //     let bytes = [48_u8, 47, 17, 255, 183, 0];
    //     // 00110000 00101111 00010001 11111111 10110111 00000000
    //     // 001 100 000 010 111 100 010 001 111 111 111 011 011 100 000 000
    //     let encoded = SPIRALTest::encode_record(&bytes);
    //     assert_eq!(
    //         encoded,
    //         IntModCyclo::from(
    //             [1_u64, 4, 0, 2, 7, 4, 2, 1, 7, 7, 7, 3, 3, 4, 0, 0].map(IntMod::from)
    //         )
    //     );
    //     let decoded = SPIRALTest::decode_record(&encoded);
    //     assert_eq!(bytes, decoded);
    // }

    #[test]
    fn test_spiral_one() {
        run_spiral::<SPIRALTest, _>([11111].into_iter());
    }

    #[ignore]
    #[test]
    fn test_spiral_stress() {
        let mut rng = ChaCha20Rng::from_entropy();
        run_spiral::<SPIRALTest, _>((0..).map(|_| rng.gen_range(0_usize..SPIRALTest::DB_SIZE)));
    }
}
