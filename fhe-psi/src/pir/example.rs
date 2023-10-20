use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo::IntModCyclo;
use crate::math::ring_elem::RingElement;
use crate::pir::pir::{SPIRALImpl, SPIRALParams, SPIRALParamsRaw, SPIRAL};
use crate::spiral;
use std::time::Instant;

pub const SPIRAL_TEST_PARAMS: SPIRALParams = SPIRALParamsRaw {
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
    P: 2,
    ETA1: 9,
    ETA2: 6,
    Z_FOLD: 2,
    Q_SWITCH1: 8, // 4P
    Q_SWITCH2: 114689, // must be prime
    D_SWITCH: 512,
    T_SWITCH: 17,
}
.expand();

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

pub fn run_spiral<TheSPIRAL: SPIRAL<Record = IntModCyclo<512, 2>>, I: Iterator<Item = usize>>(
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
    let mut records: Vec<<TheSPIRAL as SPIRAL>::Record> = Vec::with_capacity(SPIRALTest::DB_SIZE);
    for i in 0..TheSPIRAL::DB_SIZE as u64 {
        let mut record_coeff = [IntMod::<2>::zero(); 512];
        let bytes = [
            (i % 256) as u8,
            ((i / 256) % 256) as u8,
            42_u8,
            0_u8,
            (i % 100) as u8,
            ((i / 100) % 100) as u8,
            ((i / 100 / 100) % 100) as u8,
            ((i / 100 / 100 / 100) % 100) as u8,
        ];
        for (i, b) in bytes.iter().enumerate() {
            for bit_idx in 0..8 {
                record_coeff[8*i + bit_idx] = IntMod::<2>::from(((b >> bit_idx) & 1) as u64);
            }
        }
        records.push(<TheSPIRAL as SPIRAL>::Record::from(record_coeff));
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
    let db = TheSPIRAL::preprocess(records.iter());
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

        if extracted != records[idx] {
            eprintln!("  **** **** **** **** ERROR **** **** **** ****");
            eprintln!("  protocol failed");
        }
        eprintln!(
            "  {:?} total",
            query_total + answer_total + response_compress_total + response_extract_total
        );
        eprintln!("    {:?} to query", query_total);
        eprintln!("    {:?} to answer", answer_total);
        eprintln!("    {:?} to compress response", response_compress_total);
        eprintln!("    {:?} to extract response", response_extract_total);
        let rel_noise = TheSPIRAL::response_raw_stats(&qk, &response_raw, &records[idx]);

        eprintln!(
            "  relative coefficient noise (sample): 2^({})",
            rel_noise.log2()
        );
    };

    for i in iter {
        check(i);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod_poly::IntModPoly;
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
    fn test_post_process_only() {
        let (qk, pp) = SPIRALTest::setup();
        let (s_encode, _) = &qk;
        let m = <SPIRALTest as SPIRAL>::Record::from(177_u64);
        let c = SPIRALTest::encode_regev(s_encode, &m.include_dim().scale_up_into());
        let compressed = SPIRALTest::response_compress(&pp, &c);
        let extracted = SPIRALTest::response_extract(&qk, &compressed);
        assert_eq!(m, extracted);
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
}
