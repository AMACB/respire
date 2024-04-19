use crate::pir::pir::{PIRRecordBytes, Stats, PIR};
use crate::pir::respire::{RespireParams, RespireParamsExpanded};
use crate::respire;
use clap::Parser;
use itertools::Itertools;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::time::{Duration, Instant};

//
// Parameter factory functions
//

#[allow(non_snake_case)]
pub struct FactoryParams {
    pub BATCH_SIZE: usize,
    pub N_VEC: usize,
    pub P: u64,
    pub D3: usize,
    pub NU1: usize,
    pub NU2: usize,
    pub Q3: u64,
    pub Q2: u64,
    pub D2: usize,
    pub WIDTH_COMPRESS_MILLIONTHS: u64,
    pub T_PROJ_SHORT: usize,
    pub T_PROJ_LONG: usize,
}

impl FactoryParams {
    pub const fn single_record_256(nu1: usize, nu2: usize) -> Self {
        // *** NOTES ***
        //
        // Other 256 bytes, p = 16:
        // q2 = 1032193, width = 46.0
        // q2 = 2056193, width = 70.0
        // q2 = 16760833, width = 253.0
        //
        // p = 256, d3 = 256 seems hopeless (gadget term too big)
        //
        // Other d3 = d2 = 512:
        // 128 byte records:
        //     p = 4
        //     q3 = 6 * 4,
        //     q2: 61441,
        //     width = 9.2
        // 64 byte records:
        //     p = 2
        //     q3 = 6 * 2
        //     q2 = 12289
        //     width = 4.0
        //
        // Values for q2:
        // 14 bits: 12289
        // 16 bits: 61441
        // 17 bits: 114689
        // 18 bits: 249857
        // 19 bits: 520193
        // 20 bits: 1032193
        // 21 bits: 2056193
        // 22 bits: 4169729
        // 23 bits: 8380417
        // 24 bits: 16760833
        // 25 bits: 33550337
        // 26 bits: 67104769
        // 27 bits: 134176769
        // 28 bits: 268369921
        // 29 bits: 536813569
        // 30 bits: 1073692673
        // 31 bits: 2147389441
        // 32 bits: 4294955009
        /*
        # LWE estimator code
        from estimator import *
        from estimator.nd import stddevf
        from estimator.nd import NoiseDistribution as ND

        def scheme(d, q, w):
            return schemes.LWEParameters(n=d, q=q, Xs=ND.DiscreteGaussian(stddevf(w)), Xe=ND.DiscreteGaussian(stddevf(w)))
        */
        FactoryParams {
            BATCH_SIZE: 1,
            N_VEC: 1,
            P: 16,
            D3: 512,
            NU1: nu1,
            NU2: nu2,
            Q3: 16 * 16,
            Q2: 16760833,
            D2: 512,
            WIDTH_COMPRESS_MILLIONTHS: 253_600_000,
            T_PROJ_SHORT: 4,
            T_PROJ_LONG: 20,
        }
    }

    pub const fn batch_256(batch_size: usize, n_vec: usize, nu1: usize, nu2: usize) -> Self {
        FactoryParams {
            BATCH_SIZE: batch_size,
            N_VEC: n_vec,
            P: 16,
            D3: 512,
            NU1: nu1,
            NU2: nu2,
            Q3: 8 * 16,
            Q2: 249857,
            D2: 2048,
            WIDTH_COMPRESS_MILLIONTHS: 2_001_000,
            T_PROJ_SHORT: 4,
            T_PROJ_LONG: 20,
        }
    }

    pub const fn expand(&self) -> RespireParams {
        RespireParams {
            Q1A: 268369921,
            Q1B: 249561089,
            D1: 2048,
            T_GSW: 8,
            T_RLWE_TO_GSW: 4,
            T_PROJ_SHORT: self.T_PROJ_SHORT,
            T_PROJ_LONG: self.T_PROJ_LONG,
            T_VECTORIZE: 2,
            BATCH_SIZE: self.BATCH_SIZE,
            N_VEC: self.N_VEC,
            ERROR_WIDTH_MILLIONTHS: 9_900_000,
            ERROR_WIDTH_VEC_MILLIONTHS: 9_900_000,
            ERROR_WIDTH_COMPRESS_MILLIONTHS: self.WIDTH_COMPRESS_MILLIONTHS,
            SECRET_BOUND: 7,
            SECRET_WIDTH_VEC_MILLIONTHS: 9_900_000,
            SECRET_WIDTH_COMPRESS_MILLIONTHS: self.WIDTH_COMPRESS_MILLIONTHS,
            P: self.P,
            D3: self.D3,
            NU1: self.NU1,
            NU2: self.NU2,
            Q3: self.Q3,
            Q2: self.Q2,
            D2: self.D2,
        }
    }
}

// For quick testing

pub const RESPIRE_TEST_PARAMS: RespireParamsExpanded =
    FactoryParams::single_record_256(9, 9).expand().expand();

pub type RespireTest = respire!(RESPIRE_TEST_PARAMS);

#[cfg(not(target_feature = "avx2"))]
pub fn has_avx2() -> bool {
    false
}

#[cfg(target_feature = "avx2")]
pub fn has_avx2() -> bool {
    true
}

pub struct RunResult {
    pub init_times: Stats<Duration>,
    pub all_trial_times: Vec<Stats<Duration>>,
}

pub fn run_pir<ThePIR: PIR, I: Iterator<Item = usize>>(iter: I) -> RunResult {
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

    let records_generator = |i: usize| {
        let mut record = vec![0_u8; ThePIR::BYTES_PER_RECORD];
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
        ThePIR::RecordBytes::from_bytes(record.as_slice()).unwrap()
    };

    let mut init_times = Stats::new();
    let begin = Instant::now();
    let (db, db_hint) = ThePIR::encode_db(records_generator, Some(&mut init_times));
    let (qk, pp) = ThePIR::setup(Some(&mut init_times));
    let end = Instant::now();

    init_times.add(
        "total",
        init_times
            .as_vec()
            .iter()
            .fold(Duration::new(0, 0), |acc, x| acc + x.1),
    );

    eprintln!("Init times:");
    for (stat, value) in init_times.as_vec() {
        eprintln!("    {}: {:?}", stat, value);
    }
    eprintln!("Init time (end-to-end): {:?}", end - begin);
    eprintln!("========");

    let mut all_trial_times = Vec::new();

    let mut run_trial = |indices: &[usize]| {
        eprintln!("Running trial on indices {:?}", &indices);
        assert_eq!(indices.len(), ThePIR::BATCH_SIZE);
        let mut trial_times = Stats::new();

        let begin = Instant::now();
        let (q, st) = ThePIR::query(&qk, indices, &db_hint, Some(&mut trial_times));
        let response = ThePIR::answer(&pp, &db, &q, Some(&qk), Some(&mut trial_times));
        let extracted = ThePIR::extract(&qk, &response, &st, Some(&mut trial_times));
        let end = Instant::now();

        trial_times.add(
            "total",
            trial_times
                .as_vec()
                .iter()
                .fold(Duration::new(0, 0), |acc, x| acc + x.1),
        );

        eprintln!("Trial times:");
        for (stat, value) in trial_times.as_vec() {
            eprintln!("    {}: {:?}", stat, value);
        }
        eprintln!("Trial time (end-to-end): {:?}", end - begin);
        all_trial_times.push(trial_times);

        for (idx, decoded_record) in indices.iter().copied().zip(extracted) {
            if decoded_record.as_bytes() != records_generator(idx).as_bytes() {
                eprintln!("**** **** **** **** ERROR **** **** **** ****");
                eprintln!("protocol failed");
                eprintln!("idx = {}", idx);
                eprintln!("decoded record = {:?}", decoded_record.as_bytes());
                eprintln!("actual record = {:?}", records_generator(idx).as_bytes());
            }
        }
        eprintln!("========");
    };

    for chunk in iter.chunks(ThePIR::BATCH_SIZE).into_iter() {
        let c_vec = chunk.collect_vec();
        run_trial(c_vec.as_slice());
    }

    RunResult {
        init_times,
        all_trial_times,
    }
}

#[macro_export]
macro_rules! generate_main {
    ($name: path) => {
        fn main() {
            $crate::pir::respire_harness::harness_main::<$name>();
        }
    };
}

#[derive(Parser, Debug)]
struct Args {
    trials: usize,
}

pub fn harness_main<ThePIR: PIR>() {
    env_logger::init();
    let args = Args::parse();

    let mut rng = ChaCha20Rng::from_entropy();
    let record_gen = |_| rng.gen_range(0_usize..ThePIR::NUM_RECORDS);
    let run_result =
        run_pir::<ThePIR, _>((0usize..args.trials * ThePIR::BATCH_SIZE).map(record_gen));

    let trial_times = run_result
        .all_trial_times
        .iter()
        .map(|tt| tt.as_vec())
        .collect_vec();
    let stat_names = trial_times[0].iter().map(|x| x.0).collect_vec();
    for tt in trial_times.iter() {
        assert_eq!(tt.iter().map(|x| x.0).collect_vec(), stat_names);
    }

    let mut means = Vec::with_capacity(trial_times.len());
    let mut stddevs = Vec::with_capacity(trial_times.len());

    eprintln!("Summary times:");

    for (stat_i, stat_name) in stat_names.iter().copied().enumerate() {
        let mut sum = 0_f64;
        let mut sum_sq = 0_f64;
        for tt in trial_times.iter() {
            let value = tt[stat_i].1.as_nanos() as f64;
            sum += value;
            sum_sq += value.powi(2);
        }

        let mean = sum / trial_times.len() as f64;
        let stddev = (sum_sq / trial_times.len() as f64 - mean.powi(2)).sqrt();

        let mean = mean.round() as u64;
        let stddev = stddev.round() as u64;
        means.push(mean);
        stddevs.push(stddev);
        eprintln!(
            "    {}: {:?} mean, {:?} stddev ({:.3}%)",
            stat_name,
            Duration::from_nanos(mean),
            Duration::from_nanos(stddev),
            stddev as f64 / mean as f64 / 100_f64,
        );
    }

    eprintln!("mean, stddev in CSV format (times in nanoseconds):");
    eprintln!("{}", stat_names.join(", "));
    eprintln!("{}", means.iter().map(u64::to_string).join(", "));
    eprintln!("{}", stddevs.iter().map(u64::to_string).join(", "));
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod::IntMod;
    use crate::math::int_mod_cyclo::IntModCyclo;
    use crate::math::int_mod_poly::IntModPoly;
    use crate::math::matrix::Matrix;
    use crate::pir::respire::Respire;

    #[test]
    fn test_regev() {
        let s = RespireTest::encode_setup();
        let mu = <RespireTest as Respire>::RingP::from(12_u64);
        let encoded = RespireTest::encode_rlwe(&s, &mu.scale_up_into());
        let decoded: <RespireTest as Respire>::RingP =
            RespireTest::decode_rlwe(&s, &encoded).round_down_into();
        assert_eq!(mu, decoded);
    }

    #[test]
    fn test_gsw() {
        let s = RespireTest::encode_setup();
        type RingPP = IntModCyclo<{ RESPIRE_TEST_PARAMS.D1 }, 1024>;
        let mu = RingPP::from(111_u64);
        let encrypt = RespireTest::encode_gsw(&s, &mu.include_into());

        let scale = <RespireTest as Respire>::RingQ1Fast::from(RESPIRE_TEST_PARAMS.Q1 / 1024);
        let decrypt = RespireTest::decode_gsw_scaled(&s, &encrypt, &scale);
        assert_eq!(decrypt.round_down_into(), mu);
    }

    #[test]
    fn test_auto_hom() {
        let s = RespireTest::encode_setup();
        let auto_key = RespireTest::auto_setup::<
            { RESPIRE_TEST_PARAMS.T_PROJ_SHORT },
            { RESPIRE_TEST_PARAMS.Z_PROJ_SHORT },
        >(3, &s);
        let x = <RespireTest as Respire>::RingP::from(IntModPoly::x());
        let encrypt = RespireTest::encode_rlwe(&s, &x.scale_up_into());
        let encrypt_auto = RespireTest::auto_hom::<
            { RESPIRE_TEST_PARAMS.T_PROJ_SHORT },
            { RESPIRE_TEST_PARAMS.Z_PROJ_SHORT },
        >(&auto_key, &encrypt);
        let decrypt: <RespireTest as Respire>::RingP =
            RespireTest::decode_rlwe(&s, &encrypt_auto).round_down_into();
        assert_eq!(decrypt, &(&x * &x) * &x);
    }

    #[test]
    fn test_regev_to_gsw() {
        let s = RespireTest::encode_setup();
        let s_regev_to_gsw = RespireTest::rlwe_to_gsw_setup(&s);
        type RingPP = IntModCyclo<{ RESPIRE_TEST_PARAMS.D1 }, 1024>;
        let mu = RingPP::from(567_u64);
        let mut mu_curr = mu.include_into();
        let mut encrypt_vec = Vec::with_capacity(RESPIRE_TEST_PARAMS.T_GSW);
        for _ in 0..RESPIRE_TEST_PARAMS.T_GSW {
            encrypt_vec.push(RespireTest::encode_rlwe(&s, &mu_curr));
            mu_curr *= IntMod::from(RESPIRE_TEST_PARAMS.Z_GSW);
        }
        let encrypt_gsw = RespireTest::rlwe_to_gsw(&s_regev_to_gsw, encrypt_vec.as_slice());

        let scale = <RespireTest as Respire>::RingQ1Fast::from(RESPIRE_TEST_PARAMS.Q1 / 1024);
        let decrypted = RespireTest::decode_gsw_scaled(&s, &encrypt_gsw, &scale);
        assert_eq!(decrypted.round_down_into(), mu);
    }

    #[test]
    fn test_scal_to_vec() {
        let s_scal = RespireTest::encode_setup();
        let s_vec = RespireTest::encode_vec_setup();
        let s_scal_to_vec = RespireTest::vectorize_setup(&s_scal, &s_vec);

        let mut cs =
            Vec::<<RespireTest as Respire>::RLWEEncoding>::with_capacity(RESPIRE_TEST_PARAMS.N_VEC);
        let mut expected =
            Matrix::<{ RESPIRE_TEST_PARAMS.N_VEC }, 1, <RespireTest as Respire>::RingP>::zero();
        for i in 0..RESPIRE_TEST_PARAMS.N_VEC {
            let mu = <RespireTest as Respire>::RingP::from(i as u64 + 1_u64);
            expected[(i, 0)] = mu.clone();
            cs.push(RespireTest::encode_rlwe(&s_scal, &mu.scale_up_into()));
        }

        let c_vec = RespireTest::scal_to_vec(&s_scal_to_vec, cs.as_slice().try_into().unwrap());
        let decoded = RespireTest::decode_vec_rlwe(&s_vec, &c_vec);
        let actual = decoded.map_ring(|r| r.round_down_into());
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_post_process_only() {
        let (qk, pp) = RespireTest::setup(None);
        let (_, s_vec, _) = &qk;
        let mut m = <RespireTest as Respire>::RecordPackedSmall::zero();
        for i in 0..RESPIRE_TEST_PARAMS.N_VEC {
            m[(i, 0)] = IntModCyclo::from(177_u64 + i as u64)
        }
        let c =
            RespireTest::encode_vec_rlwe(s_vec, &m.map_ring(|r| r.include_dim().scale_up_into()));
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
