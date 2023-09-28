use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fhe_psi::math::int_mod::IntMod;
use fhe_psi::math::int_mod_cyclo::IntModCyclo;
use fhe_psi::math::matrix::Matrix;
use fhe_psi::math::rand_sampled::RandUniformSampled;
use fhe_psi::pir::pir::{SPIRALImpl, SPIRALParams, SPIRALParamsRaw, SPIRAL};
use fhe_psi::spiral;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn criterion_benchmark(c: &mut Criterion) {
    const SPIRAL_TEST_PARAMS: SPIRALParams = SPIRALParamsRaw {
        N: 2,
        Q_A: 268369921,
        Q_B: 249561089,
        D: 2048,
        T_GSW: 8,
        T_CONV: 4,
        T_COEFF_REGEV: 8,
        T_COEFF_GSW: 56,
        NOISE_WIDTH_MILLIONTHS: 6_400_000,
        P: 1 << 8,
        ETA1: 9,
        ETA2: 6,
        Z_FOLD: 2,
    }
    .expand();
    type SPIRALTest = spiral!(SPIRAL_TEST_PARAMS);

    c.bench_function("pir::automorphism with T_COEFF_REGEV", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let scalar_key = SPIRALTest::scalar_regev_setup();
        let auto_key_regev = SPIRALTest::auto_setup::<
            { SPIRAL_TEST_PARAMS.T_COEFF_REGEV },
            { SPIRAL_TEST_PARAMS.Z_COEFF_REGEV },
        >(3, &scalar_key);
        let ct = Matrix::rand_uniform(&mut rng);
        b.iter(|| {
            SPIRALTest::auto_hom::<
                { SPIRAL_TEST_PARAMS.T_COEFF_REGEV },
                { SPIRAL_TEST_PARAMS.Z_COEFF_REGEV },
            >(black_box(&auto_key_regev), black_box(&ct))
        })
    });

    c.bench_function("pir::automorphism with T_COEFF_GSW", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let scalar_key = SPIRALTest::scalar_regev_setup();
        let auto_key_regev = SPIRALTest::auto_setup::<
            { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
            { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
        >(3, &scalar_key);
        let ct = Matrix::rand_uniform(&mut rng);
        b.iter(|| {
            SPIRALTest::auto_hom::<
                { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
                { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
            >(black_box(&auto_key_regev), black_box(&ct))
        })
    });

    let mut group = c.benchmark_group("pir::do_expand_iter with T_COEFF_REGEV");
    for i in 0..4 {
        group.bench_with_input(BenchmarkId::from_parameter(i), &i, |b, &i| {
            let mut rng = ChaCha20Rng::from_entropy();
            let scalar_key = SPIRALTest::scalar_regev_setup();
            let auto_key_regev = SPIRALTest::auto_setup::<
                { SPIRAL_TEST_PARAMS.T_COEFF_REGEV },
                { SPIRAL_TEST_PARAMS.Z_COEFF_REGEV },
            >(SPIRAL_TEST_PARAMS.D / (1 << i) + 1, &scalar_key);
            let mut cts = Vec::with_capacity(1 << i);
            for _ in 0..(1 << i) {
                cts.push(Matrix::rand_uniform(&mut rng));
            }
            b.iter(|| {
                SPIRALTest::do_coeff_expand_iter::<
                    { SPIRAL_TEST_PARAMS.T_COEFF_REGEV },
                    { SPIRAL_TEST_PARAMS.Z_COEFF_REGEV },
                >(
                    black_box(i),
                    black_box(cts.as_slice()),
                    black_box(&auto_key_regev),
                )
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("pir::do_expand_iter with T_COEFF_GSW");
    for i in 0..4 {
        group.bench_with_input(BenchmarkId::from_parameter(i), &i, |b, &i| {
            let mut rng = ChaCha20Rng::from_entropy();
            let scalar_key = SPIRALTest::scalar_regev_setup();
            let auto_key_regev = SPIRALTest::auto_setup::<
                { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
                { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
            >(SPIRAL_TEST_PARAMS.D / (1 << i) + 1, &scalar_key);
            let mut cts = Vec::with_capacity(1 << i);
            for _ in 0..(1 << i) {
                cts.push(Matrix::rand_uniform(&mut rng));
            }
            b.iter(|| {
                SPIRALTest::do_coeff_expand_iter::<
                    { SPIRAL_TEST_PARAMS.T_COEFF_GSW },
                    { SPIRAL_TEST_PARAMS.Z_COEFF_GSW },
                >(
                    black_box(i),
                    black_box(cts.as_slice()),
                    black_box(&auto_key_regev),
                )
            });
        });
    }
    group.finish();

    c.bench_function("pir::scal to mat", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let scalar_key = SPIRALTest::scalar_regev_setup();
        let matrix_key = SPIRALTest::matrix_regev_setup();
        let scal_to_mat_key = SPIRALTest::scal_to_mat_setup(&scalar_key, &matrix_key);

        let msg = IntModCyclo::rand_uniform(&mut rng);
        let ct = SPIRALTest::encode_scalar_regev(&scalar_key, &msg);

        b.iter(|| SPIRALTest::scal_to_mat(black_box(&scal_to_mat_key), black_box(&ct)))
    });

    c.bench_function("pir::regev to gsw", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let scalar_key = SPIRALTest::scalar_regev_setup();
        let matrix_key = SPIRALTest::matrix_regev_setup();
        let regev_to_gsw_key = SPIRALTest::regev_to_gsw_setup(&scalar_key, &matrix_key);

        let msg: IntModCyclo<{ SPIRAL_TEST_PARAMS.D }, { SPIRAL_TEST_PARAMS.Q }> =
            IntModCyclo::rand_uniform(&mut rng);
        let mut msg_curr = msg.include_into();
        let mut encrypt_vec = Vec::with_capacity(SPIRAL_TEST_PARAMS.T_GSW);
        for _ in 0..SPIRAL_TEST_PARAMS.T_GSW {
            encrypt_vec.push(SPIRALTest::encode_scalar_regev(&scalar_key, &msg_curr));
            msg_curr *= IntMod::from(SPIRAL_TEST_PARAMS.Z_GSW);
        }

        b.iter(|| {
            SPIRALTest::regev_to_gsw(
                black_box(&regev_to_gsw_key),
                black_box(encrypt_vec.as_slice()),
            )
        })
    });

    c.bench_function("pir::query expansion", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let (qk, pp) = SPIRALTest::setup();
        let idx = rng.gen_range(0..<SPIRALTest as SPIRAL>::DB_SIZE);
        let q = SPIRALTest::query(&qk, idx);
        b.iter(|| {
            SPIRALTest::query_expand(black_box(&pp), black_box(&q));
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
