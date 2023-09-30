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
        let scalar_key = SPIRALTest::encode_setup();
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
        let scalar_key = SPIRALTest::encode_setup();
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
            let scalar_key = SPIRALTest::encode_setup();
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
            let scalar_key = SPIRALTest::encode_setup();
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
        let scalar_key = SPIRALTest::encode_setup();
        let matrix_key = SPIRALTest::matrix_regev_setup();
        let scal_to_mat_key = SPIRALTest::scal_to_mat_setup(&scalar_key, &matrix_key);

        let msg = IntModCyclo::rand_uniform(&mut rng);
        let ct = SPIRALTest::encode_regev(&scalar_key, &msg);

        b.iter(|| SPIRALTest::scal_to_mat(black_box(&scal_to_mat_key), black_box(&ct)))
    });

    c.bench_function("pir::regev to gsw", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let scalar_key = SPIRALTest::encode_setup();
        let matrix_key = SPIRALTest::matrix_regev_setup();
        let regev_to_gsw_key = SPIRALTest::regev_to_gsw_setup(&scalar_key, &matrix_key);

        let msg: IntModCyclo<{ SPIRAL_TEST_PARAMS.D }, { SPIRAL_TEST_PARAMS.Q }> =
            IntModCyclo::rand_uniform(&mut rng);
        let mut msg_curr = msg.include_into();
        let mut encrypt_vec = Vec::with_capacity(SPIRAL_TEST_PARAMS.T_GSW);
        for _ in 0..SPIRAL_TEST_PARAMS.T_GSW {
            encrypt_vec.push(SPIRALTest::encode_regev(&scalar_key, &msg_curr));
            msg_curr *= IntMod::from(SPIRAL_TEST_PARAMS.Z_GSW);
        }

        b.iter(|| {
            SPIRALTest::regev_to_gsw(
                black_box(&regev_to_gsw_key),
                black_box(encrypt_vec.as_slice()),
            )
        });
    });

    c.bench_function("pir::scalar_regev_mul_x_pow", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let ct = Matrix::rand_uniform(&mut rng);
        b.iter(|| SPIRALTest::regev_mul_x_pow(black_box(&ct), black_box(101)))
    });

    c.bench_function("pir::query_expand", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let (qk, pp) = SPIRALTest::setup();
        let idx = rng.gen_range(0..<SPIRALTest as SPIRAL>::DB_SIZE);
        let q = SPIRALTest::query(&qk, idx);
        b.iter(|| SPIRALTest::query_expand(black_box(&pp), black_box(&q)))
    });

    c.bench_function("pir::regev_mul_scalar_no_reduce", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let m1 = Matrix::rand_uniform(&mut rng);
        let m2 = Matrix::rand_uniform(&mut rng);
        b.iter(|| SPIRALTest::regev_mul_scalar_no_reduce(black_box(&m1), black_box(&m2)));
    });

    c.bench_function("pir::regev_add_eq_mul_scalar_no_reduce", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut m1 = Matrix::rand_uniform(&mut rng);
        let m2 = Matrix::rand_uniform(&mut rng);
        let m3 = Matrix::rand_uniform(&mut rng);
        b.iter(|| {
            SPIRALTest::regev_add_eq_mul_scalar_no_reduce(
                black_box(&mut m1),
                black_box(&m2),
                black_box(&m3),
            )
        });
    });

    c.bench_function("pir::reduce_mod", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut elem = <SPIRALTest as SPIRAL>::Ring0Fast::rand_uniform(&mut rng);
        b.iter(|| <SPIRALTest as SPIRAL>::RingQFast::reduce_mod(black_box(&mut elem)));
    });

    c.bench_function("pir::regev_sub_hom", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let m1 = Matrix::rand_uniform(&mut rng);
        let m2 = Matrix::rand_uniform(&mut rng);
        b.iter(|| SPIRALTest::regev_sub_hom(black_box(&m1), black_box(&m2)));
    });

    c.bench_function("pir::hybrid_mul_hom", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let m1 = Matrix::rand_uniform(&mut rng);
        let m2 = Matrix::rand_uniform(&mut rng);
        b.iter(|| SPIRALTest::hybrid_mul_hom(black_box(&m1), black_box(&m2)));
    });

    c.bench_function("pir::first_dim", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let db: Vec<_> = (0..)
            .map(|_| Matrix::rand_uniform(&mut rng))
            .take(SPIRALTest::DB_SIZE)
            .collect();
        let regevs: Vec<_> = (0..)
            .map(|_| Matrix::rand_uniform(&mut rng))
            .take(1 << SPIRALTest::ETA1)
            .collect();

        b.iter(|| {
            SPIRALTest::answer_first_dim(black_box(db.as_slice()), black_box(regevs.as_slice()))
        });
    });

    c.bench_function("pir::fold", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let first_dim_folded: Vec<_> = (0..)
            .map(|_| Matrix::rand_uniform(&mut rng))
            .take(1 << SPIRALTest::ETA1)
            .collect();
        let gsws: Vec<_> = (0..)
            .map(|_| Matrix::rand_uniform(&mut rng))
            .take(SPIRALTest::ETA2 * (SPIRAL_TEST_PARAMS.Z_FOLD - 1))
            .collect();

        // Note: this includes the time it takes to clone first_dim_folded
        b.iter(|| {
            SPIRALTest::answer_fold(
                black_box(first_dim_folded.clone()),
                black_box(gsws.as_slice()),
            )
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
