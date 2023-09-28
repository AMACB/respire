use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fhe_psi::math::int_mod_cyclo::IntModCyclo;
use fhe_psi::math::ntt::{ntt_neg_backward, ntt_neg_forward};
use fhe_psi::math::number_theory::find_sqrt_primitive_root;
use fhe_psi::math::rand_sampled::RandUniformSampled;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn criterion_benchmark(c: &mut Criterion) {
    const D: usize = 2048;
    const P: u64 = 268369921;
    const W: u64 = find_sqrt_primitive_root(D, P);
    type RCoeff = IntModCyclo<D, P>;

    c.bench_function("math::ntt_neg_forward", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let elem = RCoeff::rand_uniform(&mut rng);
        let mut points = elem.into_aligned();
        b.iter(|| {
            ntt_neg_forward::<D, P, W>(black_box(&mut points));
        });
    });

    c.bench_function("math::ntt_neg_backward", |b| {
        let mut rng = ChaCha20Rng::from_entropy();
        let elem = RCoeff::rand_uniform(&mut rng);
        let mut points = elem.into_aligned();
        b.iter(|| {
            ntt_neg_backward::<D, P, W>(black_box(&mut points));
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
