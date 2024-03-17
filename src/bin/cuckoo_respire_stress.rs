use fhe_psi::pir::example::{run_pir, RespireBatch32Test};
use fhe_psi::pir::pir::PIR;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn main() {
    env_logger::init();
    let mut rng = ChaCha20Rng::from_entropy();
    run_pir::<RespireBatch32Test, _>(
        (0..).map(|_| rng.gen_range(0_usize..RespireBatch32Test::NUM_RECORDS)),
    );
}
