use fhe_psi::pir::example::{run_pir, RespireTest};
use fhe_psi::pir::pir::PIR;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn main() {
    env_logger::init();
    let mut rng = ChaCha20Rng::from_entropy();
    run_pir::<RespireTest, _>((0..).map(|_| rng.gen_range(0_usize..RespireTest::NUM_RECORDS)));
}
