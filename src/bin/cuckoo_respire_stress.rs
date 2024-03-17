use fhe_psi::pir::example::{run_pir, RespireBatch256Test};
use fhe_psi::pir::pir::PIR;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

type ThePIR = RespireBatch256Test;

fn main() {
    env_logger::init();
    let mut rng = ChaCha20Rng::from_entropy();
    run_pir::<ThePIR, _>((0..).map(|_| rng.gen_range(0_usize..ThePIR::NUM_RECORDS)));
}
