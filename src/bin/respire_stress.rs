use fhe_psi::pir::example::{run_pir, RespireTest};
use fhe_psi::pir::pir::PIR;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

type ThePIR = RespireTest;

fn main() {
    env_logger::init();
    let mut rng = ChaCha20Rng::from_entropy();
    run_pir::<ThePIR, _>((0..).map(|_| rng.gen_range(0_usize..ThePIR::NUM_RECORDS)));
}
