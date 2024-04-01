use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_harness::FactoryParams;
use respire::{generate_main, respire};

const PARAMS: RespireParamsExpanded = FactoryParams {
    BATCH_SIZE: 1,
    N_VEC: 1,
    P: 16,
    D_RECORD: 512,
    NU1: 10,
    NU2: 10,
    Q_SWITCH1: 16 * 16,
    Q_SWITCH2: 1032193,
    D_SWITCH: 512,
    WIDTH_SWITCH_MILLIONTHS: 46_000_000,
}
.expand()
.expand();

type ThePIR = respire!(PARAMS);
generate_main!(ThePIR);
