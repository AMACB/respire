use respire::pir::cuckoo_respire::CuckooRespireImpl;
use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_harness::FactoryParams;
use respire::{generate_main, respire};

const BASE_PARAMS: RespireParamsExpanded = FactoryParams {
    BATCH_SIZE: 391,
    N_VEC: 13,
    P: 16,
    D_RECORD: 512,
    NU1: 7,
    NU2: 6,
    Q_SWITCH1: 4 * 16,
    Q_SWITCH2: 249857,
    D_SWITCH: 2048,
    WIDTH_SWITCH_MILLIONTHS: 2_001_000,
}
.expand()
.expand();

type BasePIR = respire!(BASE_PARAMS);
type CuckooPIR = CuckooRespireImpl<256, 391, { 2usize.pow(22) }, BasePIR>;

generate_main!(CuckooPIR);
