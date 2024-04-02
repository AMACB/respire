use respire::pir::cuckoo_respire::CuckooRespireImpl;
use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_harness::FactoryParams;
use respire::{generate_main, respire};

const BASE_PARAMS: RespireParamsExpanded = FactoryParams {
    BATCH_SIZE: 13,
    N_VEC: 4,
    P: 16,
    D_RECORD: 512,
    NU1: 8,
    NU2: 8,
    Q_SWITCH1: 8 * 16,
    Q_SWITCH2: 249857,
    D_SWITCH: 2048,
    WIDTH_SWITCH_MILLIONTHS: 9_900_000,
}
.expand()
.expand();

type BasePIR = respire!(BASE_PARAMS);
type CuckooPIR = CuckooRespireImpl<8, 13, { 2usize.pow(20) }, BasePIR>;

generate_main!(CuckooPIR);
