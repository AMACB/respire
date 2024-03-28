use respire::pir::cuckoo_respire::CuckooRespireImpl;
use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_factory::FactoryParams;
use respire::{generate_main, respire};

const BASE_PARAMS: RespireParamsExpanded = FactoryParams {
    BATCH_SIZE: 398,
    N_VEC: 7,
    P: 16,
    D_RECORD: 512,
    NU1: 7,
    NU2: 4,
    Q_SWITCH1: 8 * 16,
    Q_SWITCH2: 1032193, // 19.97 bits
    D_SWITCH: 512,
    WIDTH_SWITCH_MILLIONTHS: 46_000_000,
}
.expand();

type BasePIR = respire!(BASE_PARAMS);
type CuckooPIR = CuckooRespireImpl<256, 398, { 2usize.pow(20) }, BasePIR>;

generate_main!(CuckooPIR);
