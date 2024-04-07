use respire::pir::cuckoo_respire::CuckooRespireImpl;
use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_harness::FactoryParams;
use respire::{generate_main, respire};

const BASE_PARAMS: RespireParamsExpanded = FactoryParams::batch_256(391, 4, 7, 6).expand().expand();

type BasePIR = respire!(BASE_PARAMS);
type CuckooPIR = CuckooRespireImpl<256, 391, { 2usize.pow(22) }, BasePIR>;

generate_main!(CuckooPIR);
