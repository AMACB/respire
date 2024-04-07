use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_harness::FactoryParams;
use respire::{generate_main, respire};

const PARAMS: RespireParamsExpanded = FactoryParams::single_record_256(10, 10).expand().expand();

type ThePIR = respire!(PARAMS);
generate_main!(ThePIR);
