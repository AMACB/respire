use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_factory::FactoryParams;
use respire::{generate_main, respire};

const PARAMS: RespireParamsExpanded = FactoryParams {
    BATCH_SIZE: 1,
    N_VEC: 1,
    P: 16,
    D_RECORD: 512,
    NU1: 9,
    NU2: 9,
    Q_SWITCH1: 8 * 16,
    Q_SWITCH2: 1032193, // 19.97 bits
    D_SWITCH: 512,
    WIDTH_SWITCH_MILLIONTHS: 46_000_000,
}
.expand();

// This also works for 256 bytes:
// p = 16
// q1 = 6 * 16,
// q2 = 2056193
// width = 70.0

// Other d_record = d_switch = 512:
// 128 byte records:
//     p = 4
//     q1 = 6 * 4,
//     q2: 61441,
//     width = 9.2
// 64 byte records:
//     p = 2
//     q1 = 6 * 2
//     q2 = 12289
//     width = 4.0

type ThePIR = respire!(PARAMS);
generate_main!(ThePIR);
