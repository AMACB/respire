use respire::pir::respire::RespireParamsExpanded;
use respire::pir::respire_harness::FactoryParams;
use respire::{generate_main, respire};

const PARAMS: RespireParamsExpanded = FactoryParams {
    BATCH_SIZE: 1,
    N_VEC: 1,
    P: 16,
    D_RECORD: 512,
    NU1: 9,
    NU2: 9,
    Q_SWITCH1: 8 * 16,
    Q_SWITCH2: 1032193,
    D_SWITCH: 512,
    WIDTH_SWITCH_MILLIONTHS: 46_000_000,
}
.expand()
.expand();

// *** NOTES ***
//
// This also works for 256 bytes:
// p = 16
// q1 = 6 * 16,
// q2 = 2056193
// width = 70.0
//
// p = 256, d_record = 256 seems hopeless (gadget term too big)
//
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
//
// Values for q2:
// 14 bits: 12289
// 16 bits: 61441
// 17 bits: 114689
// 18 bits: 249857
// 19 bits: 520193
// 20 bits: 1032193
// 21 bits: 2056193
// 22 bits: 4169729
// 23 bits: 8380417
// 24 bits: 16760833
// 25 bits: 33550337
// 26 bits: 67104769
// 27 bits: 134176769
// 28 bits: 268369921
// 29 bits: 536813569
// 30 bits: 1073692673
// 31 bits: 2147389441
// 32 bits: 4294955009

type ThePIR = respire!(PARAMS);
generate_main!(ThePIR);
