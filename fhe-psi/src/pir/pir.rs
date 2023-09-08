// use crate::fhe::ringgsw_ntt::RingGSWNTT;
// use crate::math::matrix::Matrix;
// use crate::math::utils::ceil_log;
//
// pub trait PIRScheme {
//     type Database;
//     type Index;
//     type Record;
//
//     type PublicParams;
//     type QueryKey;
//     type State;
//     type Query;
//     type Response;
//
//     fn setup() -> (Self::PublicParams, Self::QueryKey);
//
//     fn query(qk: &Self::QueryKey, idx: &Self::Index) -> (Self::State, Self::Query);
//
//     fn answer(
//         pp: &Self::PublicParams,
//         database: &Self::Database,
//         q: &Self::Query,
//     ) -> Self::Response;
//
//     fn extract(qk: &Self::QueryKey, st: &Self::State, r: &Self::Response) -> Self::Record;
// }
//
// pub struct SPIRAL<
//     const N: usize,
//     const P: u64,
//     const Q: u64,
//     const D: usize,
//     const W: u64,
//     const Q1: u64,
//     const Q2: u64,
//     const Z_GSW: u64,
//     const NOISE_WIDTH_MILLIONTHS: u64,
//     const ETA1: usize,
//     const ETA2: usize,
// > {}
//
// // trait SPIRALTypes {
// //     type GSW;
// //     type MatrixRegev;
// //     type PlaintextRing;
// //     type CiphertextRing;
// // }
// //
// // impl<
// //     const N: usize,
// //     const P: u64,
// //     const Q: u64,
// //     const D: usize,
// //     const W: u64,
// //     const Q1: u64,
// //     const Q2: u64,
// //     const Z_GSW: u64,
// //     const NOISE_WIDTH_MILLIONTHS: u64,
// //     const ETA1: usize,
// //     const ETA2: usize,
// // > SPIRALTypes for SPIRAL<N, P, Q, D, W, Q1, Q2, Z_GSW, NOISE_WIDTH_MILLIONTHS, ETA1, ETA2> {
// //     type GSW = RingGSWNTT<{N - 1}, N, {N as usize * ceil_log(Z_GSW, Q)}, P, Q, D, W, Z_GSW, {ceil_log(Z_GSW, Q)}, 6400>;
// //     type MatrixRegev = ();
// //     type PlaintextRing = Z_N_CycloRaw<D, P>;
// //     type CiphertextRing = Z_N_CycloNTT<D, Q, W>;
// // }
//
// impl<
//         const N: usize,
//         const P: u64,
//         const Q: u64,
//         const D: usize,
//         const W: u64,
//         const Q1: u64,
//         const Q2: u64,
//         const Z_GSW: u64,
//         const NOISE_WIDTH_MILLIONTHS: u64,
//         const ETA1: usize,
//         const ETA2: usize,
//         const DB_SIZE: usize
//     > PIRScheme for SPIRAL<N, P, Q, D, W, Q1, Q2, Z_GSW, NOISE_WIDTH_MILLIONTHS, ETA1, ETA2>
// {
//
//     type Database = [Self::Record; DB_SIZE];
//     // type Index = [Self::GSW; (1usize << ETA1)];
//     type Index = ();
//     type Record = Matrix<N, N, Self::PlaintextRing>;
//     type PublicParams = ();
//     type QueryKey = ();
//     type State = ();
//     type Query = ();
//     type Response = ();
//
//     fn setup() -> (Self::PublicParams, Self::QueryKey) {
//         todo!()
//     }
//
//     fn query(qk: &Self::QueryKey, idx: &Self::Index) -> (Self::State, Self::Query) {
//         todo!()
//     }
//
//     fn answer(
//         pp: &Self::PublicParams,
//         database: &Self::Database,
//         q: &Self::Query,
//     ) -> Self::Response {
//         todo!()
//     }
//
//     fn extract(qk: &Self::QueryKey, st: &Self::State, r: &Self::Response) -> Self::Record {
//         todo!()
//     }
// }
//
// #[cfg(test)]
// mod test {
//     #[test]
//     fn hello_world() {}
// }
