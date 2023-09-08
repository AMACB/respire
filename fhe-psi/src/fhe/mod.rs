//! A suite of generic FHE schemes.

#![allow(non_snake_case)]
pub mod fhe;
pub mod gsw;
pub mod gsw_crt;
pub mod gsw_utils;
pub mod noise_tracker;
pub mod ringgsw_crt;
pub mod ringgsw_ntt;
pub mod ringgsw_ntt_crt;
pub mod ringgsw_raw;

// TODO
// Handle errors more gracefully (e.g. don't panic on decryption failure)
// Documentation - explain why FHEScheme is a trait; params as generics/types.
