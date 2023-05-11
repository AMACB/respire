//! FHE (Fully Homomorphic Encryption) specific constructs.
pub mod discrete_gaussian;
pub mod fhe;
pub mod gadget;
pub mod gsw;
pub mod gsw_utils;
pub mod ringgsw;
pub mod ringgsw_ntt;

// TODO
// Handle errors more gracefully (e.g. don't panic on decryption failure)
// Documentation - explain why FHEScheme is a trait; params as generics/types.
