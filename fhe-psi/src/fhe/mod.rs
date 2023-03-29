//! FHE (Fully Homomorphic Encryption) specific constructs.
pub mod discrete_gaussian;
pub mod fhe;
pub mod gadget;
pub mod gsw;
pub mod ringgsw;

// TODO
// Handle errors more gracefully (e.g. don't panic on decryption failure)