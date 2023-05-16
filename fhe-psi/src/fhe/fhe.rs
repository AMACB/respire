//! Generic FHE scheme traits.

use crate::math::z_n::Z_N;
use std::ops::{Add, Mul};

pub trait CiphertextRef<const P: u64, Owned>:
    Sized + Clone + Add<Self, Output = Owned> + Mul<Self, Output = Owned> + Mul<Z_N<P>, Output = Owned>
where
    for<'a> &'a Owned: CiphertextRef<P, Owned>,
{
}

/// A generic FHE scheme type that supports keygen, encrypt, decrypt, addition, multiplication, and
/// scalar multiplication.
/// TODO make message and scalar type generic
/// TODO
pub trait FHEScheme<const P: u64>: Sized
where
    for<'a> &'a <Self as FHEScheme<P>>::Ciphertext:
        CiphertextRef<P, <Self as FHEScheme<P>>::Ciphertext>,
{
    type Ciphertext: Sized + Clone;
    type PublicKey: Sized + Clone;
    type SecretKey: Sized + Clone;

    fn keygen() -> (Self::PublicKey, Self::SecretKey);
    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext;
    fn encrypt_sk(sk: &Self::SecretKey, mu: Z_N<P>) -> Self::Ciphertext;
    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P>;
}

/// The trivial (insecure) FHE scheme where encryption does nothing, useful for testing.
pub struct FHEInsecure {}

impl<const P: u64> CiphertextRef<P, Z_N<P>> for &Z_N<P> {}

impl<const P: u64> Mul<Z_N<P>> for &Z_N<P> {
    type Output = Z_N<P>;
    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        *self * rhs
    }
}

impl<const P: u64> FHEScheme<P> for FHEInsecure {
    type Ciphertext = Z_N<P>;
    type PublicKey = ();
    type SecretKey = ();

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        ((), ())
    }

    fn encrypt(_: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        mu
    }

    fn encrypt_sk(_: &Self::SecretKey, mu: Z_N<P>) -> Self::Ciphertext {
        mu
    }

    fn decrypt(_: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        *ct
    }
}

// TODO: Add trivial FHE scheme that tracks noise
