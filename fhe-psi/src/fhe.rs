use std::ops::{Add, Mul};
use crate::z_n::Z_N;

pub trait CiphertextRef<const P: u64, Owned>:
Sized + Add<Self, Output = Owned> + Mul<Self, Output = Owned> + Mul<Z_N<P>, Output = Owned>
    where
            for<'a> &'a Owned: CiphertextRef<P, Owned>,
{
}

pub trait FHEScheme<const P: u64>: Sized
    where
            for<'a> &'a <Self as FHEScheme<P>>::Ciphertext:
    CiphertextRef<P, <Self as FHEScheme<P>>::Ciphertext>,
{
    type Ciphertext: Sized;
    type PublicKey: Sized;
    type SecretKey: Sized;

    fn keygen() -> (Self::PublicKey, Self::SecretKey);
    fn encrypt(pk: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext;
    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P>;
}