//! Generic FHE scheme traits.

use crate::math::ring_elem::{RingElement, RingElementRef};
use crate::math::z_n::Z_N;
use std::marker::PhantomData;
use std::ops::{Add, Mul};

pub trait CiphertextRef<const P: u64, Owned>:
    Sized + Clone + Add<Self, Output = Owned> + Mul<Self, Output = Owned> + Mul<Z_N<P>, Output = Owned>
where
    for<'a> &'a Owned: CiphertextRef<P, Owned>,
{
}

/// A generic encryption scheme.
pub trait EncryptionScheme {
    type Plaintext: Clone;
    type Ciphertext: Clone;
    type PublicKey: Clone;
    type SecretKey: Clone;

    fn keygen() -> (Self::PublicKey, Self::SecretKey);
    fn encrypt(pk: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext;
    fn encrypt_sk(sk: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext;
    fn decrypt(sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Self::Plaintext;
}

pub trait AddHomEncryptionScheme: EncryptionScheme {
    fn add_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext;
}

pub trait MulHomEncryptionScheme: EncryptionScheme {
    fn mul_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext;
}

pub trait FHEScheme: AddHomEncryptionScheme + MulHomEncryptionScheme {}

pub trait AddScalarEncryptionScheme<Scalar>: EncryptionScheme {
    fn add_scalar(lhs: &Self::Ciphertext, rhs: &Scalar) -> Self::Ciphertext;
}

pub trait MulScalarEncryptionScheme<Scalar>: EncryptionScheme {
    fn mul_scalar(lhs: &Self::Ciphertext, rhs: &Scalar) -> Self::Ciphertext;
}

/// The trivial (insecure) FHE scheme where encryption does nothing, useful for testing.
pub struct FHEInsecure<R: RingElement>
where
    for<'a> &'a R: RingElementRef<R>,
{
    _marker: PhantomData<R>,
}

impl<R: RingElement> AddHomEncryptionScheme for FHEInsecure<R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    fn add_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        lhs + rhs
    }
}

impl<R: RingElement> MulHomEncryptionScheme for FHEInsecure<R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    fn mul_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        lhs * rhs
    }
}

impl<R: RingElement> FHEScheme for FHEInsecure<R> where for<'a> &'a R: RingElementRef<R> {}

impl<R: RingElement> EncryptionScheme for FHEInsecure<R>
where
    for<'a> &'a R: RingElementRef<R>,
{
    type Plaintext = R;
    type Ciphertext = R;
    type PublicKey = ();
    type SecretKey = ();

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        ((), ())
    }

    fn encrypt(_: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        mu.clone()
    }

    fn encrypt_sk(_: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        mu.clone()
    }

    fn decrypt(_: &Self::SecretKey, ct: &Self::Ciphertext) -> Self::Plaintext {
        ct.clone()
    }
}
