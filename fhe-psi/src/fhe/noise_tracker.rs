//! The trivial (insecure) FHE scheme, but additionally tracks noise for GSW schemes.
use crate::fhe::fhe::*;
use crate::math::z_n::Z_N;
use std::ops::{Add, Mul};

/// `M` refers to the implicit width of each matrix.
/// For standard GSW, this is just the lattice parameter `m`.
/// For RingGSW schemes, this should be `m * d`, where `d` is the degree of the polynomials.
/// `Q` is the (scalar) ciphertext modulus.
pub struct GSWNoiseTracker<const P: u64, const Q: u64, const M: u64> {}

#[derive(Clone, Debug)]
pub struct Ciphertext<const P: u64, const Q: u64, const M: u64> {
    x: Z_N<P>,
    noise: u64,
}

impl<'a, const P: u64, const Q: u64, const M: u64> CiphertextRef<P, Ciphertext<P, Q, M>>
    for &'a Ciphertext<P, Q, M>
{
}

impl<'a, const P: u64, const Q: u64, const M: u64> Add for &'a Ciphertext<P, Q, M> {
    type Output = Ciphertext<P, Q, M>;
    fn add(self, rhs: &Ciphertext<P, Q, M>) -> Self::Output {
        let new_noise = self.noise + rhs.noise;
        assert!(
            new_noise < Q / P,
            "ciphertext has become too noisy to decrypt correctly"
        );
        Ciphertext {
            x: self.x * rhs.x,
            noise: new_noise,
        }
    }
}

impl<'a, const P: u64, const Q: u64, const M: u64> Mul for &'a Ciphertext<P, Q, M> {
    type Output = Ciphertext<P, Q, M>;
    fn mul(self, rhs: &Ciphertext<P, Q, M>) -> Self::Output {
        let new_noise = self.noise * M + rhs.noise;
        assert!(
            new_noise < Q / P,
            "ciphertext has become too noisy to decrypt correctly"
        );
        Ciphertext {
            x: self.x * rhs.x,
            noise: new_noise,
        }
    }
}

impl<'a, const P: u64, const Q: u64, const M: u64> Mul<Z_N<P>> for &'a Ciphertext<P, Q, M> {
    type Output = Ciphertext<P, Q, M>;
    fn mul(self, rhs: Z_N<P>) -> Self::Output {
        let new_noise = self.noise * M + M;
        assert!(
            new_noise < Q / P,
            "ciphertext has become too noisy to decrypt correctly"
        );
        Ciphertext {
            x: self.x * rhs,
            noise: new_noise,
        }
    }
}

impl<const P: u64, const Q: u64, const M: u64> FHEScheme<P> for GSWNoiseTracker<P, Q, M> {
    type Ciphertext = Ciphertext<P, Q, M>;
    type PublicKey = ();
    type SecretKey = ();

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        ((), ())
    }

    fn encrypt(_: &Self::PublicKey, mu: Z_N<P>) -> Self::Ciphertext {
        Ciphertext { x: mu, noise: M }
    }

    fn encrypt_sk(_: &Self::SecretKey, mu: Z_N<P>) -> Self::Ciphertext {
        Ciphertext { x: mu, noise: 1 }
    }

    fn decrypt(_: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        ct.x
    }
}

// TODO: tests
