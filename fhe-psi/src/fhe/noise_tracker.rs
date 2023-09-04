//! The trivial (insecure) FHE scheme, but additionally tracks noise for GSW schemes.
use crate::fhe::fhe::*;
use crate::math::z_n::Z_N;

/// `M` refers to the implicit width of each matrix.
/// For standard GSW, this is just the lattice parameter `m`.
/// For RingGSW schemes, this should be `m * d`, where `d` is the degree of the polynomials.
/// `Q` is the (scalar) ciphertext modulus.
pub struct GSWNoiseTracker<const P: u64, const Q: u64, const M: u64> {}

#[derive(Clone, Debug)]
pub struct NoiseCiphertext<const P: u64, const Q: u64, const M: u64> {
    x: Z_N<P>,
    noise: u64,
}

impl<const P: u64, const Q: u64, const M: u64> EncryptionScheme for GSWNoiseTracker<P, Q, M> {
    type Plaintext = Z_N<P>;
    type Ciphertext = NoiseCiphertext<P, Q, M>;
    type PublicKey = ();
    type SecretKey = ();

    fn keygen() -> (Self::PublicKey, Self::SecretKey) {
        ((), ())
    }

    fn encrypt(_: &Self::PublicKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        Self::Ciphertext { x: *mu, noise: M }
    }

    fn encrypt_sk(_: &Self::SecretKey, mu: &Self::Plaintext) -> Self::Ciphertext {
        Self::Ciphertext { x: *mu, noise: 1 }
    }

    fn decrypt(_: &Self::SecretKey, ct: &Self::Ciphertext) -> Z_N<P> {
        ct.x
    }
}

impl<const P: u64, const Q: u64, const M: u64> FHEScheme for GSWNoiseTracker<P, Q, M> {
}

impl<const P: u64, const Q: u64, const M: u64> AddHomEncryptionScheme for GSWNoiseTracker<P, Q, M> {
    fn add_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        let new_noise = lhs.noise * M + P * rhs.noise;
        assert!(
            new_noise < Q / P,
            "ciphertext has become too noisy to decrypt correctly"
        );
        Self::Ciphertext {
            x: lhs.x * rhs.x,
            noise: new_noise,
        }
    }
}

impl<const P: u64, const Q: u64, const M: u64> MulHomEncryptionScheme for GSWNoiseTracker<P, Q, M> {
    fn mul_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        let new_noise = lhs.noise + rhs.noise;
        assert!(
            new_noise < Q / P,
            "ciphertext has become too noisy to decrypt correctly"
        );
        Self::Ciphertext {
            x: lhs.x * rhs.x,
            noise: new_noise,
        }
    }
}

// TODO: tests
