//! The trivial (insecure) FHE scheme, but additionally tracks noise for GSW schemes.
use crate::fhe::fhe::*;
use crate::math::int_mod::IntMod;

/// `M` refers to the implicit width of each matrix.
/// For standard GSW, this is just the lattice parameter `m`.
/// For RingGSW schemes, this should be `m * d`, where `d` is the degree of the polynomials.
/// `Q` is the (scalar) ciphertext modulus.
pub struct GSWNoiseTracker<const P: u64, const Q: u64, const M: u64> {}

#[derive(Clone, Debug)]
pub struct NoiseCiphertext<const P: u64, const Q: u64, const M: u64> {
    x: IntMod<P>,
    noise: u64,
}

impl<const P: u64, const Q: u64, const M: u64> EncryptionScheme for GSWNoiseTracker<P, Q, M> {
    type Plaintext = IntMod<P>;
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

    fn decrypt(_: &Self::SecretKey, ct: &Self::Ciphertext) -> IntMod<P> {
        ct.x
    }
}

impl<const P: u64, const Q: u64, const M: u64> FHEScheme for GSWNoiseTracker<P, Q, M> {}

impl<const P: u64, const Q: u64, const M: u64> AddHomEncryptionScheme for GSWNoiseTracker<P, Q, M> {
    fn add_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        let new_noise = lhs.noise + rhs.noise;
        assert!(
            new_noise < Q / P,
            "ciphertext has become too noisy to decrypt correctly"
        );
        Self::Ciphertext {
            x: lhs.x + rhs.x,
            noise: new_noise,
        }
    }
}

impl<const P: u64, const Q: u64, const M: u64> MulHomEncryptionScheme for GSWNoiseTracker<P, Q, M> {
    fn mul_hom(lhs: &Self::Ciphertext, rhs: &Self::Ciphertext) -> Self::Ciphertext {
        let new_noise = lhs.noise * M + (P-1) * rhs.noise;
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

impl<const P: u64, const Q: u64, const M: u64> AddScalarEncryptionScheme<IntMod<P>> for GSWNoiseTracker<P, Q, M> {
    fn add_scalar(lhs: &Self::Ciphertext, rhs: &Self::Plaintext) -> Self::Ciphertext {
        Self::Ciphertext {
            x: lhs.x + *rhs,
            noise: lhs.noise
        }
    }
}

impl<const P: u64, const Q: u64, const M: u64> MulScalarEncryptionScheme<IntMod<P>> for GSWNoiseTracker<P, Q, M> {
    fn mul_scalar(lhs: &Self::Ciphertext, rhs: &Self::Plaintext) -> Self::Ciphertext {
        let new_noise = lhs.noise * M;
        Self::Ciphertext {
            x: lhs.x * *rhs,
            noise: new_noise
        }
    }
}

impl<const P: u64, const Q: u64, const M: u64> NegEncryptionScheme for GSWNoiseTracker<P, Q, M> {
    fn negate(lhs: &Self::Ciphertext) -> Self::Ciphertext {
        Self::Ciphertext {
            x: -lhs.x,
            noise: lhs.noise
        }
    }
}

// TODO: tests
