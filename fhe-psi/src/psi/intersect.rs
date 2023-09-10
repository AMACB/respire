use crate::fhe::fhe::{FHEScheme, MulScalarEncryptionScheme};

use crate::math::int_mod::IntMod;
use crate::math::int_mod_poly::IntModPoly;
use crate::math::ring_elem::RingElement;
use crate::math::utils::floor_log;

// pub trait IntersectionStrategy {}

pub fn intersect_naive<const P: u64>(
    client_set: &Vec<IntMod<P>>,
    server_set: &Vec<IntMod<P>>,
) -> Vec<IntMod<P>> {
    let mut result_set = vec![];
    for i in server_set {
        if client_set.contains(&i) {
            result_set.push(*i);
        }
    }
    result_set
}

pub trait IntersectFHEScheme<const P: u64>:
    FHEScheme<Plaintext = IntMod<P>> + MulScalarEncryptionScheme<IntMod<P>>
{
}

/// Computes a PSI completely additively. For every client element `a`, the client computes and
/// sends encryptions of `a^i` for `i = 0..|B|`.
pub fn intersect_additive<const P: u64, FHE: IntersectFHEScheme<P>>(
    client_set: &Vec<IntMod<P>>,
    server_set: &Vec<IntMod<P>>,
) -> Vec<IntMod<P>> {
    let (pk, sk) = FHE::keygen();

    /* Server computes its polynomial and sets up interface for client */
    let mut server_polynomial = IntModPoly::<P>::one();
    for b in server_set {
        // Monomial x - b
        let monomial = vec![-*b, IntMod::one()].into();
        server_polynomial *= &monomial;
    }

    let server_polynomial_deg = server_polynomial.deg();
    assert!(server_polynomial_deg >= 0);

    let server_interface =
        |pk: &FHE::PublicKey, powers_of_a: &Vec<FHE::Ciphertext>| -> FHE::Ciphertext {
            let mut server_polynomial_iter = server_polynomial.coeff_iter();
            let mu = *server_polynomial_iter.next().unwrap();
            let mut result = FHE::encrypt(pk, &u64::from(mu).into());
            assert_eq!(powers_of_a.len(), server_polynomial_iter.len());
            for (pow_of_a, coeff) in powers_of_a.iter().zip(server_polynomial_iter) {
                result = FHE::add_hom(&result, &(FHE::mul_scalar(pow_of_a, coeff)));
            }
            // TODO: blinding factor
            result
        };

    /* Client interacts with server and computes intersection */
    let mut result_set = vec![];

    for a in client_set {
        let mut powers_of_a: Vec<FHE::Ciphertext> =
            Vec::with_capacity(server_polynomial_deg as usize);
        let mut curr_power_of_a: IntMod<P> = 1_u64.into();
        // TODO check: this should send 1? Off by one?
        for _ in 0..server_polynomial_deg {
            curr_power_of_a *= a;
            powers_of_a.push(FHE::encrypt_sk(&sk, &curr_power_of_a));
        }

        let result = server_interface(&pk, &powers_of_a);
        let result = FHE::decrypt(&sk, &result);
        if result == IntMod::<P>::zero() {
            result_set.push(*a);
        }
    }

    result_set
}

/// Computes a PSI with log-depth multiplications. For every client element `a`, the client computes
/// and sends encryptions of `a^(2^i)` for `i = 0..floor(log2(|B|))`.
pub fn intersect_log_multiplicative<const P: u64, FHE: IntersectFHEScheme<P>>(
    client_set: &Vec<IntMod<P>>,
    server_set: &Vec<IntMod<P>>,
) -> Vec<IntMod<P>> {
    let (pk, sk) = FHE::keygen();

    /* Server computes its polynomial and sets up interface for client */
    let mut server_polynomial = IntModPoly::<P>::one();
    for b in server_set {
        // Monomial x - b
        let monomial = vec![-*b, IntMod::one()].into();
        server_polynomial *= &monomial;
    }

    let server_polynomial_deg = server_polynomial.deg();
    assert!(server_polynomial_deg >= 0);

    let server_interface =
        |pk: &FHE::PublicKey, powers_of_a_bin: &Vec<FHE::Ciphertext>| -> FHE::Ciphertext {
            let mut server_polynomial_iter = server_polynomial.coeff_iter().enumerate();
            let mut result = FHE::encrypt(
                pk,
                &u64::from(*server_polynomial_iter.next().unwrap().1).into(),
            );
            assert_eq!(
                powers_of_a_bin.len(),
                floor_log(2, server_polynomial_iter.len() as u64) + 1
            );
            for (mut i, coeff) in server_polynomial_iter {
                assert!(i > 0);
                let mut bin_idx = 0;
                // Find the first nonzero bit to avoid an extra multiplication
                while i % 2 == 0 {
                    bin_idx += 1;
                    i /= 2;
                }
                let mut a_pow_i = powers_of_a_bin[bin_idx].clone();
                bin_idx += 1;
                i /= 2;
                while i > 0 {
                    if i % 2 == 1 {
                        a_pow_i = FHE::mul_hom(&a_pow_i, &powers_of_a_bin[bin_idx]);
                    }
                    bin_idx += 1;
                    i /= 2;
                }
                result = FHE::add_hom(&result, &FHE::mul_scalar(&a_pow_i, coeff));
            }
            // TODO: blinding factor
            result
        };

    /* Client interacts with server and computes intersection */
    let mut result_set = vec![];

    for a in client_set {
        let log_degree: u64 = floor_log(2, server_polynomial_deg as u64) as u64 + 1;
        let mut powers_of_a: Vec<FHE::Ciphertext> = Vec::with_capacity(log_degree as usize);
        let mut curr_power_of_a: IntMod<P> = *a;
        for _ in 0..log_degree {
            // powers_of_a.push(FHE::encrypt(&pk, curr_power_of_a));
            powers_of_a.push(FHE::encrypt_sk(&sk, &curr_power_of_a));
            curr_power_of_a *= curr_power_of_a;
        }

        let result = server_interface(&pk, &powers_of_a);
        let result = FHE::decrypt(&sk, &result);
        if result == IntMod::<P>::zero() {
            result_set.push(*a);
        }
    }

    result_set
}

#[cfg(test)]
mod test {
    use crate::fhe::fhe::FHEInsecure;
    use crate::fhe::gsw::GSWTest;
    use crate::fhe::gsw_crt::GSWCRTTest;
    use crate::fhe::ringgsw_crt::RingGSWCRTTest;
    use crate::fhe::ringgsw_ntt::{RingGSWNTTTest, RingGSWNTTTestMedium};
    use crate::fhe::ringgsw_ntt_crt::{RingGSWNTTCRTTest, RingGSWNTTCRTTestMedium};
    use crate::fhe::ringgsw_raw::{
        RingGSWRawTest, RingGSWRawTestMedium, RING_GSW_RAW_TEST_MEDIUM_PARAMS,
        RING_GSW_RAW_TEST_PARAMS,
    };
    use std::collections::HashSet;

    const TEST_P: u64 = RING_GSW_RAW_TEST_PARAMS.P;

    const TEST_MEDIUM_P: u64 = RING_GSW_RAW_TEST_MEDIUM_PARAMS.P;

    // Naive insecure scheme for testing

    type IntersectFHEInsecure<const P: u64> = FHEInsecure<IntMod<P>>;

    impl<const P: u64> IntersectFHEScheme<P> for IntersectFHEInsecure<P> {}

    // Actual schemes

    impl IntersectFHEScheme<TEST_P> for GSWTest {}
    impl IntersectFHEScheme<TEST_P> for GSWCRTTest {}
    impl IntersectFHEScheme<TEST_P> for RingGSWRawTest {}
    impl IntersectFHEScheme<TEST_P> for RingGSWCRTTest {}
    impl IntersectFHEScheme<TEST_P> for RingGSWNTTTest {}
    impl IntersectFHEScheme<TEST_P> for RingGSWNTTCRTTest {}
    impl IntersectFHEScheme<TEST_MEDIUM_P> for RingGSWRawTestMedium {}
    impl IntersectFHEScheme<TEST_MEDIUM_P> for RingGSWNTTTestMedium {}
    impl IntersectFHEScheme<TEST_MEDIUM_P> for RingGSWNTTCRTTestMedium {}

    use super::*;

    #[test]
    fn test_intersect_naive() {
        const P: u64 = TEST_P;
        let client_set: Vec<IntMod<P>> = vec![4_u64, 6, 7, 15]
            .into_iter()
            .map(IntMod::from)
            .collect();
        let server_set: Vec<IntMod<P>> = vec![1_u64, 3, 4, 5, 7, 10, 12, 20]
            .into_iter()
            .map(IntMod::from)
            .collect();
        let expected: Vec<IntMod<P>> = vec![4_u64, 7].into_iter().map(IntMod::from).collect();

        assert_eq!(
            HashSet::<IntMod<P>>::from_iter(intersect_naive(&client_set, &server_set)),
            HashSet::<IntMod<P>>::from_iter(expected)
        );
    }

    // Additive-only tests

    #[test]
    fn test_intersect_additive_insecure() {
        do_intersect_additive::<TEST_P, IntersectFHEInsecure<TEST_P>>();
    }

    #[test]
    fn test_intersect_additive_gsw() {
        do_intersect_additive::<TEST_P, GSWTest>();
    }

    #[test]
    fn test_intersect_additive_gsw_crt() {
        do_intersect_additive::<TEST_P, GSWCRTTest>();
    }

    #[test]
    fn test_intersect_additive_ringgsw_raw() {
        do_intersect_additive::<TEST_P, RingGSWRawTest>();
    }

    #[test]
    fn test_intersect_additive_ringgsw_ntt() {
        do_intersect_additive::<TEST_P, RingGSWNTTTest>();
    }

    // Multiplicative-log-depth tests

    #[test]
    fn test_intersect_log_multiplicative_insecure() {
        do_intersect_log_multiplicative::<TEST_P, IntersectFHEInsecure<TEST_P>>();
    }

    #[test]
    fn test_intersect_log_multiplicative_ringgsw_raw() {
        do_intersect_log_multiplicative::<TEST_P, RingGSWRawTest>();
    }

    #[test]
    fn test_intersect_log_multiplicative_ringgsw_ntt() {
        do_intersect_log_multiplicative::<TEST_P, RingGSWNTTTest>();
    }

    #[test]
    fn test_intersect_log_multiplicative_ringgsw_crt() {
        do_intersect_log_multiplicative::<TEST_P, RingGSWCRTTest>();
    }

    #[test]
    fn test_intersect_log_multiplicative_ringgsw_crt_ntt() {
        do_intersect_log_multiplicative::<TEST_P, RingGSWNTTCRTTest>();
    }

    #[ignore]
    #[test]
    fn test_intersect_log_multiplicative_ringgsw_raw_medium() {
        do_intersect_log_multiplicative::<TEST_MEDIUM_P, RingGSWRawTestMedium>();
    }

    #[ignore]
    #[test]
    fn test_intersect_log_multiplicative_ringgsw_ntt_medium() {
        do_intersect_log_multiplicative::<TEST_MEDIUM_P, RingGSWNTTTestMedium>();
    }

    #[ignore]
    #[test]
    fn test_intersect_log_multiplicative_ringgsw_ntt_crt_medium() {
        do_intersect_log_multiplicative::<TEST_MEDIUM_P, RingGSWNTTCRTTestMedium>();
    }

    // TODO: make these generic over intersection functions
    // TODO: make general test function that takes client_set, server_set, expected_set & generic intersection function

    fn do_intersect_additive<const P: u64, FHE: IntersectFHEScheme<P>>() {
        let client_set: Vec<IntMod<P>> = vec![4_u64, 6, 7, 15]
            .into_iter()
            .map(IntMod::from)
            .collect();
        let server_set: Vec<IntMod<P>> = vec![1_u64, 3, 4, 5, 7, 10, 12, 20]
            .into_iter()
            .map(IntMod::from)
            .collect();
        let expected: Vec<IntMod<P>> = vec![4_u64, 7].into_iter().map(IntMod::from).collect();

        assert_eq!(
            HashSet::<IntMod<P>>::from_iter(intersect_additive::<P, FHE>(&client_set, &server_set)),
            HashSet::<IntMod<P>>::from_iter(expected)
        );
    }

    fn do_intersect_log_multiplicative<const P: u64, FHE: IntersectFHEScheme<P>>() {
        let client_set: Vec<IntMod<P>> = vec![4_u64, 6, 7, 15]
            .into_iter()
            .map(IntMod::from)
            .collect();
        let server_set: Vec<IntMod<P>> = vec![1_u64, 3, 4, 5, 7, 10, 12, 20]
            .into_iter()
            .map(IntMod::from)
            .collect();
        let expected: Vec<IntMod<P>> = vec![4_u64, 7].into_iter().map(IntMod::from).collect();

        assert_eq!(
            HashSet::<IntMod<P>>::from_iter(intersect_log_multiplicative::<P, FHE>(
                &client_set,
                &server_set
            )),
            HashSet::<IntMod<P>>::from_iter(expected)
        );
    }

    // fn do_intersect_additive_large<const P: u64, FHE: FHEScheme<P>>()
    // where
    //     for<'a> &'a <FHE as FHEScheme<P>>::Ciphertext:
    //         CiphertextRef<P, <FHE as FHEScheme<P>>::Ciphertext>,
    // {
    //     let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1984);
    //     let NUM_CLIENT = 2000;
    //     let NUM_SERVER = 2000;
    //     let NUM_BOTH = 1000;
    //
    //     let mut client_set: Vec<Z_N<P>> = vec![];
    //     let mut server_set: Vec<Z_N<P>> = vec![];
    //     let mut expected: Vec<Z_N<P>> = vec![];
    //     let mut seen: HashSet<Z_N<P>> = HashSet::new();
    //
    //     for _ in 0..NUM_CLIENT {
    //         let n = Z_N::<P>::random();
    //         assert!(!seen.contains(&n));
    //         seen.insert(n);
    //         client_set.push(n);
    //     }
    //
    //     for _ in 0..NUM_SERVER {
    //         let n = Z_N::<P>::random();
    //         assert!(!seen.contains(&n));
    //         seen.insert(n);
    //         server_set.push(n);
    //     }
    //
    //     for _ in 0..NUM_BOTH {
    //         let n = Z_N::<P>::random();;
    //         assert!(!seen.contains(&n));
    //         seen.insert(n);
    //         client_set.push(n);
    //         server_set.push(n);
    //         expected.push(n);
    //     }
    //
    //     let expected = HashSet::<Z_N<P>>::from_iter(expected);
    //
    //     assert_eq!(
    //         HashSet::<Z_N<P>>::from_iter(intersect_naive(&client_set, &server_set)),
    //         expected
    //     );
    //     assert_eq!(
    //         HashSet::<Z_N<P>>::from_iter(intersect_poly_no_encrypt::<{ PolyU32::<0>::P }>(
    //             &client_set,
    //             &server_set,
    //         )),
    //         expected
    //     );
    // }
}
