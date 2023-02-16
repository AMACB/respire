use crate::fhe::{CiphertextRef, FHEScheme};

use crate::polynomial::PolynomialZ_N;
use crate::ring_elem::RingElement;
use crate::z_n::Z_N;

// pub trait IntersectionStrategy {}

pub fn intersect_naive<const P: u64>(
    client_set: &Vec<Z_N<P>>,
    server_set: &Vec<Z_N<P>>,
) -> Vec<Z_N<P>> {
    let mut result_set = vec![];
    for i in server_set {
        if client_set.contains(&i) {
            result_set.push(*i);
        }
    }
    result_set
}

pub fn intersect_additive<const P: u64, FHE: FHEScheme<P>>(
    client_set: &Vec<Z_N<P>>,
    server_set: &Vec<Z_N<P>>,
) -> Vec<Z_N<P>>
where
    for<'a> &'a <FHE as FHEScheme<P>>::Ciphertext:
        CiphertextRef<P, <FHE as FHEScheme<P>>::Ciphertext>,
{
    let (pk, sk) = FHE::keygen();

    /* Server computes its polynomial and sets up interface for client */
    let mut server_polynomial = PolynomialZ_N::<P>::one();
    for i in server_set {
        // Monomial x - i
        let monomial = vec![-*i, Z_N::one()].into();
        server_polynomial *= &monomial;
    }

    let server_polynomial_deg = server_polynomial.deg();
    assert!(server_polynomial_deg >= 0);

    let server_interface = |pk: &<FHE as FHEScheme<P>>::PublicKey,
                            powers_of_i: &Vec<FHE::Ciphertext>|
     -> FHE::Ciphertext {
        let mut server_polynomial_iter = server_polynomial.coeff_iter();
        let mut result = FHE::encrypt(pk, *server_polynomial_iter.next().unwrap());
        assert_eq!(powers_of_i.len(), server_polynomial_iter.len());
        for (pow_of_i, coeff) in powers_of_i.iter().zip(server_polynomial_iter) {
            result = &result + &(pow_of_i * *coeff);
        }
        // TODO: blinding factor
        result
    };

    /* Client interacts with server and computes intersection */
    let mut result_set = vec![];

    for i in client_set {
        let mut powers_of_i: Vec<FHE::Ciphertext> =
            Vec::with_capacity(server_polynomial_deg as usize);
        let mut curr_power_of_i: Z_N<P> = 1_u64.into();
        for _ in 0..server_polynomial_deg {
            curr_power_of_i *= i;
            powers_of_i.push(FHE::encrypt(&pk, curr_power_of_i));
        }

        let result = server_interface(&pk, &powers_of_i);
        let result = FHE::decrypt(&sk, &result);
        if result == Z_N::<P>::zero() {
            result_set.push(*i);
        }
    }

    result_set
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    // use rand::{Rng, SeedableRng};
    use crate::params::TEST_PARAMS_RAW;

    use super::*;

    #[test]
    fn test_intersect_naive() {
        const P: u64 = TEST_PARAMS_RAW.P;
        let client_set: Vec<Z_N<P>> = vec![4_u64, 6, 7, 15].into_iter().map(Z_N::from).collect();
        let server_set: Vec<Z_N<P>> = vec![1_u64, 3, 4, 5, 7, 10, 12, 20]
            .into_iter()
            .map(Z_N::from)
            .collect();
        let expected: Vec<Z_N<P>> = vec![4_u64, 7].into_iter().map(Z_N::from).collect();

        assert_eq!(
            HashSet::<Z_N<P>>::from_iter(intersect_naive(&client_set, &server_set)),
            HashSet::<Z_N<P>>::from_iter(expected)
        );
    }

    // #[test]
    // fn test_intersect_poly_no_encrypt() {
    //     let client_set = vec![4, 6, 7, 15];
    //     let server_set = vec![1, 3, 4, 5, 7, 10, 12, 20];
    //     let expected = HashSet::<u32>::from_iter(vec![4, 7]);
    //
    //     assert_eq!(
    //         HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
    //         expected
    //     );
    //     assert_eq!(
    //         HashSet::<u32>::from_iter(intersect_poly_no_encrypt::<41>(&client_set, &server_set)),
    //         expected
    //     );
    // }
    //
    // #[test]
    // fn test_intersect_poly_gsw() {
    //     let client_set = vec![4, 6, 7, 15];
    //     let server_set = vec![1, 3, 4, 5, 7, 10, 12, 20];
    //     let expected = HashSet::<u32>::from_iter(vec![4, 7]);
    //
    //     assert_eq!(
    //         HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
    //         expected
    //     );
    //     assert_eq!(
    //         HashSet::<u32>::from_iter(intersect_poly_gsw(&client_set, &server_set)),
    //         expected
    //     );
    // }
    //
    // #[test]
    // fn test_intersect_poly_no_encrypt_large() {
    //     let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1984);
    //     let NUM_CLIENT = 2000;
    //     let NUM_SERVER = 2000;
    //     let NUM_BOTH = 1000;
    //
    //     let mut client_set = vec![];
    //     let mut server_set = vec![];
    //     let mut expected = vec![];
    //     let mut seen = HashSet::new();
    //
    //     for _ in 0..NUM_CLIENT {
    //         let n = rng.gen::<u32>();
    //         assert!(!seen.contains(&n));
    //         seen.insert(n);
    //         client_set.push(n);
    //     }
    //
    //     for _ in 0..NUM_SERVER {
    //         let n = rng.gen::<u32>();
    //         assert!(!seen.contains(&n));
    //         seen.insert(n);
    //         server_set.push(n);
    //     }
    //
    //     for _ in 0..NUM_BOTH {
    //         let n = rng.gen::<u32>();
    //         assert!(!seen.contains(&n));
    //         seen.insert(n);
    //         client_set.push(n);
    //         server_set.push(n);
    //         expected.push(n);
    //     }
    //
    //     let expected = HashSet::<u32>::from_iter(expected);
    //
    //     assert_eq!(
    //         HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
    //         expected
    //     );
    //     assert_eq!(
    //         HashSet::<u32>::from_iter(intersect_poly_no_encrypt::<{ PolyU32::<0>::P }>(
    //             &client_set,
    //             &server_set,
    //         )),
    //         expected
    //     );
    // }
}
