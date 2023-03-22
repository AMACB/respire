use fhe_psi::{fhe::*, gsw::*, intersect::*, z_n::*};
use std::collections::HashSet;

const TEST_P: u64 = GSW_TEST_PARAMS.P;

fn test_intersect_additive_gsw() {
    do_intersect_additive::<TEST_P, GSWTest>();
}

// TODO: make these generic over intersection functions
// TODO: make general test function that takes client_set, server_set, expected_set & generic intersection function

fn do_intersect_additive<const P: u64, FHE: FHEScheme<P>>()
where
    for<'a> &'a <FHE as FHEScheme<P>>::Ciphertext:
        CiphertextRef<P, <FHE as FHEScheme<P>>::Ciphertext>,
{
    let client_set: Vec<Z_N<P>> = vec![4_u64, 6, 7, 15].into_iter().map(Z_N::from).collect();
    let server_set: Vec<Z_N<P>> = vec![1_u64, 3, 4, 5, 7, 10, 12, 20]
        .into_iter()
        .map(Z_N::from)
        .collect();
    let expected: Vec<Z_N<P>> = vec![4_u64, 7].into_iter().map(Z_N::from).collect();

    assert_eq!(
        HashSet::<Z_N<P>>::from_iter(intersect_additive::<P, FHE>(&client_set, &server_set)),
        HashSet::<Z_N<P>>::from_iter(expected)
    );
}

fn main() {
    test_intersect_additive_gsw();
}
