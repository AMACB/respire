use crate::gsw::gsw;
use crate::params::TEST_PARAMS;
use crate::polynomial::PolyU32;
use crate::ring_elem::RingElement;
use crate::z_n::Z_N;

pub fn intersect_naive(client_set: &Vec<u32>, server_set: &Vec<u32>) -> Vec<u32> {
    let mut result_set = vec![];
    for i in server_set {
        if client_set.contains(&i) {
            result_set.push(*i);
        }
    }
    result_set
}

pub fn intersect_poly_no_encrypt<const P: u32>(
    client_set: &Vec<u32>,
    server_set: &Vec<u32>,
) -> Vec<u32> {
    let mut result_set = vec![];
    let mut p = PolyU32::<P>::new(vec![1]);
    for i in server_set {
        p = p.mul(&PolyU32::<P>::new(vec![P - *i, 1]));
    }

    for i in client_set {
        if p.eval(*i) == 0 {
            result_set.push(*i);
        }
    }

    result_set
}

pub fn intersect_poly_gsw(client_set: &Vec<u32>, server_set: &Vec<u32>) -> Vec<u32> {
    const P: u32 = PolyU32::<0>::P;
    let mut result_set = vec![];
    let mut p = PolyU32::<P>::new(vec![1]);
    for i in server_set {
        p = p.mul(&PolyU32::<P>::new(vec![P - *i, 1]));
    }

    let (A, s_T) = gsw::keygen(TEST_PARAMS);
    for i in client_set {
        let mu = Z_N::new_u(*i as u64);
        let ct = gsw::encrypt(&A, &mu);
        let eval_ct = p.eval_gsw(&A, &ct);
        // TODO: blinding factor for security
        // eprintln!("p({}) = {:?}", *i, gsw::decrypt(&s_T, &eval_ct));
        if gsw::decrypt(&s_T, &eval_ct) == Z_N::zero() {
            result_set.push(*i);
        }
    }

    result_set
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_intersect_poly_no_encrypt() {
        let client_set = vec![4, 6, 7, 15];
        let server_set = vec![1, 3, 4, 5, 7, 10, 12, 20];
        let expected = HashSet::<u32>::from_iter(vec![4, 7]);

        assert_eq!(
            HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
            expected
        );
        assert_eq!(
            HashSet::<u32>::from_iter(intersect_poly_no_encrypt::<41>(&client_set, &server_set)),
            expected
        );
    }

    #[test]
    fn test_intersect_poly_gsw() {
        let client_set = vec![4, 6, 7, 15];
        let server_set = vec![1, 3, 4, 5, 7, 10, 12, 20];
        let expected = HashSet::<u32>::from_iter(vec![4, 7]);

        assert_eq!(
            HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
            expected
        );
        assert_eq!(
            HashSet::<u32>::from_iter(intersect_poly_gsw(&client_set, &server_set)),
            expected
        );
    }

    #[test]
    fn test_intersect_poly_no_encrypt_large() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1984);
        let NUM_CLIENT = 2000;
        let NUM_SERVER = 2000;
        let NUM_BOTH = 1000;

        let mut client_set = vec![];
        let mut server_set = vec![];
        let mut expected = vec![];
        let mut seen = HashSet::new();

        for _ in 0..NUM_CLIENT {
            let n = rng.gen::<u32>();
            assert!(!seen.contains(&n));
            seen.insert(n);
            client_set.push(n);
        }

        for _ in 0..NUM_SERVER {
            let n = rng.gen::<u32>();
            assert!(!seen.contains(&n));
            seen.insert(n);
            server_set.push(n);
        }

        for _ in 0..NUM_BOTH {
            let n = rng.gen::<u32>();
            assert!(!seen.contains(&n));
            seen.insert(n);
            client_set.push(n);
            server_set.push(n);
            expected.push(n);
        }

        let expected = HashSet::<u32>::from_iter(expected);

        assert_eq!(
            HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
            expected
        );
        assert_eq!(
            HashSet::<u32>::from_iter(intersect_poly_no_encrypt::<{ PolyU32::<0>::P }>(
                &client_set,
                &server_set
            )),
            expected
        );
    }
}
