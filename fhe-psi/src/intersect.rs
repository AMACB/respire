use crate::polynomial::PolyU32;

pub fn intersect_naive(client_set: &Vec<u32>, server_set: &Vec<u32>) -> Vec<u32> {
    let mut result_set = vec![];
    for i in server_set {
        if client_set.contains(&i) {
            result_set.push(*i);
        }
    }
    result_set
}

pub fn intersect_polynomial(client_set: &Vec<u32>, server_set: &Vec<u32>) -> Vec<u32> {
    const P: u32 = PolyU32::<0>::P;
    let mut result_set = vec![];
    let mut p = PolyU32::<P>::new(vec![1]);
    for i in server_set {
        p = p.mul(&PolyU32::<P>::new(vec![P - *i, 1]));
    }

    dbg!(&server_set, &client_set, &p);

    for i in client_set {
        if p.eval(*i) == 0 {
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
    fn test_intersect() {
        let client_set = vec![4, 6, 7, 15];
        let server_set = vec![1, 3, 4, 5, 7, 10, 12, 20];
        let expected = HashSet::<u32>::from_iter(vec![4, 7]);

        assert_eq!(
            HashSet::<u32>::from_iter(intersect_naive(&client_set, &server_set)),
            expected
        );
        assert_eq!(
            HashSet::<u32>::from_iter(intersect_polynomial(&client_set, &server_set)),
            expected
        );
    }

    #[test]
    fn test_intersect_large() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1984);
        let NUM_CLIENT = 20;
        let NUM_SERVER = 20;
        let NUM_BOTH = 0;

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
            HashSet::<u32>::from_iter(intersect_polynomial(&client_set, &server_set)),
            expected
        );
    }
}