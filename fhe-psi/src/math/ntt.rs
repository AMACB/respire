use crate::math::ring_elem::*;
use crate::math::z_n::Z_N;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

// TODO: this should go somewhere else probably
pub fn pow<const N: u64>(mut val: Z_N<N>, mut e: u64) -> Z_N<N> {
    let mut res = Z_N::one();
    while e > 0 {
        if (e & 1) == 1 {
            res *= val;
        }
        e >>= 1;
        val *= val;
    }
    return res;
}

// TODO: this should also go somewhere else, aslo this is not efficient
pub fn inverse<const N: u64>(val: Z_N<N>) -> Z_N<N> {
    return pow(val, N - 2);
}

/// Memoization table for discrete gaussian sampling.
/// Each key is (root, degree, modulus), which is likely overdescriptive.

// TODO: currently, forward and reverse tables are computed separately. This isn't necessary since the reverse table is just the forward table, reversed.
static ROOT_TABLES: Lazy<RwLock<HashMap<(u64, u64, usize), Vec<u64>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn ntt<const D: usize, const N: u64>(values: &mut [Z_N<N>; D], root: Z_N<N>, log_d: usize) {
    // compute root table, if necessary
    // TODO: see above. This unnecessarily recomputes the reverse table.
    let key = (root.into(), N, D);
    if ROOT_TABLES.read().unwrap().get(&key) == None {
        let mut table = vec![0; D];
        let mut cur = Z_N::one();
        for i in 0..D {
            table[i] = cur.into();
            cur *= root;
        }
        ROOT_TABLES.write().unwrap().insert(key.clone(), table);
    }

    // get table
    let table_map = ROOT_TABLES.read().unwrap();
    let table = table_map.get(&key).unwrap();

    // Cooley Tukey
    for round in 0..log_d {
        let prev_block_size = 1 << round;
        let s = (D as usize) >> (round + 1);

        for block_start in (0..D).step_by(prev_block_size * 2) {
            for i in 0..prev_block_size {
                let w: Z_N<N> = table[s * i].into();
                let x = values[block_start + i];
                let y = w * (values[block_start + i + prev_block_size]);
                values[block_start + i] = x + y;
                values[block_start + i + prev_block_size] = x - y;
            }
        }
    }
}

pub fn bit_reverse_order<const D: usize, const N: u64>(values: &mut [Z_N<N>; D], log_d: usize) {
    for i in 0..D {
        let mut ri = i.reverse_bits();
        ri >>= (usize::BITS as usize) - log_d;
        if i < ri {
            let tmp = values[ri];
            values[ri] = values[i];
            values[i] = tmp;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const D: usize = 4;
    const LOG_D: usize = 2;
    const P: u64 = 268369921u64;
    const W: u64 = 180556700u64; // order = 1 << 16

    // TODO: add more tests.
    #[test]
    fn ntt_self_inverse() {
        let mut coeff: [Z_N<P>; 4] = [1u64.into(), 2u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let coeff_orig = coeff.clone();
        let root = pow(W.into(), 1 << 14);

        bit_reverse_order(&mut coeff, LOG_D);
        ntt(&mut coeff, root, LOG_D);
        bit_reverse_order(&mut coeff, LOG_D);
        ntt(&mut coeff, inverse(root), LOG_D);

        for i in 0..coeff.len() {
            coeff[i] *= inverse((D as u64).into());
        }

        assert_eq!(coeff, coeff_orig);
    }

    #[test]
    fn forward_ntt() {
        let mut coeff: [Z_N<P>; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let root = pow(W.into(), 1 << 14);

        bit_reverse_order(&mut coeff, LOG_D);
        ntt(&mut coeff, root, LOG_D);

        let one = Z_N::one();
        let evaluated = [
            one + one,
            root + one,
            root * root + one,
            root * root * root + one,
        ];

        assert_eq!(coeff, evaluated);
    }

    #[test]
    fn backward_ntt() {
        let root = pow(W.into(), 1 << 14);
        let one = Z_N::one();

        let mut evaluated = [
            one + one,
            root + one,
            root * root + one,
            root * root * root + one,
        ];

        bit_reverse_order(&mut evaluated, LOG_D);
        ntt(&mut evaluated, inverse(root), LOG_D);

        for i in 0..evaluated.len() {
            evaluated[i] *= inverse((D as u64).into());
        }

        let coeff: [Z_N<P>; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        assert_eq!(coeff, evaluated);
    }

    #[test]
    fn test_fancy_ntt() {
        let mut coeff1: [Z_N<P>; D] = [1u64.into(), 2u64.into(), 3u64.into(), 4u64.into()];
        let mut coeff2: [Z_N<P>; D] = [1u64.into(), 1u64.into(), 1u64.into(), 1u64.into()];
        let ans: [Z_N<P>; D] = [(P - 8).into(), (P - 4).into(), 2u64.into(), 10u64.into()];

        let root = pow(W.into(), 1 << 13);

        for i in 0..coeff1.len() {
            coeff1[i] *= pow(root, i as u64);
            coeff2[i] *= pow(root, i as u64);
        }

        bit_reverse_order(&mut coeff1, LOG_D);
        ntt(&mut coeff1, root * root, LOG_D);
        bit_reverse_order(&mut coeff2, LOG_D);
        ntt(&mut coeff2, root * root, LOG_D);

        let mut coeff3 = [0u64.into(); 4];
        for i in 0..coeff1.len() {
            coeff3[i] = coeff1[i] * coeff2[i];
        }

        bit_reverse_order(&mut coeff3, LOG_D);
        ntt(&mut coeff3, inverse(root * root), LOG_D);

        for i in 0..coeff3.len() {
            coeff3[i] *= inverse((D as u64).into());
            coeff3[i] *= inverse(pow(root, i as u64));
        }
        assert_eq!(coeff3, ans);
    }
}
