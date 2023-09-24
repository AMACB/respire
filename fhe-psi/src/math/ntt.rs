use crate::math::int_mod::{IntMod, NoReduce};
use crate::math::ring_elem::*;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::RwLock;

/// Memoization table for discrete gaussian sampling.
/// Each key is (root, degree, modulus), which is likely overdescriptive.

// TODO: currently, forward and reverse tables are computed separately. This isn't necessary since the reverse table is just the forward table, reversed.
type RootTables = HashMap<(u64, u64, usize), Vec<u64>>;
static ROOT_TABLES: Lazy<RwLock<RootTables>> = Lazy::new(|| RwLock::new(HashMap::new()));

pub enum NegacyclicType<const N: u64> {
    None, Forward(IntMod<N>), Reverse(IntMod<N>),
}

pub fn ntt<const D: usize, const N: u64>(
    values: &mut [IntMod<N>; D],
    root: IntMod<N>,
    log_d: usize,
    negacyclic: NegacyclicType<N>,
) {
    // compute root table, if necessary
    // TODO: see above. This unnecessarily recomputes the reverse table.
    let key = (u64::from(root), N, D);
    if ROOT_TABLES.read().unwrap().get(&key).is_none() {
        let mut table = vec![0; D];
        let mut cur = IntMod::one();
        for entry in table.iter_mut() {
            *entry = u64::from(cur);
            cur *= root;
        }
        ROOT_TABLES.write().unwrap().insert(key, table);
    }

    // get table
    let table_map = ROOT_TABLES.read().unwrap();
    let table = table_map.get(&key).unwrap();

    if let NegacyclicType::Forward(neg_root) = negacyclic {
        // Pre-process, if we are doing a forward NTT
        let mut neg_root_power: IntMod<N> = 1u64.into();
        // negacyclic preprocessing
        for p in values.iter_mut() {
            *p *= neg_root_power;
            neg_root_power *= neg_root;
        }
    }

    bit_reverse_order(values, log_d);

    // Cooley Tukey
    for round in 0..log_d {
        let prev_block_size = 1 << round;
        let s = D >> (round + 1);

        for block_start in (0..D).step_by(prev_block_size * 2) {
            for i in 0..prev_block_size {
                let w: IntMod<N> = IntMod::from(NoReduce(table[s * i]));
                let x = values[block_start + i];
                let y = w * (values[block_start + i + prev_block_size]);
                values[block_start + i] = x + y;
                values[block_start + i + prev_block_size] = x - y;
            }
        }
    }

    if let NegacyclicType::Reverse(neg_root) = negacyclic {
        // Post-process, if we are doing a reverse NTT
        let mut neg_root_power: IntMod<N> = 1u64.into();
        let inv_neg_root = neg_root.inverse();
        let inv_d = IntMod::<N>::from(D as u64).inverse();
        for c in values.iter_mut() {
            // divide by degree
            *c *= inv_d;
            // negacyclic post-processing
            *c *= neg_root_power;
            neg_root_power *= inv_neg_root;
        }
    }

}

pub fn bit_reverse_order<const D: usize, const N: u64>(values: &mut [IntMod<N>; D], log_d: usize) {
    for i in 0..D {
        let mut ri = i.reverse_bits();
        ri >>= (usize::BITS as usize) - log_d;
        if i < ri {
            values.swap(ri, i);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod_cyclo::IntModCyclo;
    use crate::math::int_mod_cyclo_eval::IntModCycloEval;
    use crate::math::number_theory::find_sqrt_primitive_root;
    use crate::math::rand_sampled::RandUniformSampled;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::time::Instant;

    const D: usize = 4;
    const LOG_D: usize = 2;
    const P: u64 = 268369921u64;
    const W: u64 = 180556700u64; // order = 1 << 16

    // TODO: add more tests.
    #[test]
    fn ntt_self_inverse() {
        let mut coeff: [IntMod<P>; 4] = [1u64.into(), 2u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let coeff_orig = coeff;
        let root = IntMod::<P>::from(W).pow(1 << 14);

        ntt(&mut coeff, root, LOG_D, NegacyclicType::None);
        ntt(&mut coeff, root.inverse(), LOG_D, NegacyclicType::None);

        for c in coeff.iter_mut() {
            *c *= IntMod::<P>::from(D as u64).inverse();
        }

        assert_eq!(coeff, coeff_orig);
    }

    #[test]
    fn forward_ntt() {
        let mut coeff: [IntMod<P>; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let root = IntMod::<P>::from(W).pow(1 << 14);

        ntt(&mut coeff, root, LOG_D, NegacyclicType::None);

        let one = IntMod::one();
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
        let root = IntMod::<P>::from(W).pow(1 << 14);
        let one = IntMod::one();

        let mut evaluated = [
            one + one,
            root + one,
            root * root + one,
            root * root * root + one,
        ];

        ntt(&mut evaluated, root.inverse(), LOG_D, NegacyclicType::None);

        for c in evaluated.iter_mut() {
            *c *= IntMod::<P>::from(D as u64).inverse();
        }

        let coeff: [IntMod<P>; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        assert_eq!(coeff, evaluated);
    }

    #[test]
    fn test_fancy_ntt() {
        let mut coeff1: [IntMod<P>; D] = [1u64.into(), 2u64.into(), 3u64.into(), 4u64.into()];
        let mut coeff2: [IntMod<P>; D] = [1u64.into(), 1u64.into(), 1u64.into(), 1u64.into()];
        let ans: [IntMod<P>; D] = [(P - 8).into(), (P - 4).into(), 2u64.into(), 10u64.into()];

        let root = IntMod::<P>::from(W).pow(1 << 13);

        for i in 0..coeff1.len() {
            coeff1[i] *= root.pow(i as u64);
            coeff2[i] *= root.pow(i as u64);
        }

        ntt(&mut coeff1, root * root, LOG_D, NegacyclicType::None);
        ntt(&mut coeff2, root * root, LOG_D, NegacyclicType::None);

        let mut coeff3 = [0u64.into(); 4];
        for i in 0..coeff1.len() {
            coeff3[i] = coeff1[i] * coeff2[i];
        }

        ntt(&mut coeff3, (root * root).inverse(), LOG_D, NegacyclicType::None);

        for (i, c) in coeff3.iter_mut().enumerate() {
            *c *= IntMod::<P>::from(D as u64).inverse();
            *c *= root.pow(i as u64).inverse();
        }
        assert_eq!(coeff3, ans);
    }

    #[ignore]
    #[test]
    fn test_ntt_stress() {
        const D: usize = 2048;
        const P: u64 = 268369921;
        type RCoeff = IntModCyclo<D, P>;
        type REval = IntModCycloEval<D, P, { find_sqrt_primitive_root(D, P) }>;

        const NUM_ITER: usize = 1 << 16;

        let mut rng = ChaCha20Rng::from_entropy();
        let mut elems = Vec::with_capacity(NUM_ITER);
        for _ in 0..NUM_ITER {
            elems.push(RCoeff::rand_uniform(&mut rng));
        }
        let elems_clone = elems.clone();

        let start = Instant::now();
        for (x_expected, x_test) in elems.into_iter().zip(elems_clone.into_iter()) {
            let x_test_eval = REval::from(x_test);
            let x_test = RCoeff::from(x_test_eval);
            assert_eq!(x_expected, x_test);
        }
        let end = Instant::now();
        eprintln!(
            "took {:?} to do {} iterations ({:?} / iter)",
            end - start,
            NUM_ITER,
            (end - start) / NUM_ITER as u32
        );
    }
}
