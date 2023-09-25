use crate::math::int_mod::IntMod;
use crate::math::utils::{floor_log, mod_inverse};

/// Compile time lookup table for NTT-related operations
struct NTTTable<const D: usize, const N: u64, const W: u64> {}

impl<const D: usize, const N: u64, const W: u64> NTTTable<D, N, W> {
    // Note: The ROOT_POWERS and INV_ROOT_POWERS tables are twice as long as they need to be. But
    // it doesn't really hurt to have them (apart from possibly longer compile times).
    const ROOT_POWERS: [IntMod<N>; D] = get_root_powers::<D, N, W>(false, true);
    const INV_ROOT_POWERS: [IntMod<N>; D] = get_root_powers::<D, N, W>(true, true);
    const SQRT_ROOT_POWERS: [IntMod<N>; D] = get_root_powers::<D, N, W>(false, false);
    const INV_SQRT_ROOT_POWERS: [IntMod<N>; D] = get_root_powers::<D, N, W>(true, false);
    const LOG_D: usize = floor_log(2, D as u64);
    const INV_D: IntMod<N> = IntMod::from_u64_const(mod_inverse(D as u64, N));
}

const fn get_root_powers<const D: usize, const N: u64, const W: u64>(
    invert: bool,
    square: bool,
) -> [IntMod<N>; D] {
    let root = if invert {
        IntMod::from_u64_const(mod_inverse(W, N))
    } else {
        IntMod::from_u64_const(W)
    };

    let root = if square {
        IntMod::mul_const(root, root)
    } else {
        root
    };

    let mut table = [IntMod::from_u64_const(0_u64); D];
    let mut cur = IntMod::from_u64_const(1_u64);
    let mut idx = 0;

    while idx < D {
        table[idx] = cur;
        cur = IntMod::mul_const(cur, root);
        idx += 1
    }
    table
}

pub fn ntt_neg_forward<const D: usize, const N: u64, const W: u64>(values: &mut [IntMod<N>; D]) {
    // Preprocess
    for (value, sqrt_root_power) in values.iter_mut().zip(NTTTable::<D, N, W>::SQRT_ROOT_POWERS) {
        *value *= sqrt_root_power;
    }

    // NTT
    ntt_common::<D, N, W>(values, false);
}

pub fn ntt_neg_backward<const D: usize, const N: u64, const W: u64>(values: &mut [IntMod<N>; D]) {
    // NTT
    ntt_common::<D, N, W>(values, true);

    // Postprocess
    for (value, inv_sqrt_root_power) in values
        .iter_mut()
        .zip(NTTTable::<D, N, W>::INV_SQRT_ROOT_POWERS)
    {
        *value *= NTTTable::<D, N, W>::INV_D;
        *value *= inv_sqrt_root_power;
    }
}

pub fn ntt_forward<const D: usize, const N: u64, const W: u64>(values: &mut [IntMod<N>; D]) {
    ntt_common::<D, N, W>(values, false);
}

pub fn ntt_backward<const D: usize, const N: u64, const W: u64>(values: &mut [IntMod<N>; D]) {
    ntt_common::<D, N, W>(values, true);
}

fn ntt_common<const D: usize, const N: u64, const W: u64>(
    values: &mut [IntMod<N>; D],
    invert: bool,
) {
    assert_ne!(D, 0, "expected D to be non-zero");
    assert_eq!(D & (D - 1), 0, "expected D to be power of two; got {}", D);

    bit_reverse_order(values, NTTTable::<D, N, W>::LOG_D);

    // Cooley Tukey

    // For the SAFETY below, we have the loop invariants:
    // (i) prev_block_size * s == D / 2
    // (ii) 1 <= prev_block_size <= D / 2
    // (iii) 1 <= s <= D / 2
    // (iv) prev_block_size, s are powers of 2
    for round in 0..NTTTable::<D, N, W>::LOG_D {
        let prev_block_size = 1 << round;
        let s = D >> (1 + round);
        for block_start in (0..D).step_by(prev_block_size * 2) {
            for i in 0..prev_block_size {
                // SAFETY:
                // `i < prev_block_size` ==> by (i), `s * i < D / 2 <= D`.
                let w: IntMod<N> = if invert {
                    unsafe { *NTTTable::<D, N, W>::INV_ROOT_POWERS.get_unchecked(s * i) }
                } else {
                    unsafe { *NTTTable::<D, N, W>::ROOT_POWERS.get_unchecked(s * i) }
                };

                // SAFETY:
                // Since `prev_block_size * 2` is a power of two (iv) and <= D (ii) which is also a
                // power of two (iv), we know `prev_block_size * 2` divides `D`. Therefore, we have
                // `0 <= block_start <= block_start + 2 * prev_block_size < D`.
                unsafe {
                    let x = *values.get_unchecked(block_start + i);
                    let y = w * *values.get_unchecked(block_start + i + prev_block_size);
                    *values.get_unchecked_mut(block_start + i) = x + y;
                    *values.get_unchecked_mut(block_start + i + prev_block_size) = x - y;
                }
            }
        }
    }
}

fn bit_reverse_order<const D: usize, const N: u64>(values: &mut [IntMod<N>; D], log_d: usize) {
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
    use crate::math::ring_elem::RingElement;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::time::Instant;

    const D: usize = 4;
    const P: u64 = 268369921u64;
    const W: u64 = find_sqrt_primitive_root(D, P);

    // TODO: add more tests.
    #[test]
    fn ntt_self_inverse() {
        let mut coeff: [IntMod<P>; 4] = [1u64.into(), 2u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let coeff_orig = coeff;

        ntt_forward::<D, P, W>(&mut coeff);
        ntt_backward::<D, P, W>(&mut coeff);

        for c in coeff.iter_mut() {
            *c *= IntMod::<P>::from(D as u64).inverse();
        }

        assert_eq!(coeff, coeff_orig);
    }

    #[test]
    fn forward_ntt() {
        let mut coeff: [IntMod<P>; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let sqrt_root = IntMod::<P>::from(W);
        let root = sqrt_root * sqrt_root;

        ntt_forward::<D, P, W>(&mut coeff);

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
        let sqrt_root = IntMod::<P>::from(W);
        let root = sqrt_root * sqrt_root;
        let one = IntMod::one();

        let mut evaluated = [
            one + one,
            root + one,
            root * root + one,
            root * root * root + one,
        ];

        ntt_backward::<D, P, W>(&mut evaluated);

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

        let sqrt_root = IntMod::<P>::from(W);

        for i in 0..coeff1.len() {
            coeff1[i] *= sqrt_root.pow(i as u64);
            coeff2[i] *= sqrt_root.pow(i as u64);
        }

        ntt_forward::<D, P, W>(&mut coeff1);
        ntt_forward::<D, P, W>(&mut coeff2);

        let mut coeff3 = [0u64.into(); 4];
        for i in 0..coeff1.len() {
            coeff3[i] = coeff1[i] * coeff2[i];
        }

        ntt_backward::<D, P, W>(&mut coeff3);

        for (i, c) in coeff3.iter_mut().enumerate() {
            *c *= IntMod::<P>::from(D as u64).inverse();
            *c *= sqrt_root.pow(i as u64).inverse();
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
