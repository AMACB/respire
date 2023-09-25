use crate::math::int_mod::IntMod;
use crate::math::utils::{floor_log, mod_inverse, reverse_bits};

/// Compile time lookup table for NTT-related operations
struct NTTTable<const D: usize, const N: u64, const W: u64> {}

impl<const D: usize, const N: u64, const W: u64> NTTTable<D, N, W> {
    const W_POWERS_BIT_REVERSED: [IntMod<N>; D] = get_powers_bit_reversed::<D, N, W>(false);
    const W_INV_POWERS_BIT_REVERSED: [IntMod<N>; D] = get_powers_bit_reversed::<D, N, W>(true);
    const LOG_D: usize = floor_log(2, D as u64);
    const INV_D: IntMod<N> = IntMod::from_u64_const(mod_inverse(D as u64, N));
}

const fn get_powers_bit_reversed<const D: usize, const N: u64, const W: u64>(
    invert: bool,
) -> [IntMod<N>; D] {
    let root = if invert {
        IntMod::from_u64_const(mod_inverse(W, N))
    } else {
        IntMod::from_u64_const(W)
    };

    let mut table = [IntMod::from_u64_const(0_u64); D];
    let mut cur = IntMod::from_u64_const(1_u64);
    let mut idx = 0;

    while idx < D {
        table[reverse_bits::<D>(idx)] = cur;
        cur = IntMod::mul_const(cur, root);
        idx += 1
    }
    table
}

pub fn ntt_neg_forward<const D: usize, const N: u64, const W: u64>(values: &mut [IntMod<N>; D]) {
    // Algorithm 2 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N, W>::LOG_D {
        let block_count = 1_usize << round;
        let block_half_stride = D >> (1_usize + round);
        let block_stride = 2 * block_half_stride;
        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w: IntMod<N> = unsafe {
                *NTTTable::<D, N, W>::W_POWERS_BIT_REVERSED.get_unchecked(block_count + block_idx)
            };
            for left_idx in block_left_half_range {
                let right_idx = left_idx + block_half_stride;
                unsafe {
                    // Butterfly
                    let x = *values.get_unchecked(left_idx);
                    let y = w * *values.get_unchecked(right_idx);
                    *values.get_unchecked_mut(left_idx) = x + y;
                    *values.get_unchecked_mut(right_idx) = x - y;
                }
            }
        }
    }
}

pub fn ntt_neg_backward<const D: usize, const N: u64, const W: u64>(values: &mut [IntMod<N>; D]) {
    // Algorithm 3 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N, W>::LOG_D {
        let block_count = D >> (1_usize + round);
        let block_half_stride = 1 << round;
        let block_stride = 2 * block_half_stride;

        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w: IntMod<N> = unsafe {
                *NTTTable::<D, N, W>::W_INV_POWERS_BIT_REVERSED
                    .get_unchecked(block_count + block_idx)
            };

            for left_idx in block_left_half_range {
                let right_idx = left_idx + block_half_stride;
                unsafe {
                    // Butterfly
                    let x = *values.get_unchecked(left_idx);
                    let y = *values.get_unchecked(right_idx);
                    *values.get_unchecked_mut(left_idx) = x + y;
                    *values.get_unchecked_mut(right_idx) = (x - y) * w;
                }
            }
        }
    }

    for value in values.iter_mut() {
        *value *= NTTTable::<D, N, W>::INV_D;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod_cyclo::IntModCyclo;
    use crate::math::int_mod_cyclo_eval::IntModCycloEval;
    use crate::math::int_mod_poly::IntModPoly;
    use crate::math::number_theory::find_sqrt_primitive_root;
    use crate::math::rand_sampled::RandUniformSampled;
    use crate::math::ring_elem::RingElement;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::iter;
    use std::time::Instant;

    const D: usize = 4;
    const P: u64 = 268369921_u64;
    const W: u64 = find_sqrt_primitive_root(D, P);

    #[test]
    fn test_ntt_neg_forward() {
        let mut coeff: [IntMod<P>; 4] = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let coeff_poly = IntModPoly::from(vec![1_u64, 2_u64, 3_u64, 4_u64]);

        let w = IntMod::from(W);

        ntt_neg_forward::<D, P, W>(&mut coeff);
        let expected = [
            coeff_poly.eval(w),
            coeff_poly.eval(w.pow(3)),
            coeff_poly.eval(w.pow(5)),
            coeff_poly.eval(w.pow(7)),
        ];

        assert_eq!(coeff, expected);
    }

    #[test]
    fn test_ntt_neg_inverses() {
        let mut coeff: [IntMod<P>; 4] = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let expected = coeff;

        ntt_neg_forward::<D, P, W>(&mut coeff);
        ntt_neg_backward::<D, P, W>(&mut coeff);

        assert_eq!(coeff, expected);
    }

    #[test]
    fn test_ntt_neg_mul() {
        let mut coeff1: [IntMod<P>; 4] = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let mut coeff2: [IntMod<P>; 4] = [5_u64.into(), 6_u64.into(), 7_u64.into(), 8_u64.into()];

        ntt_neg_forward::<D, P, W>(&mut coeff1);
        ntt_neg_forward::<D, P, W>(&mut coeff2);
        let mut result = [
            coeff1[0] * coeff2[0],
            coeff1[1] * coeff2[1],
            coeff1[2] * coeff2[2],
            coeff1[3] * coeff2[3],
        ];
        ntt_neg_backward::<D, P, W>(&mut result);

        let coeff1_poly = IntModPoly::from(vec![1_u64, 2_u64, 3_u64, 4_u64]);
        let coeff2_poly = IntModPoly::from(vec![5_u64, 6_u64, 7_u64, 8_u64]);
        let result_poly = &coeff1_poly * &coeff2_poly;

        let (result_first, result_second) = result_poly.coeff.as_slice().split_at(4);
        // x^4 = -1
        let expected: Vec<IntMod<P>> = result_first
            .iter()
            .zip(result_second.iter().chain(iter::repeat(&IntMod::zero())))
            .map(|(x, y)| x - y)
            .collect();

        assert_eq!(expected, result);
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
