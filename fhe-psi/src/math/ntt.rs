use crate::math::int_mod::IntMod;
use crate::math::utils::{floor_log, mod_inverse, reverse_bits};
use std::num::Wrapping;
use std::ptr;

/// Compile time lookup table for NTT-related operations
struct NTTTable<const D: usize, const N: u64, const W: u64> {}

#[derive(Copy, Clone)]
struct MulTable<const N: u64> {
    value: IntMod<N>,
    ratio32: u64,
}

impl<const N: u64> MulTable<N> {
    const fn zero() -> Self {
        MulTable {
            value: IntMod::from_u64_const(0_u64),
            ratio32: 0,
        }
    }
}

impl<const D: usize, const N: u64, const W: u64> NTTTable<D, N, W> {
    const W_POWERS_BIT_REVERSED: [MulTable<N>; D] = get_powers_bit_reversed::<D, N, W>(false);
    const W_INV_POWERS_BIT_REVERSED: [MulTable<N>; D] = get_powers_bit_reversed::<D, N, W>(true);
    const LOG_D: usize = floor_log(2, D as u64);
    const INV_D: IntMod<N> = IntMod::from_u64_const(mod_inverse(D as u64, N));
}

const fn get_powers_bit_reversed<const D: usize, const N: u64, const W: u64>(
    invert: bool,
) -> [MulTable<N>; D] {
    let root = if invert {
        IntMod::from_u64_const(mod_inverse(W, N))
    } else {
        IntMod::from_u64_const(W)
    };

    let mut table = [MulTable::zero(); D];
    let mut cur = IntMod::from_u64_const(1_u64);
    let mut idx = 0;

    while idx < D {
        let ratio32 = (cur.into_u64_const() << 32) / N;
        table[reverse_bits::<D>(idx)] = MulTable {
            value: cur,
            ratio32,
        };

        cur = IntMod::mul_const(cur, root);
        idx += 1
    }
    table
}

pub fn ntt_neg_forward<const D: usize, const N: u64, const W: u64>(
    values: [IntMod<N>; D],
) -> [IntMod<N>; D] {
    fn butterfly32<const N: u64>(x: u64, y: u64, w: u64, ratio: u64) -> (u64, u64) {
        let x = if x >= 2 * N { x - 2 * N } else { x };
        let quotient = (ratio * y) >> 32;
        let product = (Wrapping(w as u32) * Wrapping(y as u32)
            - Wrapping(N as u32) * Wrapping(quotient as u32))
        .0;
        (x + product as u64, x + (2 * N - product as u64))
    }

    let mut values: [u64; D] =
        unsafe { ptr::read(&values as *const [IntMod<N>; D] as *const [u64; D]) };

    // Algorithm 2 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N, W>::LOG_D {
        let block_count = 1_usize << round;
        let block_half_stride = D >> (1_usize + round);
        let block_stride = 2 * block_half_stride;
        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w_table = unsafe {
                *NTTTable::<D, N, W>::W_POWERS_BIT_REVERSED.get_unchecked(block_count + block_idx)
            };

            if block_left_half_range.len() < 4 {
                for left_idx in block_left_half_range {
                    let right_idx = left_idx + block_half_stride;
                    unsafe {
                        // Butterfly
                        let (x_new, y_new) = butterfly32::<N>(
                            *values.get_unchecked(left_idx),
                            *values.get_unchecked(right_idx),
                            w_table.value.into_u64_const(),
                            w_table.ratio32,
                        );
                        *values.get_unchecked_mut(left_idx) = x_new;
                        *values.get_unchecked_mut(right_idx) = y_new;
                    }
                }
            } else {
                unsafe {
                    use std::arch::x86_64::*;
                    let w = _mm256_set1_epi64x(w_table.value.into_u64_const() as i64);
                    let ratio = _mm256_set1_epi64x(w_table.ratio32 as i64);
                    let double_modulus = _mm256_set1_epi64x(2 * N as i64);
                    let neg_modulus = _mm256_set1_epi64x(-(N as i64));
                    for left_idx in block_left_half_range.step_by(4) {
                        let right_idx = left_idx + block_half_stride;
                        let left_ptr =
                            values.get_unchecked(left_idx) as *const u64 as *const __m256i;
                        let right_ptr =
                            values.get_unchecked(right_idx) as *const u64 as *const __m256i;

                        // Butterfly
                        let x = _mm256_load_si256(left_ptr);
                        let y = _mm256_load_si256(right_ptr);
                        // This works because the upper 32 bits of each 64 bit are zero
                        let x = _mm256_min_epu32(x, _mm256_sub_epi32(x, double_modulus));
                        let quotient = _mm256_srli_epi64::<32>(_mm256_mul_epu32(ratio, y));
                        let w_times_y = _mm256_mullo_epi32(w, y);
                        let product =
                            _mm256_add_epi64(w_times_y, _mm256_mul_epu32(neg_modulus, quotient));
                        let x_new_vec = _mm256_add_epi64(x, product);
                        let y_new_vec =
                            _mm256_add_epi64(x, _mm256_sub_epi64(double_modulus, product));

                        _mm256_store_si256(left_ptr as *mut __m256i, x_new_vec);
                        _mm256_store_si256(right_ptr as *mut __m256i, y_new_vec);
                    }
                }
            }
        }
    }

    for i in 0..D {
        if values[i] >= 2 * N {
            values[i] -= 2 * N;
        }
        if values[i] >= N {
            values[i] -= N;
        }
    }

    let values: [IntMod<N>; D] =
        unsafe { ptr::read(&values as *const [u64; D] as *const [IntMod<N>; D]) };

    values
}

pub fn ntt_neg_backward<const D: usize, const N: u64, const W: u64>(
    mut values: [IntMod<N>; D],
) -> [IntMod<N>; D] {
    // Algorithm 3 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N, W>::LOG_D {
        let block_count = D >> (1_usize + round);
        let block_half_stride = 1 << round;
        let block_stride = 2 * block_half_stride;

        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w: IntMod<N> = unsafe {
                NTTTable::<D, N, W>::W_INV_POWERS_BIT_REVERSED
                    .get_unchecked(block_count + block_idx)
                    .value
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

    values
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
        let coeff: [IntMod<P>; 4] = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let coeff_poly = IntModPoly::from(vec![1_u64, 2_u64, 3_u64, 4_u64]);

        let w = IntMod::from(W);

        let points = ntt_neg_forward::<D, P, W>(coeff);
        let expected = [
            coeff_poly.eval(w),
            coeff_poly.eval(w.pow(5)), // swapped, since order is bit reversed
            coeff_poly.eval(w.pow(3)),
            coeff_poly.eval(w.pow(7)),
        ];

        assert_eq!(points, expected);
    }

    #[test]
    fn test_ntt_neg_inverses() {
        let coeff: [IntMod<P>; 4] = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let expected = coeff;

        let points = ntt_neg_forward::<D, P, W>(coeff);
        let coeff = ntt_neg_backward::<D, P, W>(points);

        assert_eq!(coeff, expected);
    }

    #[test]
    fn test_ntt_neg_mul() {
        let coeff1: [IntMod<P>; 4] = [1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()];
        let coeff2: [IntMod<P>; 4] = [5_u64.into(), 6_u64.into(), 7_u64.into(), 8_u64.into()];

        let points1 = ntt_neg_forward::<D, P, W>(coeff1);
        let points2 = ntt_neg_forward::<D, P, W>(coeff2);
        let result_points = [
            points1[0] * points2[0],
            points1[1] * points2[1],
            points1[2] * points2[2],
            points1[3] * points2[3],
        ];
        let result_coeff = ntt_neg_backward::<D, P, W>(result_points);

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

        assert_eq!(expected, result_coeff);
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
