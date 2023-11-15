use crate::math::int_mod::IntMod;
use crate::math::int_mod_cyclo_eval::IntModCycloEval;
use crate::math::simd_utils::Aligned32;
use crate::math::utils::{floor_log, get_ratio32, mod_inverse, reverse_bits_fast};

/// Compile time lookup table for NTT-related operations
struct NTTTable<const D: usize, const N: u64> {}

#[derive(Copy, Clone)]
struct MulTable<const N: u64> {
    value: IntMod<N>,
    #[allow(unused)]
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

impl<const D: usize, const N: u64> NTTTable<D, N> {
    const W_POWERS_BIT_REVERSED: [MulTable<N>; D] = get_powers_bit_reversed::<D, N>(false);
    const W_INV_POWERS_BIT_REVERSED: [MulTable<N>; D] = get_powers_bit_reversed::<D, N>(true);
    const LOG_D: usize = floor_log(2, D as u64);
    const INV_D: IntMod<N> = IntMod::from_u64_const(mod_inverse(D as u64, N));
    #[allow(unused)]
    const INV_D_RATIO32: u64 = get_ratio32::<N>(mod_inverse(D as u64, N) % N);
}

const fn get_powers_bit_reversed<const D: usize, const N: u64>(invert: bool) -> [MulTable<N>; D] {
    let root = if invert {
        IntMod::from_u64_const(mod_inverse(IntModCycloEval::<D, N>::W, N))
    } else {
        IntMod::from_u64_const(IntModCycloEval::<D, N>::W)
    };

    let mut table = [MulTable::zero(); D];
    let mut cur = IntMod::from_u64_const(1_u64);
    let mut idx = 0;

    while idx < D {
        let ratio32 = get_ratio32::<N>(cur.into_u64_const());
        table[reverse_bits_fast::<D>(idx)] = MulTable {
            value: cur,
            ratio32,
        };

        cur = IntMod::mul_const(cur, root);
        idx += 1
    }
    table
}

fn ntt_neg_forward_fallback<const D: usize, const N: u64>(values: &mut Aligned32<[IntMod<N>; D]>) {
    // Algorithm 2 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N>::LOG_D {
        let block_count = 1_usize << round;
        let block_half_stride = D >> (1_usize + round);
        let block_stride = 2 * block_half_stride;
        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w: IntMod<N> = unsafe {
                NTTTable::<D, N>::W_POWERS_BIT_REVERSED
                    .get_unchecked(block_count + block_idx)
                    .value
            };
            for left_idx in block_left_half_range {
                let right_idx = left_idx + block_half_stride;
                unsafe {
                    // Butterfly
                    let x = *values.0.get_unchecked(left_idx);
                    let y = w * *values.0.get_unchecked(right_idx);
                    *values.0.get_unchecked_mut(left_idx) = x + y;
                    *values.0.get_unchecked_mut(right_idx) = x - y;
                }
            }
        }
    }
}

#[cfg(not(target_feature = "avx2"))]
pub fn ntt_neg_forward<const D: usize, const N: u64>(values: &mut Aligned32<[IntMod<N>; D]>) {
    ntt_neg_forward_fallback::<D, N>(values)
}

#[cfg(target_feature = "avx2")]
pub fn ntt_neg_forward<const D: usize, const N: u64>(values: &mut Aligned32<[IntMod<N>; D]>) {
    use crate::math::simd_utils::*;
    use std::arch::x86_64::*;
    use std::num::Wrapping;

    if N >= (1 << 30) {
        return ntt_neg_forward_fallback::<D, N>(values);
    }

    let values = values as *mut Aligned32<[IntMod<N>; D]>;
    let values = values as *mut Aligned32<[u64; D]>;
    let values = unsafe { &mut *values };

    let modulus = unsafe { _mm256_set1_epi64x(N as i64) };
    let double_modulus = unsafe { _mm256_set1_epi64x(2 * N as i64) };

    // Algorithm 2/6 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N>::LOG_D {
        let block_count = 1_usize << round;
        let block_half_stride = D >> (1_usize + round);
        let block_stride = 2 * block_half_stride;
        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            unsafe {
                let w_table =
                    *NTTTable::<D, N>::W_POWERS_BIT_REVERSED.get_unchecked(block_count + block_idx);

                let w = _mm256_set1_epi64x(w_table.value.into_u64_const() as i64);
                let w_ratio32 = _mm256_set1_epi64x(w_table.ratio32 as i64);

                if block_left_half_range.len() < 4 {
                    for left_idx in block_left_half_range {
                        let right_idx = left_idx + block_half_stride;
                        let x = *values.0.get_unchecked(left_idx);
                        let y = *values.0.get_unchecked(right_idx);
                        let w = w_table.value.into_u64_const();
                        let ratio = w_table.ratio32;

                        // Butterfly
                        let x = if x >= 2 * N { x - 2 * N } else { x };
                        let quotient = (ratio * y) >> 32;
                        let product = (Wrapping(w as u32) * Wrapping(y as u32)
                            - Wrapping(N as u32) * Wrapping(quotient as u32))
                        .0;
                        let x_new = x + product as u64;
                        let y_new = x + (2 * N - product as u64);

                        *values.0.get_unchecked_mut(left_idx) = x_new;
                        *values.0.get_unchecked_mut(right_idx) = y_new;
                    }
                } else {
                    for left_idx in block_left_half_range.step_by(4) {
                        let right_idx = left_idx + block_half_stride;
                        let left_ptr =
                            values.0.get_unchecked(left_idx) as *const u64 as *const __m256i;
                        let right_ptr =
                            values.0.get_unchecked(right_idx) as *const u64 as *const __m256i;

                        // Butterfly
                        let x = _mm256_load_si256(left_ptr);
                        let y = _mm256_load_si256(right_ptr);
                        let x = _mm256_reduce_half(x, double_modulus);
                        let product = _mm256_mod_mul32(y, w, w_ratio32, modulus);
                        let y_new = _mm256_add_epi64(x, _mm256_sub_epi64(double_modulus, product));
                        let x_new = _mm256_add_epi64(x, product);

                        _mm256_store_si256(left_ptr as *mut __m256i, x_new);
                        _mm256_store_si256(right_ptr as *mut __m256i, y_new);
                    }
                }
            }
        }
    }

    unsafe {
        for i in (0..D).step_by(4) {
            let val = _mm256_load_si256(values.0.get_unchecked(i) as *const u64 as *const __m256i);
            let val = _mm256_reduce_half(val, double_modulus);
            let val = _mm256_reduce_half(val, modulus);
            _mm256_store_si256(
                values.0.get_unchecked_mut(i) as *mut u64 as *mut __m256i,
                val,
            );
        }
    }
}

fn ntt_neg_backward_fallback<const D: usize, const N: u64>(values: &mut Aligned32<[IntMod<N>; D]>) {
    // Algorithm 3 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N>::LOG_D {
        let block_count = D >> (1_usize + round);
        let block_half_stride = 1 << round;
        let block_stride = 2 * block_half_stride;

        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w: IntMod<N> = unsafe {
                NTTTable::<D, N>::W_INV_POWERS_BIT_REVERSED
                    .get_unchecked(block_count + block_idx)
                    .value
            };

            for left_idx in block_left_half_range {
                let right_idx = left_idx + block_half_stride;
                unsafe {
                    // Butterfly
                    let x = *values.0.get_unchecked(left_idx);
                    let y = *values.0.get_unchecked(right_idx);
                    *values.0.get_unchecked_mut(left_idx) = x + y;
                    *values.0.get_unchecked_mut(right_idx) = (x - y) * w;
                }
            }
        }
    }

    for value in values.0.iter_mut() {
        *value *= NTTTable::<D, N>::INV_D;
    }
}

#[cfg(not(target_feature = "avx2"))]
pub fn ntt_neg_backward<const D: usize, const N: u64>(values: &mut Aligned32<[IntMod<N>; D]>) {
    ntt_neg_backward_fallback::<D, N>(values)
}

#[cfg(target_feature = "avx2")]
pub fn ntt_neg_backward<const D: usize, const N: u64>(values: &mut Aligned32<[IntMod<N>; D]>) {
    use crate::math::simd_utils::*;
    use std::arch::x86_64::*;
    use std::num::Wrapping;

    if N >= (1 << 30) {
        return ntt_neg_backward_fallback::<D, N>(values);
    }

    let values = values as *mut Aligned32<[IntMod<N>; D]>;
    let values = values as *mut Aligned32<[u64; D]>;
    let values = unsafe { &mut *values };

    let modulus = unsafe { _mm256_set1_epi64x(N as i64) };
    let double_modulus = unsafe { _mm256_set1_epi64x(2 * N as i64) };

    // Algorithm 3/7 of https://arxiv.org/pdf/2103.16400.pdf
    for round in 0..NTTTable::<D, N>::LOG_D {
        let block_count = D >> (1_usize + round);
        let block_half_stride = 1 << round;
        let block_stride = 2 * block_half_stride;

        for block_idx in 0..block_count {
            let block_left_half_range =
                (block_idx * block_stride)..(block_idx * block_stride + block_half_stride);

            let w_table = unsafe {
                NTTTable::<D, N>::W_INV_POWERS_BIT_REVERSED.get_unchecked(block_count + block_idx)
            };

            if block_left_half_range.len() < 4 {
                for left_idx in block_left_half_range {
                    let right_idx = left_idx + block_half_stride;
                    unsafe {
                        // Inverse Butterfly
                        let x = *values.0.get_unchecked(left_idx);
                        let y = *values.0.get_unchecked(right_idx);

                        let x_new = x + y;
                        let x_new = if x_new >= 2 * N { x_new - 2 * N } else { x_new };
                        let sum = x - y + 2 * N;
                        let quotient = (w_table.ratio32 * sum) >> 32;
                        let y_new = (Wrapping(u64::from(w_table.value) as u32)
                            * Wrapping(sum as u32)
                            - Wrapping(N as u32) * Wrapping(quotient as u32))
                        .0 as u64;

                        *values.0.get_unchecked_mut(left_idx) = x_new;
                        *values.0.get_unchecked_mut(right_idx) = y_new;
                    }
                }
            } else {
                unsafe {
                    let w = _mm256_set1_epi64x(w_table.value.into_u64_const() as i64);
                    let w_ratio32 = _mm256_set1_epi64x(w_table.ratio32 as i64);

                    for left_idx in block_left_half_range.step_by(4) {
                        let right_idx = left_idx + block_half_stride;
                        let left_ptr =
                            values.0.get_unchecked(left_idx) as *const u64 as *const __m256i;
                        let right_ptr =
                            values.0.get_unchecked(right_idx) as *const u64 as *const __m256i;

                        // Inverse Butterfly
                        let x = _mm256_load_si256(left_ptr);
                        let y = _mm256_load_si256(right_ptr);
                        let x_new = _mm256_add_epi64(x, y);
                        let x_new = _mm256_reduce_half(x_new, double_modulus);
                        let sum = _mm256_sub_epi64(_mm256_add_epi64(x, double_modulus), y);
                        let y_new = _mm256_mod_mul32(sum, w, w_ratio32, modulus);

                        _mm256_store_si256(left_ptr as *mut __m256i, x_new);
                        _mm256_store_si256(right_ptr as *mut __m256i, y_new);
                    }
                }
            }
        }
    }

    unsafe {
        let inv_d = _mm256_set1_epi64x(u64::from(NTTTable::<D, N>::INV_D) as i64);
        let inv_d_ratio32 = _mm256_set1_epi64x(NTTTable::<D, N>::INV_D_RATIO32 as i64);

        for i in (0..D).step_by(4) {
            let val = _mm256_load_si256(values.0.get_unchecked(i) as *const u64 as *const __m256i);
            let val = _mm256_mod_mul32(val, inv_d, inv_d_ratio32, modulus);
            let val = _mm256_reduce_half(val, modulus);
            _mm256_store_si256(
                values.0.get_unchecked_mut(i) as *mut u64 as *mut __m256i,
                val,
            );
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::int_mod_cyclo::IntModCyclo;
    use crate::math::int_mod_cyclo_eval::IntModCycloEval;
    use crate::math::int_mod_poly::IntModPoly;
    use crate::math::rand_sampled::RandUniformSampled;
    use crate::math::ring_elem::RingElement;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::iter;
    use std::time::Instant;

    const D: usize = 4;
    const P: u64 = 268369921_u64;

    #[test]
    fn test_ntt_neg_forward() {
        let mut values: Aligned32<[IntMod<P>; 4]> =
            Aligned32([1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()]);
        let coeff_poly = IntModPoly::from(vec![1_u64, 2_u64, 3_u64, 4_u64]);

        let w = IntMod::from(IntModCycloEval::<D, P>::W);

        ntt_neg_forward::<D, P>(&mut values);
        let expected = [
            coeff_poly.eval(w),
            coeff_poly.eval(w.pow(5)), // swapped, since order is bit reversed
            coeff_poly.eval(w.pow(3)),
            coeff_poly.eval(w.pow(7)),
        ];

        assert_eq!(values.0, expected);
    }

    #[test]
    fn test_ntt_neg_inverses() {
        let mut values: Aligned32<[IntMod<P>; 4]> =
            Aligned32([1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()]);
        let expected = values.0;

        ntt_neg_forward::<D, P>(&mut values);
        ntt_neg_backward::<D, P>(&mut values);

        assert_eq!(values.0, expected);
    }

    #[test]
    fn test_ntt_neg_mul() {
        let mut values1: Aligned32<[IntMod<P>; 4]> =
            Aligned32([1_u64.into(), 2_u64.into(), 3_u64.into(), 4_u64.into()]);
        let mut values2: Aligned32<[IntMod<P>; 4]> =
            Aligned32([5_u64.into(), 6_u64.into(), 7_u64.into(), 8_u64.into()]);

        ntt_neg_forward::<D, P>(&mut values1);
        ntt_neg_forward::<D, P>(&mut values2);
        let mut result_points = Aligned32([
            values1.0[0] * values2.0[0],
            values1.0[1] * values2.0[1],
            values1.0[2] * values2.0[2],
            values1.0[3] * values2.0[3],
        ]);
        ntt_neg_backward::<D, P>(&mut result_points);

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

        assert_eq!(expected, result_points.0);
    }

    fn test_ntt_forward_size<const DD: usize>() {
        let mut vec_mod = vec![];
        for i in 0..DD as u64 {
            vec_mod.push(IntMod::from(i + 1));
        }

        let mut values: Aligned32<[IntMod<P>; DD]> = Aligned32(vec_mod.clone().try_into().unwrap());
        let coeff_poly = IntModPoly::<P>::from(vec_mod);

        let w = IntMod::from(IntModCycloEval::<DD, P>::W);

        ntt_neg_forward::<DD, P>(&mut values);
        let mut expected = vec![];
        expected.resize(DD, IntMod::zero());
        for i in 0..DD {
            expected[reverse_bits_fast::<DD>(i)] = coeff_poly.eval(w.pow(2 * (i as u64) + 1));
        }

        assert_eq!(values.0, expected.as_slice());
    }

    #[test]
    fn test_ntt_forward_8() {
        test_ntt_forward_size::<8>();
    }

    #[test]
    fn test_ntt_forward_16() {
        test_ntt_forward_size::<16>();
    }

    #[test]
    fn test_ntt_forward_32() {
        test_ntt_forward_size::<32>();
    }

    #[test]
    fn test_ntt_forward_64() {
        test_ntt_forward_size::<64>();
    }

    #[test]
    fn test_ntt_forward_128() {
        test_ntt_forward_size::<128>();
    }

    #[ignore]
    #[test]
    fn test_ntt_stress() {
        const D: usize = 2048;
        const P: u64 = 268369921;
        type RCoeff = IntModCyclo<D, P>;
        type REval = IntModCycloEval<D, P>;

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
