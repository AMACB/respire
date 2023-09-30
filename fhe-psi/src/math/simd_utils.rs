#![allow(clippy::missing_safety_doc)]
use std::arch::x86_64::*;

///
/// Executes `s += a * b` on all four lanes. `a` and `b` are 32 bit; `s` is 64 bit.
///
pub unsafe fn _mm256_ptr_add_eq_mul32(
    s_ptr: *mut __m256i,
    a_ptr: *const __m256i,
    b_ptr: *const __m256i,
) {
    let a = _mm256_load_si256(a_ptr);
    let b = _mm256_load_si256(b_ptr);
    let s = _mm256_load_si256(s_ptr);
    let prod = _mm256_mul_epu32(a, b);
    let sum_prod = _mm256_add_epi64(s, prod);
    _mm256_store_si256(s_ptr, sum_prod);
}

///
/// Compute a representative of `lhs * rhs mod N` in the range `[0, 2N)` on all four lanes.
/// - The input `lhs` must be in the range `[0, 4N)`.
/// - The modulus `N` must satisfy `N < 2^30`.
/// - `rhs` resp. `rhs_ratio32` must be in the range `[0, N)`. The latter value is to be computed via
/// `get_ratio32::<N>` of the former value.
/// - `modulus` must have `N` in all lanes, e.g. via `_mm256_set1_epi64x(N as i64)`
///
#[inline(always)]
pub unsafe fn _mm256_mod_mul32(
    lhs: __m256i,
    rhs: __m256i,
    rhs_ratio32: __m256i,
    modulus: __m256i,
) -> __m256i {
    let quotient = _mm256_srli_epi64::<32>(_mm256_mul_epu32(rhs_ratio32, lhs));
    let lhs_times_rhs = _mm256_mul_epu32(lhs, rhs);
    let modulus_times_quotient = _mm256_mul_epu32(modulus, quotient);
    _mm256_sub_epi64(lhs_times_rhs, modulus_times_quotient)
}

///
/// Reduce the input from the range `[0, 2*modulus)` to `[0, modulus)` on all four lanes.
/// - The modulus must be `< 2^31`.
///
#[inline(always)]
pub unsafe fn _mm256_reduce_half(value: __m256i, modulus: __m256i) -> __m256i {
    _mm256_min_epu32(value, _mm256_sub_epi32(value, modulus))
}
