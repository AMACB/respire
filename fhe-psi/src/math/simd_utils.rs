use std::arch::x86_64::*;

///
/// Compute a representative of `lhs * rhs mod N` in the range `[0, 2N)` on all four lanes.
/// - The input `lhs` must be in the range `[0, 4N)`.
/// - The modulus `N` must satisfy `N < 2^30`.
/// - `rhs` resp. `rhs_ratio32` must be in the range `[0, N)`. The latter value is to be computed via
/// `get_ratio32::<N>` of the former value.
/// - `neg_modulus` must have `-N` in all lanes, e.g. via `_mm256_set1_epi64x(-(N as i64))`
///
pub unsafe fn _mm256_mod_mul32(
    lhs: __m256i,
    rhs: __m256i,
    rhs_ratio32: __m256i,
    neg_modulus: __m256i,
) -> __m256i {
    let quotient = _mm256_srli_epi64::<32>(_mm256_mul_epu32(rhs_ratio32, lhs));
    let lhs_times_rhs = _mm256_mullo_epi32(lhs, rhs);
    let neg_modulus_times_quotient = _mm256_mullo_epi32(neg_modulus, quotient);
    _mm256_add_epi32(lhs_times_rhs, neg_modulus_times_quotient)
}

///
/// Reduce the input from the range `[0, 2*modulus)` to `[0, modulus)` on all four lanes.
/// - The modulus must be `< 2^31`.
///
pub unsafe fn _mm256_reduce_half(value: __m256i, modulus: __m256i) -> __m256i {
    _mm256_min_epu32(value, _mm256_sub_epi32(value, modulus))
}
