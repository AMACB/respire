//! Miscellaneous math utilities.

/// Computes `ceil(log_base(x))`. This method is not intended for efficiency.
pub const fn ceil_log(base: u64, x: u64) -> usize {
    assert!(base > 1);
    assert!(x > 0);
    let mut e = 0;
    let mut y = 1;

    while y < x {
        y *= base;
        e += 1;
    }

    e
}

struct ReverseBitsTable<const MAX: usize> {}
impl<const MAX: usize> ReverseBitsTable<MAX> {
    const TABLE: [usize; MAX] = make_reverse_bits_table::<MAX>();
}

const fn make_reverse_bits_table<const MAX: usize>() -> [usize; MAX] {
    let mut result = [0; MAX];
    let mut i = 0;
    let width = ceil_log(2, MAX as u64 - 1);
    while i < MAX {
        result[i] = i.reverse_bits() >> ((usize::BITS as usize) - width);
        i += 1;
    }
    result
}

///
/// Compute the bit reversal of `a`, assumed to be of length `d` where `MAX = 2^d`. Note that this
/// generates a lookup table of size `MAX`.
///
pub const fn reverse_bits_fast<const MAX: usize>(a: usize) -> usize {
    ReverseBitsTable::<MAX>::TABLE[a]
}

pub const fn reverse_bits(max: usize, a: usize) -> usize {
    a.reverse_bits() >> ((usize::BITS as usize) - ceil_log(2, max as u64 - 1))
}

pub const fn floor_log(base: u64, mut x: u64) -> usize {
    assert!(base > 1);
    assert!(x > 0);
    let mut e = 0;

    while x >= base {
        x /= base;
        e += 1;
    }

    e
}

pub const fn mod_inverse(x: u64, modulus: u64) -> u64 {
    let modulus = modulus as i64;
    let mut a = x as i64;
    let mut b = modulus;
    let mut x = 1;
    let mut y = 0;
    while a > 1 {
        let q = a / b;

        let t = b;
        b = a % b;
        a = t;

        let t = y;
        y = (x - (q * y)) % modulus;
        x = t;
    }

    (x + (x < (0_i64)) as i64 * modulus) as u64
}

///
/// Compute the 32-bit barrett reduction ratio.
///
pub const fn get_ratio32<const N: u64>(a: u64) -> u64 {
    (a << 32) / N
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ceil_log() {
        assert_eq!(ceil_log(2, 1), 0);
        assert_eq!(ceil_log(2, 2), 1);
        assert_eq!(ceil_log(2, 3), 2);
        assert_eq!(ceil_log(2, 4), 2);
        assert_eq!(ceil_log(2, 5), 3);
        assert_eq!(ceil_log(2, 6), 3);
        assert_eq!(ceil_log(2, 7), 3);
        assert_eq!(ceil_log(2, 8), 3);
        assert_eq!(ceil_log(2, 9), 4);

        assert_eq!(ceil_log(2, (1 << 31) - 1), 31);
        assert_eq!(ceil_log(2, 1 << 31), 31);
        assert_eq!(ceil_log(2, (1 << 31) + 1), 32);
        assert_eq!(ceil_log(4, 1 << 31), 16);
        assert_eq!(ceil_log(4, (1 << 31) + 1), 16);

        assert_eq!(ceil_log(354, 44361864), 3);
        assert_eq!(ceil_log(354, 44361865), 4);
    }

    #[test]
    fn test_floor_log() {
        assert_eq!(floor_log(2, 1), 0);
        assert_eq!(floor_log(2, 2), 1);
        assert_eq!(floor_log(2, 3), 1);
        assert_eq!(floor_log(2, 4), 2);
        assert_eq!(floor_log(2, 5), 2);
        assert_eq!(floor_log(2, 6), 2);
        assert_eq!(floor_log(2, 7), 2);
        assert_eq!(floor_log(2, 8), 3);
        assert_eq!(floor_log(2, 9), 3);

        assert_eq!(floor_log(2, (1 << 31) - 1), 30);
        assert_eq!(floor_log(2, 1 << 31), 31);
        assert_eq!(floor_log(2, (1 << 31) + 1), 31);
        assert_eq!(floor_log(4, (1 << 30) - 1), 14);
        assert_eq!(floor_log(4, 1 << 30), 15);
        assert_eq!(floor_log(4, (1 << 30) + 1), 15);

        assert_eq!(floor_log(354, 44361863), 2);
        assert_eq!(floor_log(354, 44361864), 3);
    }

    #[test]
    fn test_mod_inverse() {
        for i in 1..17 {
            assert_eq!(mod_inverse(i, 17) * i % 17, 1);
        }
    }
}
