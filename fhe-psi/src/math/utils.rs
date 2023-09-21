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
