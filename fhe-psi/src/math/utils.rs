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

        assert_eq!(ceil_log(2, 1 << 31), 31);
        assert_eq!(ceil_log(2, 1 << 31 + 1), 32);
        assert_eq!(ceil_log(4, 1 << 31), 16);
        assert_eq!(ceil_log(4, 1 << 31 + 1), 16);

        assert_eq!(ceil_log(354, 44361864), 3);
        assert_eq!(ceil_log(354, 44361865), 4);
    }
}
