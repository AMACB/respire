//! Miscellaneous math utilities.

/// Computes `ceil(log_base(x))`. This method is not intended for efficiency.
pub const fn ceil_log(base: u64, x: u64) -> usize {
    let mut e = 0;
    let mut y = 1;

    while y < x {
        y *= base;
        e += 1;
    }

    e
}
