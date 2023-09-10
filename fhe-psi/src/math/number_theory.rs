use crate::math::utils::floor_log;

pub const fn mod_pow(mut a: u64, mut e: u64, n: u64) -> u64 {
    let mut res = 1_u64;
    while e > 0 {
        if (e & 1) == 1 {
            res = (((res as u128) * (a as u128)) % (n as u128)) as u64;
        }
        e >>= 1;
        a = (((a as u128) * (a as u128)) % (n as u128)) as u64;
    }
    res
}

pub const fn find_sqrt_primitive_root(degree: usize, modulus: u64) -> u64 {
    let double_degree = (2 * degree) as u64;
    if (modulus - 1) % double_degree != 0 {
        panic!("invalid degree / modulus");
    }
    let quotient_size = (modulus - 1) / double_degree;
    let log_double_degree = floor_log(2, double_degree);
    if 1 << log_double_degree != double_degree {
        panic!("invalid degree");
    }
    let mut base = 2;
    'tries: while base < modulus {
        let candidate = mod_pow(base, quotient_size, modulus);
        let mut curr_pow = candidate;
        let mut i = 0;
        'checks: while i < log_double_degree {
            if curr_pow == 1 {
                base += 1;
                continue 'tries;
            }
            curr_pow = mod_pow(curr_pow, 2, modulus);
            i += 1;
            continue 'checks;
        }
        if curr_pow != 1 {
            panic!("unexpected: primitive root failed")
        }
        return candidate;
    }
    panic!("unexpected: no primitive root found")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(11, 65520, 268369921), 228368554);
    }

    #[test]
    fn test_find_sqrt_primitive_root() {
        let prim1 = find_sqrt_primitive_root(2048, 268369921);
        assert_eq!(prim1, mod_pow(11, (268369921 - 1) / 4096, 268369921));

        let prim2 = find_sqrt_primitive_root(2048, 249561089);
        assert_eq!(prim2, mod_pow(3, (249561089 - 1) / 4096, 249561089));
    }
}
