use crate::math::ring_elem::*;
use crate::math::z_n::Z_N;

// TODO: this should go somewhere else probably
pub fn pow<const N: u64>(mut val: Z_N<N>, mut e: u64) -> Z_N<N> {
    let mut res = Z_N::one();
    while e > 0 {
        if (e & 1) == 1 {
            res *= val;
        }
        e >>= 1;
        val *= val;
    }
    return res;
}

// TODO: this should also go somewhere else, aslo this is not efficient
pub fn inverse<const N: u64>(val: Z_N<N>) -> Z_N<N> {
    return pow(val, N-2);
}

pub fn ntt<const D: usize, const N: u64>(values: &mut [Z_N<N>; D], root: Z_N<N>, log_d: usize) {
    // Cooley Tukey
    for round in 0..log_d {
        let prev_block_size = 1 << round;
        let w_m = pow(root, (D as u64) >> (round + 1));

        for block_start in (0..D).step_by(prev_block_size * 2) {
            let mut w : Z_N<N> = 1u64.into();
            for i in block_start..block_start+prev_block_size {
                let x = values[i];
                let y = w * (values[i + prev_block_size]);
                values[i] = x+y;
                values[i+prev_block_size] = x-y;
                w *= w_m;
            }
        }
    }
}

pub fn bit_reverse_order<const D: usize, const N: u64>(values: &mut [Z_N<N>; D], log_d: usize) {
    for i in 0..D {
        let mut ri = i.reverse_bits();
        ri >>= (usize::BITS as usize) - log_d;
        if i < ri {
            let tmp = values[ri];
            values[ri] = values[i];
            values[i] = tmp;
        }
    }
}

mod test {
    use super::*;

    const D: usize = 4;
    const LOG_D: usize = 2;
    const P: u64 = 268369921u64;
    const W: u64 = 180556700u64; // order = 1 << 16

    // TODO: add more tests.
    #[test]
    fn ntt_self_inverse() {
        let mut coeff : [Z_N<P> ; 4] = [1u64.into(), 2u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let coeff_orig = coeff.clone();
        let root = pow(W.into(), 1 << 14);

        bit_reverse_order(&mut coeff, LOG_D);
        ntt(&mut coeff, root, LOG_D);
        bit_reverse_order(&mut coeff, LOG_D);
        ntt(&mut coeff, inverse(root), LOG_D);

        for i in 0..coeff.len() {
            coeff[i] *= inverse((D as u64).into());
        }

        assert_eq!(coeff, coeff_orig);
    }

    #[test]
    fn forward_ntt() {
        let mut coeff : [Z_N<P> ; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        let root = pow(W.into(), 1 << 14);

        bit_reverse_order(&mut coeff, LOG_D);
        ntt(&mut coeff, root, LOG_D);

        let one = Z_N::one();
        let evaluated = [one+one, root+one, root*root+one, root*root*root+one];

        assert_eq!(coeff, evaluated);
    }

    #[test]
    fn backward_ntt() {
        let root = pow(W.into(), 1 << 14);
        let one = Z_N::one();

        let mut evaluated = [one+one, root+one, root*root+one, root*root*root+one];

        bit_reverse_order(&mut evaluated, LOG_D);
        ntt(&mut evaluated, inverse(root), LOG_D);

        for i in 0..evaluated.len() {
            evaluated[i] *= inverse((D as u64).into());
        }

        let coeff : [Z_N<P> ; 4] = [1u64.into(), 1u64.into(), 0u64.into(), 0u64.into()]; // 1 + x

        assert_eq!(coeff, evaluated);
    }


    #[test]
    fn test_fancy_ntt() {
        let mut coeff1 : [Z_N<P> ; D] = [1u64.into(), 2u64.into(), 3u64.into(), 4u64.into()];
        let mut coeff2 : [Z_N<P> ; D] = [1u64.into(), 1u64.into(), 1u64.into(), 1u64.into()];
        let ans : [Z_N<P> ; D] = [(P-8).into(), (P-4).into(), 2u64.into(), 10u64.into()];

        let root = pow(W.into(), 1 << 13);

        for i in 0..coeff1.len() {
            coeff1[i] *= pow(root, i as u64);
            coeff2[i] *= pow(root, i as u64);
        }

        bit_reverse_order(&mut coeff1, LOG_D);
        ntt(&mut coeff1, root*root, LOG_D);
        bit_reverse_order(&mut coeff2, LOG_D);
        ntt(&mut coeff2, root*root, LOG_D);

        let mut coeff3 = [0u64.into(); 4];
        for i in 0..coeff1.len() {
            coeff3[i] = coeff1[i] * coeff2[i];
        }

        bit_reverse_order(&mut coeff3, LOG_D);
        ntt(&mut coeff3, inverse(root * root), LOG_D);

        for i in 0..coeff3.len() {
            coeff3[i] *= inverse((D as u64).into());
            coeff3[i] *= inverse(pow(root, i as u64));
        }
        assert_eq!(coeff3, ans);
    }
}
