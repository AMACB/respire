use crate::gsw;
use crate::gsw::{Ciphertext, PublicKey};
use crate::z_n::Z_N;

#[derive(PartialEq, Debug)]
pub struct PolyU32<const N: u32> {
    coeff: Vec<u32>,
}

// use crate::ring_elem::*;
// use std::ops::{Add, AddAssign, Sub, SubAssign, Neg, Mul, MulAssign};
// use rand::Rng;
// use rand::distributions::Standard;
// use std::fmt;

// pub struct Polynomial<const D: usize, R: RingElement> {
//     pub coeffs: [R; D]
// }

// // TODO: align pls
// pub struct PolynomialNTT<const D: usize, R: RingElement> {
//     pub evals: [R; D]
// }


impl<const N: u32> PolyU32<N> {
    pub const P: u32 = u32::MAX - 5 + 1;

    pub fn new(mut coeff: Vec<u32>) -> PolyU32<N> {
        let mut idx = coeff.len();
        loop {
            if idx == 0 || coeff[idx - 1] != 0 {
                break;
            }
            idx -= 1;
        }
        coeff.resize(idx, 0);
        return PolyU32 {
            coeff,
        };
    }

    pub fn zero() -> PolyU32<N> {
        PolyU32 {
            coeff: vec![0],
        }
    }

    pub fn one() -> PolyU32<N> {
        PolyU32 {
            coeff: vec![1],
        }
    }

    pub fn eval(&self, x: u32) -> u32 {
        let mut result: u64 = 0;
        let mut current_pow: u64 = 1;
        for a in &self.coeff {
            result += (current_pow as u64) * (*a as u64) % (N as u64);
            result %= N as u64;
            current_pow *= x as u64;
            current_pow %= N as u64;
        }
        result as u32
    }

    pub fn eval_gsw<const NN: usize, const M: usize, const P: u64, const Q: u64, const G_BASE: u64, const G_LEN: usize>(&self, pk: &PublicKey<NN,M,P,Q,G_BASE,G_LEN>, ct: &Ciphertext<NN,M,P,Q,G_BASE,G_LEN>) -> Ciphertext<NN,M,P,Q,G_BASE,G_LEN> {
        let mut result = gsw::gsw::encrypt(&pk, &Z_N::new_u(0));
        let mut current_pow = gsw::gsw::encrypt(&pk, &Z_N::new_u(1));
        for a in &self.coeff {
            let term = &current_pow * &gsw::gsw::encrypt(&pk, &Z_N::new_u(*a as u64));
            result = &result + &term;
            current_pow = &current_pow * ct;
        }
        result
    }

    pub fn mul(&self, other: &PolyU32<N>) -> PolyU32<N> {
        let mut result_coeff = vec![];
        result_coeff.resize(self.coeff.len() + other.coeff.len(), 0);
        for (i, a) in self.coeff.iter().enumerate() {
            for (j, b) in other.coeff.iter().enumerate() {
                result_coeff[i + j] =
                    (((result_coeff[i+j] as u64) + (*a as u64) * (*b as u64))
                        % (N as u64)) as u32;
            }
        }

        PolyU32::new(result_coeff)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const P: u32 = PolyU32::<0>::P;

    #[test]
    fn test_new() {
        let p = PolyU32::<{ PolyU32::<0>::P }>::new(vec![42, 6, 1, 0, 0, 0]);
        let q = PolyU32::<{ PolyU32::<0>::P }>::new(vec![42, 6, 1, 0]);
        let r = PolyU32::<{ PolyU32::<0>::P }>::new(vec![42, 6, 1]);

        assert_eq!(p, q);
        assert_eq!(p, r);
        assert_eq!(q, r);
    }

    #[test]
    fn test_eval() {
        let p = PolyU32::<P>::new(vec![5, 3, 1]);

        assert_eq!(p.eval(P-3), 5);
        assert_eq!(p.eval(P-2), 3);
        assert_eq!(p.eval(P-1), 3);
        assert_eq!(p.eval(0), 5);
        assert_eq!(p.eval(1), 9);
        assert_eq!(p.eval(2), 15);
        assert_eq!(p.eval(3), 23);
    }

    #[test]
    fn test_mul() {
        let p = PolyU32::<P>::new(vec![5, 3, 1]);
        let q = PolyU32::<P>::new(vec![P-4, 2, 1]);
        let r = PolyU32::<P>::new(vec![P-20, P-2, 7, 5, 1]);

        assert_eq!(p.mul(&q), r);
    }
}