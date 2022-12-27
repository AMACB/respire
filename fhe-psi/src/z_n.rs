use crate::ring_elem::*;
use std::ops::{Add, AddAssign, Sub, SubAssign, Neg, Mul, MulAssign};
use rand::Rng;
use rand::distributions::Standard;
use std::fmt;

#[derive(Clone, Copy)]
pub struct Z_N<const N: u64> {
    a: u64,
}

impl<const N: u64> Z_N<N> {
    pub fn new_u(a: u64) -> Self {
        Z_N {
            a: a % N
        }
    }
    pub fn new_i(a: i64) -> Self {
        Z_N {
            a: (a % (N as i64) + (N as i64)) as u64 % N
        }
    }
    pub fn to_u(self) -> u64 {
        self.a
    }
}

impl<const N: u64> fmt::Debug for Z_N<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.a <= N/2 {
            write!(f, "{}", self.a)
        } else {
            write!(f, "-{}", N-self.a)
        }
    }
}

impl<const N: u64> RingElement for Z_N<N> {
    fn zero() -> Self {
        Z_N { a: 0 }
    }
    fn one() -> Self {
        Z_N { a: 1 }
    }
    fn random<T: Rng>(rng: &mut T) -> Self {
        // TODO: not actually uniform :clown:
        let mut iter = rng.sample_iter(&Standard);
        let val: u64 = iter.next().unwrap();
        Z_N::new_u(val)
    }
}

impl<const N: u64> Neg for Z_N<N> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Z_N { a: N - self.a }
    }
}

impl<const N: u64> Add for Z_N<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Z_N { a: (self.a + rhs.a) % N }
    }
}

impl<const N: u64> AddAssign for Z_N<N> {
    fn add_assign(&mut self, rhs: Self) {
        self.a = (self.a + rhs.a) % N;
    }
}

impl<const N: u64> Sub for Z_N<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Z_N { a: (N + self.a - rhs.a) % N }
    }
}

impl<const N: u64> SubAssign for Z_N<N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a = (N + self.a - rhs.a) % N;
    }
}

impl<const N: u64> Mul for Z_N<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        // TODO: overflows when N exceeds u32
        let a = (self.a * rhs.a) % N;
        Z_N { a }
    }
}

impl<const N: u64> MulAssign for Z_N<N> {
    fn mul_assign(&mut self, rhs: Self) {
        // TODO: overflows when N exceeds u32
        self.a = (self.a * rhs.a) % N;
    }
}

