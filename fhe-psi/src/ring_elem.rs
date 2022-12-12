use std::ops::{Add, AddAssign, Sub, SubAssign, Neg, Mul, MulAssign};
use rand::Rng;

pub trait RingElement:
// TODO: where might clean this up a lot
// TODO: no copy
// TODO: all operations move .__.
    Sized + Clone + Copy
    + Add<Self> + Add<<Self as Add>::Output> + Add<<Self as Mul>::Output>
    + AddAssign<Self> + AddAssign<<Self as Add>::Output> + AddAssign<<Self as Mul>::Output>
    + Sub<Self> + Sub<<Self as Add>::Output> + Sub<<Self as Mul>::Output>
    + SubAssign<Self> + SubAssign<<Self as Add>::Output> + SubAssign<<Self as Mul>::Output>
    + Neg
    + Mul<Self> + Mul<<Self as Add>::Output> + Mul<<Self as Mul>::Output>
    + MulAssign<Self> + MulAssign<<Self as Add>::Output> + MulAssign<<Self as Mul>::Output>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn random<T: Rng>(rng: &mut T) -> Self;
}
