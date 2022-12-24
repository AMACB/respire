use std::ops::{Add, AddAssign, Sub, SubAssign, Neg, Mul, MulAssign};
use rand::Rng;

pub trait RingElement:
// TODO: no copy
// TODO: all operations move .__.
    Sized + Clone + Copy
    + Add<Self, Output=Self>
    + Sub<Self, Output=Self>
    + Mul<Self, Output=Self>
    + Neg<Output=Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn random<T: Rng>(rng: &mut T) -> Self;
}
