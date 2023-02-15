use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand::Rng;

pub trait RingElement:
    Sized
    + Clone
    + PartialEq
    + Eq
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
where
    for<'a> &'a Self: RingElementRef<Self>,
{
    fn zero() -> Self;
    fn one() -> Self;
    fn random<T: Rng>(rng: &mut T) -> Self;
}

pub trait RingElementRef<Owned: RingElement>:
    Sized
    + Clone
    + Copy
    + Add<Self, Output = Owned>
    + Sub<Self, Output = Owned>
    + Mul<Self, Output = Owned>
    + Neg<Output = Owned>
where
    for<'a> &'a Owned: RingElementRef<Owned>,
{
}
