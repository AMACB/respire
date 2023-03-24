use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait RingElement:
    Sized
    + Clone
    + PartialEq
    + Eq
    + From<u64>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
where
    for<'a> &'a Self: RingElementRef<Self>,
{
    fn zero() -> Self;
    fn one() -> Self;
}

pub trait RingElementDivModdable: RingElement
where
    for<'a> &'a Self: RingElementRef<Self>,
{
    fn div_mod(&self, a: u64) -> (Self, Self);
}

pub trait RingElementRef<Owned: RingElement>:
    Sized
    + Clone
    + Add<Self, Output = Owned>
    + Sub<Self, Output = Owned>
    + Mul<Self, Output = Owned>
    + Neg<Output = Owned>
where
    for<'a> &'a Owned: RingElementRef<Owned>,
{
}
