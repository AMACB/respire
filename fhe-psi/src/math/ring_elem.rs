//! Generic ring-related traits.
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A generic element of a ring (with `1`), supporting the operations `+=`, `-=`, `*=` and the
/// special elements `0` and `1`.
///
/// The `From<u64>` trait is expected to be implemented as the canonical map `Z -> R`. That is, it
/// ought to be equal to `1 + ... + 1` for the appropriate number of `1`s.
///
/// Non-inplace arithmetic is intentionally not part of this trait. In general, a `RingElement`
/// need not be `Clone`. To work around this, see [`RingElementRef`].
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
    /// Constructs the zero element (additive identity) of the ring.
    fn zero() -> Self;
    /// Constructs the one element (multiplicative identity) of the ring.
    fn one() -> Self;

    /// Add `a * b` to `self` in-place. This method is used for matrix multiplication, so optimizing
    /// it may be desirable.
    fn add_eq_mul(&mut self, a: &Self, b: &Self) {
        *self += &(a * b);
    }
}

/// A reference to a RingElement that supports non-inplace ring operations. This is required for
/// e.g. matrices over a ring to avoid possibly expensive copying.
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

pub trait NormedRingElement: RingElement
where
    for<'a> &'a Self: RingElementRef<Self>,
{
    fn norm(&self) -> u64;
}
