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
}

/// A ring element where we can sensibly perform division by a scalar. This is useful for computing
/// `n`-ary decompositions of ring elements.
///
/// Mathematically, implementing this trait means (1) there is a canonical representative of `R/aR`
/// and (2) this can be efficiently computed.
pub trait RingElementDivModdable: RingElement
where
    for<'a> &'a Self: RingElementRef<Self>,
{
    /// Computes the `(quotient, remainder)` upon division by `a`.
    fn div_mod(&self, a: u64) -> (Self, Self);
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
