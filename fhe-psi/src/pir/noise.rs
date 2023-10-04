use std::ops::{Add, Mul};

#[derive(Clone, Copy, Debug)]
pub struct Independent(pub f64);

#[derive(Clone, Copy, Debug)]
pub struct Scale(pub f64);

#[derive(Clone, Copy, Debug)]
pub struct SubGaussianNoise {
    variance: f64,
    degree: u64,
    rows: usize,
    cols: usize,
}

impl SubGaussianNoise {
    pub fn new(variance: f64, degree: u64) -> Self {
        Self::new_matrix(variance, degree, 1, 1)
    }

    pub fn new_matrix(variance: f64, degree: u64, rows: usize, cols: usize) -> Self {
        Self {
            variance,
            degree,
            rows,
            cols,
        }
    }

    pub fn with_dimension(self, rows: usize, cols: usize) -> Self {
        let mut copy = self;
        copy.rows = rows;
        copy.cols = cols;
        copy
    }

    pub fn variance(self) -> f64 {
        self.variance
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BoundedNoise {
    bound: f64,
    degree: u64,
    rows: usize,
    cols: usize,
}

impl BoundedNoise {
    pub fn new(bound: f64, degree: u64) -> Self {
        Self::new_matrix(bound, degree, 1, 1)
    }
    pub fn new_matrix(bound: f64, degree: u64, rows: usize, cols: usize) -> Self {
        Self {
            bound,
            degree,
            rows,
            cols,
        }
    }

    pub fn with_dimension(self, rows: usize, cols: usize) -> Self {
        let mut copy = self;
        copy.rows = rows;
        copy.cols = cols;
        copy
    }
}

impl Mul<f64> for BoundedNoise {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            bound: self.bound * rhs,
            degree: self.degree,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Mul<Scale> for SubGaussianNoise {
    type Output = Self;
    fn mul(self, rhs: Scale) -> Self {
        Self {
            variance: self.variance * rhs.0 * rhs.0,
            degree: self.degree,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Mul<Independent> for SubGaussianNoise {
    type Output = Self;
    fn mul(self, rhs: Independent) -> Self {
        Self {
            variance: self.variance * rhs.0,
            degree: self.degree,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Add<SubGaussianNoise> for SubGaussianNoise {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.degree, rhs.degree);
        Self {
            variance: self.variance + rhs.variance,
            degree: self.degree,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Mul<BoundedNoise> for SubGaussianNoise {
    type Output = Self;
    fn mul(self, rhs: BoundedNoise) -> Self {
        assert_eq!(self.degree, rhs.degree);
        assert_eq!(self.cols, rhs.rows);
        Self {
            variance: self.variance * (self.degree as f64) * rhs.bound * rhs.bound,
            degree: self.degree,
            rows: self.rows,
            cols: rhs.cols,
        } * Independent(self.cols as f64)
    }
}

impl Mul<SubGaussianNoise> for BoundedNoise {
    type Output = SubGaussianNoise;
    fn mul(self, rhs: SubGaussianNoise) -> SubGaussianNoise {
        assert_eq!(self.degree, rhs.degree);
        assert_eq!(self.cols, rhs.rows);
        SubGaussianNoise {
            variance: rhs.variance * (self.degree as f64) * self.bound * self.bound,
            degree: self.degree,
            rows: self.rows,
            cols: rhs.cols,
        } * Independent(self.cols as f64)
    }
}
