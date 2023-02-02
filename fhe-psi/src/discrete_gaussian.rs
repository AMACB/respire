use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand_chacha::ChaCha20Rng;
use crate::z_n::*;
use crate::matrix::*;
use std::f64::consts::PI;

pub const NUM_WIDTHS: usize = 8;

pub struct DiscreteGaussian {
    choices: Vec<i64>,
    dist: WeightedIndex<f64>,
}

impl DiscreteGaussian {
    pub fn init(noise_width: f64) -> Self {
        let max_val = (noise_width * (NUM_WIDTHS as f64)).ceil() as i64;
        let mut choices = Vec::new();
        let mut table = vec![0f64; 0];
        for i in -max_val..max_val + 1 {
            let p_val = f64::exp(-PI * f64::powi(i as f64, 2) / f64::powi(noise_width, 2));
            choices.push(i);
            table.push(p_val);
        }
        let dist = WeightedIndex::new(&table).unwrap();

        Self { choices, dist }
    }

    // FIXME: not constant-time
    pub fn sample<const Q: u64>(&self, rng: &mut ChaCha20Rng) -> Z_N<Q> {
        Z_N::new_i(self.choices[self.dist.sample(rng)])
    }

    pub fn sample_int_matrix<const N: usize, const M: usize, const Q: u64>(&self, rng: &mut ChaCha20Rng) -> Matrix<N, M, Z_N<Q>> {
        let mut mat = Matrix::zero();
        for r in 0..N {
            for c in 0..M {
                mat[(r,c)] = self.sample(rng);
            }
        }
        mat
    }
}
