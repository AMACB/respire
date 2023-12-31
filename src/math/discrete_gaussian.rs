//! Discrete gaussian distribution over the integers. Related to [RandDiscreteGaussianSampled].

use once_cell::sync::Lazy;
use rand::prelude::Distribution;
use rand::Rng;
use rand_distr::WeightedAliasIndex;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::RwLock;

/// Number of std deviations (width) to sample from. A z-score of 8 has probability < 1.23 x 10^-15.
pub const NUM_WIDTHS: usize = 8;

/// A single table used for sampling a discrete gaussian of a particular width.
struct DiscreteGaussianTable {
    choices: Vec<i64>,
    dist: WeightedAliasIndex<f64>,
}

impl DiscreteGaussianTable {
    fn init(noise_width: f64) -> Self {
        let max_val = (noise_width * (NUM_WIDTHS as f64)).ceil() as i64;
        let mut choices = Vec::new();
        let mut weights = vec![0f64; 0];
        for i in -max_val..max_val + 1 {
            let p_val = f64::exp(-PI * f64::powi(i as f64, 2) / f64::powi(noise_width, 2));
            choices.push(i);
            weights.push(p_val);
        }
        let dist = WeightedAliasIndex::new(weights).unwrap();
        Self { choices, dist }
    }

    fn sample<T: Rng>(&self, rng: &mut T) -> i64 {
        self.choices[self.dist.sample(rng)]
    }
}

/// Memoization table for discrete gaussian sampling.
static DISCRETE_GAUSSIAN_TABLES: Lazy<RwLock<HashMap<u64, DiscreteGaussianTable>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Discrete gaussian distributions
pub struct DiscreteGaussian {}

impl DiscreteGaussian {
    /// Samples a discrete gaussian of the given width. This function memoized based on the noise
    /// width, so the first call of a particular noise width will take longer than future calls.
    pub fn sample<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> i64 {
        if let Some(table) = DISCRETE_GAUSSIAN_TABLES
            .read()
            .unwrap()
            .get(&NOISE_WIDTH_MILLIONTHS)
        {
            return table.sample(rng);
        }

        let table = DiscreteGaussianTable::init(NOISE_WIDTH_MILLIONTHS as f64 / 1_000_000_f64);
        let ret = table.sample(rng);
        DISCRETE_GAUSSIAN_TABLES
            .write()
            .unwrap()
            .insert(NOISE_WIDTH_MILLIONTHS, table);
        ret
    }
}
