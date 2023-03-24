use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::RwLock;

use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::Rng;

pub const NUM_WIDTHS: usize = 8;

struct DiscreteGaussianTable {
    choices: Vec<i64>,
    dist: WeightedIndex<f64>,
}

static DISCRETE_GAUSSIAN_TABLES: Lazy<RwLock<HashMap<u64, DiscreteGaussianTable>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

impl DiscreteGaussianTable {
    fn init(noise_width: f64) -> Self {
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
    fn sample<T: Rng>(&self, rng: &mut T) -> i64 {
        self.choices[self.dist.sample(rng)]
    }
}

pub struct DiscreteGaussian {}

impl DiscreteGaussian {
    pub fn sample<T: Rng, const NOISE_WIDTH_MILLIONTHS: u64>(rng: &mut T) -> i64 {
        if let Some(table) = DISCRETE_GAUSSIAN_TABLES
            .read()
            .unwrap()
            .get(&NOISE_WIDTH_MILLIONTHS)
        {
            return table.sample(rng)
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
