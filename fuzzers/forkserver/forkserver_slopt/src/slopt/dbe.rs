use rand_distr::{Distribution, weighted::WeightedIndex};
use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

const DBE_GAMMA: f64 = 0.99;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct DbeArm {
    num_selected: u64,
    num_rewarded: u64,
    total_rewards: f64,
    dis_num_selected: f64,
    sample_mean: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Dbe {
    arms: Vec<DbeArm>,
}

impl Bandit for Dbe {
    fn with_arms(n_arms: usize) -> Self {
        Self {
            arms: (0..n_arms).map(|_| DbeArm::default()).collect(),
        }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut max_avg = 0.0;
        let mut redcoef = 1.0;

        for arm in &self.arms {
            if arm.dis_num_selected > 0.0 && arm.sample_mean > max_avg {
                max_avg = arm.sample_mean;
            }
        }

        if max_avg > 0.0 {
            redcoef = 1.0 / (2.0 * max_avg);
        }

        let mut unsampled = Vec::with_capacity(self.arms.len());

        for i in 0..self.arms.len() {
            if self.arms[i].dis_num_selected <= 0.0 {
                unsampled.push(i);
            }
        }

        if !unsampled.is_empty() {
            let k = rng.random_range(0..unsampled.len());
            return unsampled[k];
        }

        if self.arms.is_empty() {
            return 0;
        }

        let beta = 4.0 + 2.0 * (self.arms.len() as f64);
        let mut weights = vec![0.0; self.arms.len()];

        for i in 0..self.arms.len() {
            let cur = beta * (redcoef *  self.arms[i].sample_mean);
            weights[i] = 2f64.powf(cur);
        }

        let dist = WeightedIndex::new(&weights).expect("invalid weights");
        let index = dist.sample(rng);

        index
    }

    fn add_reward(&mut self, arm_idx: usize, reward: u8) {
        let mut max_avg = 0.0;
        let mut redcoef = 1.0;

        for arm in &self.arms {
            if arm.dis_num_selected > 0.0 && arm.sample_mean > max_avg {
                max_avg = arm.sample_mean;
            }
        }

        if max_avg > 0.0 {
            redcoef = 1.0 / (2.0 * max_avg);
        }

        if redcoef > (1_i64 << 30) as f64 {
            for arm in &mut self.arms {
                arm.total_rewards = 1.0;
                arm.dis_num_selected = 1.0;
                arm.sample_mean = 1.0;
            }
        }

        for arm in &mut self.arms {
            arm.total_rewards *= DBE_GAMMA;
            arm.dis_num_selected *= DBE_GAMMA;
        }

        let arm = &mut self.arms[arm_idx];

        arm.num_selected += 1;
        arm.num_rewarded += reward as u64;
        arm.total_rewards += reward as f64;
        arm.dis_num_selected += 1.0;
        arm.sample_mean = arm.total_rewards / arm.dis_num_selected;
    }
}
