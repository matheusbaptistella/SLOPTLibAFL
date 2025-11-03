use rand_distr::StandardUniform;
use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ExpixArm {
    weight: f64,
    loss: f64,
    total_rewards: u64,
    pulls: u64,
}

impl ExpixArm {
    pub fn new_uniform(n_arms: usize) -> Self {
        Self {
            weight: 1.0 / n_arms as f64,
            loss: 0.0,
            total_rewards: 0,
            pulls: 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Expix {
    arms: Vec<ExpixArm>,
    t: u64,
}

impl Bandit for Expix {
    fn with_arms(n_arms: usize) -> Self {
        let mut arms = Vec::with_capacity(n_arms);
        for _ in 0..n_arms {
            arms.push(ExpixArm::new_uniform(n_arms));
        }

        Self { arms, t: 1 }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut sum_of_possibility = 0.0;
        let sampled: f64 = rng.sample(StandardUniform);

        for i in 0..self.arms.len() {
            sum_of_possibility += self.arms[i].weight;

            if sampled < sum_of_possibility {
                return i;
            }
        }
        let last = self.arms.len() - 1;

        last
    }

    fn add_reward(&mut self, arm_idx: usize, reward: u8) {
        let n = self.arms.len() as f64;
        let arm = &mut self.arms[arm_idx];

        arm.pulls += 1;
        arm.total_rewards = arm.total_rewards.saturating_add(reward as u64);

        let t = self.t as f64;

        let eta = (2.0 * n.ln() / n / t).sqrt();
        let gamma = eta / 2.0;

        let mut loss = 1.0 - reward as f64;

        loss /= arm.weight + gamma;
        arm.loss += loss;

        let mut min_loss = f64::INFINITY;

        for arm in &self.arms {
            if arm.loss < min_loss {
                min_loss = arm.loss;
            }
        }

        let mut denom = 0.0;

        for arm in &mut self.arms {
            arm.weight = (-eta * (arm.loss - min_loss)).exp();
            denom += arm.weight;
        }

        for arm in &mut self.arms {
            arm.weight /= denom;
        }

        self.t += 1;
    }
}
