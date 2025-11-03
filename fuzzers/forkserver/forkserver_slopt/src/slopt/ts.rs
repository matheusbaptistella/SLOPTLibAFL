use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct TsArm {
    num_selected: u64,
    total_rewards: u64,
    sample_mean: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Ts {
    arms: Vec<TsArm>,
}

impl Bandit for Ts {
    fn with_arms(n_arms: usize) -> Self {
        Self {
            arms: (0..n_arms).map(|_| TsArm::default()).collect(),
        }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut max_sampled = -1.0;
        let mut selected_idx = 0;

        for (i, arm) in self.arms.iter().enumerate() {
            let alpha = (arm.total_rewards + 1) as f64;
            let beta = (arm.num_selected - arm.total_rewards + 1) as f64;
            let sampled = Beta::new(alpha, beta)
                .expect("invalid beta distr")
                .sample(rng);

            if sampled > max_sampled {
                max_sampled = sampled;
                selected_idx = i;
            }
        }

        selected_idx
    }

    fn add_reward(&mut self, arm_idx: usize, reward: u8) {
        let arm = &mut self.arms[arm_idx];
        arm.num_selected += 1;
        arm.total_rewards += reward as u64;
        arm.sample_mean = arm.total_rewards as f64 / arm.num_selected as f64;
    }
}
