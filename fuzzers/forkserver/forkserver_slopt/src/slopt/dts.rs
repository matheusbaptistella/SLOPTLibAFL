use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

const DTS_GAMMA: f64 = 0.9999999;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct DtsArm {
    num_selected: u64,
    num_rewarded: u64,
    total_rewards: f64,
    total_losses: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Dts {
    arms: Vec<DtsArm>,
}

impl Bandit for Dts {
    fn with_arms(n_arms: usize) -> Self {
        Self {
            arms: (0..n_arms).map(|_| DtsArm::default()).collect(),
        }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut max_sampled = -1.0;
        let mut selected_idx = 0;

        for (i, arm) in self.arms.iter().enumerate() {
            let alpha = arm.total_rewards + 1.0;
            let beta = arm.total_losses + 1.0;
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
        // Moved from select_arm
        for arm in &mut self.arms {
            arm.total_rewards *= DTS_GAMMA;
            arm.total_losses *= DTS_GAMMA;
        }

        let arm = &mut self.arms[arm_idx];

        arm.num_selected += 1;
        arm.num_rewarded += reward as u64;
        arm.total_rewards += reward as f64;
        arm.total_losses += 1.0 - (reward as f64);
    }
}
