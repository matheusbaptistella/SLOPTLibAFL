use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct Ucb1Arm {
    num_selected: u64,
    total_rewards: u64,
    sample_mean: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Ucb1 {
    arms: Vec<Ucb1Arm>,
    time_step: f64,
}

impl Bandit for Ucb1 {
    fn with_arms(n_arms: usize) -> Self {
        Self {
            arms: (0..n_arms).map(|_| Ucb1Arm::default()).collect(),
            time_step: 0.0,
        }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> usize {
        let mut max_sampled = -1.0;
        let mut selected_idx = 0;

        for (i, arm) in self.arms.iter().enumerate() {
            if arm.num_selected == 0 {
                selected_idx = i;
                break;
            }

            let sampled =
                arm.sample_mean + (2.0 * self.time_step.ln() / arm.num_selected as f64).sqrt();

            if sampled > max_sampled {
                max_sampled = sampled;
                selected_idx = i;
            }
        }

        selected_idx
    }

    fn add_reward(&mut self, arm_idx: usize, reward: u8) {
        self.time_step += 1.0;

        let arm = &mut self.arms[arm_idx];

        arm.num_selected += 1;
        arm.total_rewards += reward as u64;
        arm.sample_mean = arm.total_rewards as f64 / arm.num_selected as f64;
    }
}
