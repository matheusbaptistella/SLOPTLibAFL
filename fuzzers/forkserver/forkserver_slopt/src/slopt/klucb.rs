use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

const KLUCB_DELTA: f64 = 1e-8;
const KLUCB_EPS: f64 = 1e-12;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct KLUcbArm {
    num_selected: u64,
    total_rewards: u64,
    sample_mean: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KLUcb {
    arms: Vec<KLUcbArm>,
    time_step: f64,
}

impl KLUcb {
    fn klucb_klucb(&self, arm_idx: usize) -> f64 {
        let logndn = self.time_step.ln() / self.arms[arm_idx].num_selected as f64;
        let p = self.arms[arm_idx].sample_mean.max(KLUCB_DELTA);

        if p >= 1.0 {
            return 1.0;
        }

        let mut q = p + KLUCB_DELTA;

        for _ in 0..25 {
            let f = logndn - Self::kl(p, q);
            let df = -Self::dkl(p, q);

            if (f * f) < KLUCB_EPS {
                break;
            }

            q -= f / df;

            if q < (p + KLUCB_DELTA) {
                q = p + KLUCB_DELTA
            }
            if q > (1.0 - KLUCB_DELTA) {
                q = 1.0 - KLUCB_DELTA
            }
        }

        return q;
    }

    #[inline]
    fn kl(p: f64, q: f64) -> f64 {
        return p * (p / q).ln() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).ln();
    }

    #[inline]
    fn dkl(p: f64, q: f64) -> f64 {
        return (q - p) / (q * (1.0 - q));
    }
}

impl Bandit for KLUcb {
    fn with_arms(n_arms: usize) -> Self {
        Self {
            arms: (0..n_arms).map(|_| KLUcbArm::default()).collect(),
            time_step: 0.0,
        }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> usize {
        let mut max_ucb = -1.0;
        let mut selected_idx = 0;

        for (i, arm) in self.arms.iter().enumerate() {
            if arm.num_selected == 0 {
                selected_idx = i;
                break;
            }

            let ucb = self.klucb_klucb(i);

            if ucb > max_ucb {
                max_ucb = ucb;
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
