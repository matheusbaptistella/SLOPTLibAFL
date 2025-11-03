use std::collections::VecDeque;

use libafl_bolts::SerdeAny;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

const ADWIN_M: usize = 10;
const ADWIN_DELTA: f64 = 1e-7;
const ADWIN_MIN_ELEM_TO_START_DROP: u64 = 17;
const ADWIN_MIN_ELEM_TO_CHECK: u64 = 5;
const ADWIN_DROP_INTERVAL: usize = 100; // every N inserts

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct AdwinNode {
    // Oldest at the front, newest at the back
    sums: VecDeque<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Adwin {
    // Level 0 has bucket size, last has size 2^last_node_idx
    levels: VecDeque<AdwinNode>,
    num_add: usize,
    w: u64,
    sum: u64,
}

impl Default for Adwin {
    fn default() -> Self {
        let mut levels = VecDeque::new();
        levels.push_back(AdwinNode::default());

        Self {
            levels,
            w: 0,
            sum: 0,
            num_add: 0,
        }
    }
}

impl Adwin {
    pub fn add(&mut self, reward: u8) {
        self.w += 1;
        self.sum += reward as u64;

        if self.levels.is_empty() {
            self.levels.push_back(AdwinNode::default());
        }

        self.levels[0].sums.push_back(reward as u64);

        self.normalize_buckets();

        if ADWIN_DROP_INTERVAL == 1 {
            self.drop_last_till_identical();
        } else {
            self.num_add += 1;
            if self.num_add == ADWIN_DROP_INTERVAL {
                self.drop_last_till_identical();
                self.num_add = 0
            }
        }
    }

    fn normalize_buckets(&mut self) {
        let mut level_idx = 0;
        while level_idx < self.levels.len() {
            if self.levels[level_idx].sums.len() > ADWIN_M && level_idx + 1 == self.levels.len() {
                self.levels.push_back(AdwinNode::default());
            }

            while self.levels[level_idx].sums.len() > ADWIN_M {
                let s0 = self.levels[level_idx]
                    .sums
                    .pop_front()
                    .expect("overflow implies at least 2 buckets");
                let s1 = self.levels[level_idx]
                    .sums
                    .pop_front()
                    .expect("overflow implies at least 2 buckets");
                let s = s0 + s1;

                if level_idx + 1 == self.levels.len() {
                    self.levels.push_back(AdwinNode::default());
                }

                self.levels[level_idx + 1].sums.push_back(s);
            }

            level_idx += 1;
        }
    }

    fn expire_oldest_bucket(&mut self) {
        let last_idx = self.levels.len() - 1;

        let bucket_size = 1u64
            .checked_shl(last_idx as u32)
            .expect("bucket size overflow");
        let tail = self
            .levels
            .back_mut()
            .expect("at leat one level must exist");

        if let Some(oldest_sum) = tail.sums.pop_front() {
            self.w -= bucket_size;
            self.sum -= oldest_sum;
        }

        if tail.sums.is_empty() && self.levels.len() > 1 {
            self.levels.pop_back();
        }
    }

    #[inline]
    fn should_drop(s0: u64, n0: u64, s1: u64, n1: u64, ddv2: f64, dd2_over_3: f64) -> bool {
        let u0 = s0 as f64 / n0 as f64;
        let u1 = s1 as f64 / n1 as f64;
        let du = (u0 - u1).abs();

        let inv_m = 1.0 / (1.0 + n0 as f64 - ADWIN_MIN_ELEM_TO_CHECK as f64)
            + 1.0 / (1.0 + n1 as f64 - ADWIN_MIN_ELEM_TO_CHECK as f64);
        let eps = (ddv2 * inv_m).sqrt() + dd2_over_3 * inv_m;

        du > eps
    }

    fn try_drop_once(&mut self) -> bool {
        if self.w < ADWIN_MIN_ELEM_TO_START_DROP {
            return false;
        }

        // Precompute constants for this pass
        let n = self.w as f64;
        let dd2 = 2.0 * ((2.0 * n.ln() / ADWIN_DELTA).ln());
        let u = self.sum as f64 / n;
        let ddv2 = u * (1.0 - u) * dd2;
        let dd2_over_3 = dd2 / 3.0;

        // Sliding window split: [0..split) vs [split..w)
        let mut n0: u64 = 0;
        let mut s0: u64 = 0;
        let mut n1: u64 = self.w;
        let mut s1: u64 = self.sum;

        for level_idx in (0..self.levels.len()).rev() {
            // bucket_size = 2^level_idx
            let bucket_size = 1u64
                .checked_shl(level_idx as u32)
                .expect("bucket size overflow");

            // Snapshot to avoid borrow conflicts
            let sums_snapshot: Vec<u64> = self.levels[level_idx].sums.iter().copied().collect();

            for s in sums_snapshot {
                n0 += bucket_size;
                s0 += s;
                n1 -= bucket_size;
                s1 -= s;

                if n1 < ADWIN_MIN_ELEM_TO_CHECK {
                    return false;
                }
                if n0 < ADWIN_MIN_ELEM_TO_CHECK {
                    continue;
                }

                if Self::should_drop(s0, n0, s1, n1, ddv2, dd2_over_3) {
                    self.expire_oldest_bucket();
                    return true; // exactly one drop per pass
                }
            }
        }

        false
    }

    fn drop_last_till_identical(&mut self) {
        if self.w < ADWIN_MIN_ELEM_TO_START_DROP {
            return;
        }
        while self.try_drop_once() {}
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct AdwinArm {
    adwin: Adwin,
    num_selected: u64,
    total_rewards: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, SerdeAny)]
pub struct Adsts {
    arms: Vec<AdwinArm>,
}

impl Bandit for Adsts {
    fn with_arms(n_arms: usize) -> Self {
        let mut arms = Vec::with_capacity(n_arms);
        for _ in 0..n_arms {
            arms.push(AdwinArm::default());
        }

        Self { arms }
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut selected_idx = 0usize;
        let mut max_sampled = -1.0;

        for (i, arm) in self.arms.iter().enumerate() {
            let w = arm.adwin.w;
            let s = arm.adwin.sum;

            let alpha = 1.0 + s as f64;
            let beta = 1.0 + (w - s) as f64;
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
        self.arms[arm_idx].num_selected += 1;
        self.arms[arm_idx].total_rewards += reward as u64;
        self.arms[arm_idx].adwin.add(reward);
    }
}
