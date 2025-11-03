use rand_distr::StandardUniform;
use serde::{Deserialize, Serialize};

use crate::slopt::Bandit;

const EXP_LOWER: f64 = 0.0;
const EXP_AMPLITUDE: f64 = 1.0;
const EXP_ALPHA: f64 = 3.0;
const EXP_BETA: f64 = 256.0;
const NUM_TOL: f64 = 1e-8;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ExpppArm {
    weight: f64,
    loss: f64,
    unweighted_loss: f64,
    total_rewards: u64,
    pulls: u64,
    trust: f64,
}

impl ExpppArm {
    pub fn new_uniform(n_arms: usize) -> Self {
        let u = 1.0 / n_arms as f64;

        Self {
            weight: u,
            loss: n_arms as f64,
            unweighted_loss: 1.0,
            total_rewards: 0,
            pulls: 1,
            trust: u,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Exppp {
    arms: Vec<ExpppArm>,
    t: u64,
}

impl Exppp {
    pub fn exppp_gap_estimate(&self) -> Vec<f64> {
        let mut average_losses = vec![0.0; self.arms.len()];
        let mut exploration_term = vec![0.0; self.arms.len()];
        let mut ucb = vec![0.0; self.arms.len()];
        let mut lcb = vec![0.0; self.arms.len()];

        let t = self.t as f64;
        let n = self.arms.len() as f64;

        let mut min_ucb = f64::INFINITY;

        for i in 0..self.arms.len() {
            let pulls = self.arms[i].pulls as f64;
            average_losses[i] = Self::div_inf(self.arms[i].unweighted_loss, pulls);
            exploration_term[i] = Self::div_inf(EXP_ALPHA * t.ln() + n.ln(), 2.0 * pulls).sqrt();
            ucb[i] = (average_losses[i] + exploration_term[i]).min(1.0);
            lcb[i] = (average_losses[i] - exploration_term[i]).max(0.0);
            min_ucb = min_ucb.min(ucb[i]);
        }

        let mut delta = vec![0.0; self.arms.len()];

        for i in 0..self.arms.len() {
            delta[i] = (lcb[i] - min_ucb).max(0.0);
        }

        delta
    }

    pub fn exppp_xi(&self, arm: usize, gap_estimated: &[f64]) -> f64 {
        let t = self.t as f64;

        Self::div_inf(EXP_BETA * t.ln(), t * gap_estimated[arm].powi(2))
    }

    pub fn exppp_epsilon(&self) -> Vec<f64> {
        let gaps = self.exppp_gap_estimate();
        let n = self.arms.len() as f64;
        let t = self.t as f64;

        let cap1 = 0.5 / n;
        let cap2 = 0.5 * (Self::div_inf(n.ln(), t * n)).sqrt();

        let mut eps = vec![0.0; self.arms.len()];

        for arm in 0..self.arms.len() {
            let xi = self.exppp_xi(arm, &gaps);
            eps[arm] = cap1.min(cap2).min(xi);
        }

        eps
    }

    pub fn exppp_eta(&self) -> f64 {
        let n = self.arms.len() as f64;
        0.5 * (Self::div_inf(n.ln(), n * (self.t + 1) as f64)).sqrt()
    }

    pub fn exppp_update_trusts(&mut self) {
        let epsilons = self.exppp_epsilon();
        let sum_eps: f64 = epsilons.iter().sum();

        let mut sum_trusts = 0.0;

        for i in 0..self.arms.len() {
            let tr = ((1.0 - sum_eps) * self.arms[i].weight) + epsilons[i];
            self.arms[i].trust = tr;
            sum_trusts += tr;
        }

        if sum_trusts < NUM_TOL {
            let u = 1.0 / self.arms.len() as f64;
            for arm in &mut self.arms {
                arm.trust = u;
            }
            sum_trusts = 1.0;
        }

        for arm in &mut self.arms {
            arm.trust /= sum_trusts;
        }
    }

    pub fn choice_from_distr<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let sampled: f64 = rng.sample(StandardUniform);
        let mut cumulative = 0.0;

        for (i, arm) in self.arms.iter().enumerate() {
            cumulative += arm.trust;
            if sampled < cumulative {
                return i;
            }
        }

        self.arms.len() - 1
    }

    #[inline]
    fn div_inf(x: f64, y: f64) -> f64 {
        if y == 0.0 || y == -0.0 {
            f64::INFINITY
        } else {
            x / y
        }
    }
}

impl Bandit for Exppp {
    fn with_arms(n_arms: usize) -> Self {
        let mut arms = Vec::with_capacity(n_arms);
        for _ in 0..n_arms {
            arms.push(ExpppArm::new_uniform(n_arms));
        }
        let mut exppp = Self {
            arms,
            t: 1 + n_arms as u64,
        };

        exppp.exppp_update_trusts();
        exppp
    }

    fn sample_argmax<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        self.choice_from_distr(rng)
    }

    fn add_reward(&mut self, arm_idx: usize, reward: u8) {
        let arm = &mut self.arms[arm_idx];

        arm.pulls += 1;
        arm.total_rewards = arm.total_rewards.saturating_add(reward as u64);

        let reward = (reward as f64 - EXP_LOWER) / EXP_AMPLITUDE;
        let mut loss = 1.0 - reward;

        arm.unweighted_loss += loss;
        loss /= arm.trust;
        arm.loss += loss;

        let eta = self.exppp_eta();
        let mut min_loss_eta = f64::INFINITY;

        for arm in &self.arms {
            min_loss_eta = min_loss_eta.min(-eta * arm.loss);
        }

        let mut sum_weights = 0.0;

        for arm in &mut self.arms {
            arm.weight = (-eta * arm.loss - min_loss_eta).exp();
            sum_weights += arm.weight;
        }

        for arm in &mut self.arms {
            arm.weight /= sum_weights;
        }

        self.exppp_update_trusts();
        self.t += 1;
    }
}
