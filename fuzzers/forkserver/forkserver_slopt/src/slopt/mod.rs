use core::fmt::Debug;

use libafl_bolts::serdeany::SerdeAny;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub mod adsts;
pub use adsts::*;

pub mod dbe;
pub use dbe::*;

pub mod dts;
pub use dts::*;

pub mod expix;
pub use expix::*;

pub mod exppp;
pub use exppp::*;

pub mod klucb;
pub use klucb::*;

pub mod ts;
pub use ts::*;

pub mod ucb1;
pub use ucb1::*;

pub const BUCKET: [usize; 4] = [100, 1_000, 10_000, 100_000]; // {[0, 100), [100, 1_000), [1_000, 10_000), [10_000, 100_000), [100_000, +inf)}
pub const MAX_STACK_POW: usize = 7; // [0, 6]

pub trait Bandit: Sized {
    fn with_arms(n_arms: usize) -> Self;
    fn sample_argmax<R: rand::Rng>(&self, rng: &mut R) -> usize;
    fn add_reward(&mut self, arm_idx: usize, reward: u8);
}

pub trait Slopt: SerdeAny + Clone {
    type MAB: Bandit;

    fn op_bandit(&self) -> &Self::MAB;
    fn op_bandit_mut(&mut self) -> &mut Self::MAB;

    fn bs_bandits(&self) -> &Vec<Vec<Self::MAB>>;
    fn bs_bandits_mut(&mut self) -> &mut Vec<Vec<Self::MAB>>;

    fn bucket_len(&self, len: usize) -> usize;

    fn last_choices(&self) -> (usize, usize, usize);
    fn set_last_choices(&mut self, op: usize, bucket: usize, bs: usize);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "B: Bandit + for<'a> Deserialize<'a> + Serialize")]
pub struct SloptMetadata<B> {
    op_bandit: B,
    bs_bandit: Vec<Vec<B>>,
    bucket: Vec<usize>,
    last_choices: (usize, usize, usize),
}

libafl_bolts::impl_serdeany!(SloptMetadata<B: Bandit + Debug + 'static + Serialize + DeserializeOwned>, <Adsts>, <Dbe>, <Dts>, <Expix>, <Exppp>, <KLUcb>, <Ts>, <Ucb1>);

impl<B: Bandit> SloptMetadata<B> {
    pub fn new(mutations: usize, max_stack_pow: usize, bucket: Vec<usize>) -> Self {
        let op_bandit = B::with_arms(mutations);
        let len = bucket.len() + 1;
        let mut bs_bandit = Vec::with_capacity(len);

        for _ in 0..len {
            let mut inner = Vec::with_capacity(mutations);
            for _ in 0..mutations {
                inner.push(B::with_arms(max_stack_pow));
            }
            bs_bandit.push(inner);
        }

        Self {
            op_bandit,
            bs_bandit,
            bucket,
            last_choices: (0, 0, 0),
        }
    }
}

impl<B> Slopt for SloptMetadata<B>
where
    B: Bandit + Clone + Debug + for<'a> Deserialize<'a> + Serialize + 'static,
{
    type MAB = B;

    fn op_bandit(&self) -> &Self::MAB {
        &self.op_bandit
    }

    fn op_bandit_mut(&mut self) -> &mut Self::MAB {
        &mut self.op_bandit
    }

    fn bs_bandits(&self) -> &Vec<Vec<Self::MAB>> {
        &self.bs_bandit
    }

    fn bs_bandits_mut(&mut self) -> &mut Vec<Vec<Self::MAB>> {
        &mut self.bs_bandit
    }

    fn bucket_len(&self, len: usize) -> usize {
        self.bucket
            .iter()
            .position(|&bound| len < bound)
            .unwrap_or(self.bucket.len())
    }

    fn last_choices(&self) -> (usize, usize, usize) {
        self.last_choices
    }

    fn set_last_choices(&mut self, op: usize, bucket: usize, bs: usize) {
        self.last_choices = (op, bucket, bs)
    }
}
