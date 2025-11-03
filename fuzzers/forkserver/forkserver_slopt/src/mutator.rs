use std::{borrow::Cow, marker::PhantomData};

use libafl::{
    HasMetadata,
    corpus::CorpusId,
    mutators::{ComposedByMutations, MutationResult, Mutator, MutatorsTuple},
    state::{HasCurrentTestcase, HasRand},
};
use libafl_bolts::{Named, tuples::NamedTuple};

use crate::slopt::{Bandit, Slopt};

pub trait SloptScheduledMutator<I, S>: ComposedByMutations + Mutator<I, S>
where
    Self::Mutations: MutatorsTuple<I, S>,
{
    /// Compute the number of iterations used to apply stacked mutations
    fn iterations(&self, state: &mut S, input: &I, bucket_idx: usize, op_idx: usize) -> usize;

    /// Get the next mutation to apply
    fn schedule(&self, state: &mut S, input: &I) -> usize;

    fn scheduled_mutate(
        &mut self,
        state: &mut S,
        input: &mut I,
    ) -> Result<MutationResult, libafl::Error>;
}

#[derive(Debug)]
pub struct HavocSloptScheduledMutator<MT, SL> {
    name: Cow<'static, str>,
    mutations: MT,
    phantom: PhantomData<SL>,
}

impl<MT, SL> Named for HavocSloptScheduledMutator<MT, SL> {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<I, MT, S, SL> Mutator<I, S> for HavocSloptScheduledMutator<MT, SL>
where
    MT: MutatorsTuple<I, S>,
    S: HasRand + HasCurrentTestcase<I> + HasMetadata,
    S::Rand: rand::Rng,
    SL: Slopt,
{
    #[inline]
    fn mutate(&mut self, state: &mut S, input: &mut I) -> Result<MutationResult, libafl::Error> {
        self.scheduled_mutate(state, input)
    }

    #[inline]
    fn post_exec(
        &mut self,
        _state: &mut S,
        _new_corpus_id: Option<CorpusId>,
    ) -> Result<(), libafl::Error> {
        Ok(())
    }
}

impl<MT, SL> ComposedByMutations for HavocSloptScheduledMutator<MT, SL> {
    type Mutations = MT;

    #[inline]
    fn mutations(&self) -> &Self::Mutations {
        &self.mutations
    }

    #[inline]
    fn mutations_mut(&mut self) -> &mut Self::Mutations {
        &mut self.mutations
    }
}

impl<I, MT, S, SL> SloptScheduledMutator<I, S> for HavocSloptScheduledMutator<MT, SL>
where
    MT: MutatorsTuple<I, S>,
    S: HasRand + HasCurrentTestcase<I> + HasMetadata,
    S::Rand: rand::Rng,
    SL: Slopt,
{
    fn iterations(&self, state: &mut S, _input: &I, bucket_idx: usize, op_idx: usize) -> usize {
        let slopt = state.metadata::<SL>().unwrap().clone();
        slopt.bs_bandits()[bucket_idx][op_idx].sample_argmax(state.rand_mut())
    }

    fn schedule(&self, state: &mut S, _input: &I) -> usize {
        let slopt = state.metadata::<SL>().unwrap().clone();
        slopt.op_bandit().sample_argmax(state.rand_mut())
    }

    fn scheduled_mutate(
        &mut self,
        state: &mut S,
        input: &mut I,
    ) -> Result<MutationResult, libafl::Error> {
        let testcase = state.current_testcase().unwrap();
        let len = testcase.input().as_slice().len();
        drop(testcase);

        let mut r = MutationResult::Skipped;
        let op_idx = self.schedule(state, input);
        let bucket_idx = state.metadata::<SL>()?.bucket_len(len);
        let bs_idx = self.iterations(state, input, bucket_idx, op_idx);

        let slopt = state.metadata_mut::<SL>()?;
        slopt.set_last_choices(op_idx, bucket_idx, bs_idx);

        let num = 1 << (1 + bs_idx);
        for _ in 0..num {
            let outcome = self
                .mutations_mut()
                .get_and_mutate(op_idx.into(), state, input)?;
            if outcome == MutationResult::Mutated {
                r = MutationResult::Mutated;
            }
        }
        Ok(r)
    }
}

impl<MT, B> HavocSloptScheduledMutator<MT, B>
where
    MT: NamedTuple,
{
    #[inline]
    pub fn new(mutations: MT) -> Self {
        Self {
            name: Cow::from(format!(
                "HavocSloptScheduledMutator[{}]",
                mutations.names().join(", ")
            )),
            mutations,
            phantom: PhantomData,
        }
    }
}
