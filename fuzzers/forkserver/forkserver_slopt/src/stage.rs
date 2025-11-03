use std::{borrow::Cow, marker::PhantomData};

use libafl::{
    Evaluator, HasMetadata, HasNamedMetadata,
    corpus::HasCurrentCorpusId,
    inputs::Input,
    mutators::Mutator,
    stages::{
        MutationalStage, Restartable, RetryCountRestartHelper, Stage,
        mutational::{MutatedTransform, MutatedTransformPost},
    },
    state::{HasCorpus, HasCurrentTestcase, HasExecutions, HasRand, MaybeHasClientPerfMonitor},
};
use libafl_bolts::Named;

/// The unique id for mutational stage
static mut MUTATIONAL_STAGE_ID: usize = 0;
/// The name for mutational stage
pub static MUTATIONAL_STAGE_NAME: &str = "slopt-mutational";

#[derive(Debug, Clone)]
pub struct SloptMutationalStage<E, EM, I1, I2, M, S, Z> {
    /// The name
    name: Cow<'static, str>,
    /// The mutator(s) to use
    mutator: M,
    /// The maximum amount of iterations we should do each round
    phantom: PhantomData<(E, EM, I1, I2, S, Z)>,
}

impl<E, EM, I1, I2, M, S, Z> MutationalStage<S> for SloptMutationalStage<E, EM, I1, I2, M, S, Z>
where
    S: HasRand,
{
    type Mutator = M;

    /// The mutator, added to this stage
    #[inline]
    fn mutator(&self) -> &Self::Mutator {
        &self.mutator
    }

    /// The list of mutators, added to this stage (as mutable ref)
    #[inline]
    fn mutator_mut(&mut self) -> &mut Self::Mutator {
        &mut self.mutator
    }

    /// Gets the number of iterations as a random number
    fn iterations(&self, _state: &mut S) -> Result<usize, libafl::Error> {
        Ok(1)
    }
}

impl<E, EM, I1, I2, M, S, Z> Named for SloptMutationalStage<E, EM, I1, I2, M, S, Z> {
    fn name(&self) -> &Cow<'static, str> {
        &self.name
    }
}

impl<E, EM, I1, I2, M, S, Z> Stage<E, EM, S, Z> for SloptMutationalStage<E, EM, I1, I2, M, S, Z>
where
    I1: Clone + MutatedTransform<I2, S>,
    I2: Input,
    M: Mutator<I1, S>,
    S: HasRand
        + HasCorpus<I2>
        + HasMetadata
        + HasExecutions
        + HasMetadata
        + HasCurrentCorpusId
        + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I2, S>,
{
    #[inline]
    fn perform(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), libafl::Error> {
        self.perform_mutational(fuzzer, executor, state, manager)
    }
}

impl<E, EM, I1, I2, M, S, Z> Restartable<S> for SloptMutationalStage<E, EM, I1, I2, M, S, Z>
where
    S: HasMetadata + HasNamedMetadata + HasCurrentCorpusId,
{
    fn should_restart(&mut self, state: &mut S) -> Result<bool, libafl::Error> {
        RetryCountRestartHelper::should_restart(state, &self.name, 3)
    }

    fn clear_progress(&mut self, state: &mut S) -> Result<(), libafl::Error> {
        RetryCountRestartHelper::clear_progress(state, &self.name)
    }
}

impl<E, EM, I, M, S, Z> SloptMutationalStage<E, EM, I, I, M, S, Z>
where
    M: Mutator<I, S>,
    I: MutatedTransform<I, S> + Input + Clone,
    S: HasCorpus<I> + HasRand + HasCurrentCorpusId + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I, S>,
{
    /// Creates a new default mutational stage
    pub fn new(mutator: M) -> Self {
        // Safe to unwrap: DEFAULT_MUTATIONAL_MAX_ITERATIONS is never 0.
        Self::transforming(mutator)
    }
}

impl<E, EM, I1, I2, M, S, Z> SloptMutationalStage<E, EM, I1, I2, M, S, Z>
where
    I1: MutatedTransform<I2, S> + Clone,
    I2: Input,
    M: Mutator<I1, S>,
    S: HasCorpus<I2> + HasRand + HasCurrentCorpusId + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I2, S>,
{
    #[inline]
    pub fn transforming(mutator: M) -> Self {
        let stage_id = unsafe {
            let ret = MUTATIONAL_STAGE_ID;
            MUTATIONAL_STAGE_ID += 1;
            ret
        };
        let name =
            Cow::Owned(MUTATIONAL_STAGE_NAME.to_owned() + ":" + stage_id.to_string().as_str());
        Self {
            name,
            mutator,
            phantom: PhantomData,
        }
    }
}

impl<E, EM, I1, I2, M, S, Z> SloptMutationalStage<E, EM, I1, I2, M, S, Z>
where
    I1: MutatedTransform<I2, S> + Clone,
    I2: Input,
    M: Mutator<I1, S>,
    S: HasRand + HasCurrentTestcase<I2> + MaybeHasClientPerfMonitor,
    Z: Evaluator<E, EM, I2, S>,
{
    /// Runs this (mutational) stage for the given testcase
    fn perform_mutational(
        &mut self,
        fuzzer: &mut Z,
        executor: &mut E,
        state: &mut S,
        manager: &mut EM,
    ) -> Result<(), libafl::Error> {
        let mut testcase = state.current_testcase_mut()?;

        let Ok(input) = I1::try_transform_from(&mut testcase, state) else {
            return Ok(());
        };
        drop(testcase);

        let mut input = input.clone();

        // Calls scheduled_mutate from Slopt mutator
        let _mutated = self.mutator_mut().mutate(state, &mut input)?;

        let (untransformed, post) = input.try_transform_into(state)?;
        let (_, corpus_id) = fuzzer.evaluate_filtered(state, executor, manager, &untransformed)?;

        self.mutator_mut().post_exec(state, corpus_id)?;
        post.post_exec(state, corpus_id)?;

        Ok(())
    }
}
