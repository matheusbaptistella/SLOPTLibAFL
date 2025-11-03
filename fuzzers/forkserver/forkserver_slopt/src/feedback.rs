use std::{borrow::Cow, marker::PhantomData};

use libafl::{
    HasMetadata,
    feedbacks::{Feedback, StateInitializer},
};
use libafl_bolts::Named;

use crate::slopt::{Bandit, Slopt};

pub struct SloptFeedback<F, SL> {
    inner: F,
    phantom: PhantomData<SL>,
}

impl<F, SL> SloptFeedback<F, SL> {
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<F: Named, SL> Named for SloptFeedback<F, SL> {
    fn name(&self) -> &Cow<'static, str> {
        self.inner.name()
    }
}

impl<S, F, SL> StateInitializer<S> for SloptFeedback<F, SL>
where
    S: HasMetadata,
    F: StateInitializer<S>,
{
    fn init_state(&mut self, state: &mut S) -> Result<(), libafl::Error> {
        self.inner.init_state(state)
    }
}

impl<EM, I, OT, S, F, SL> Feedback<EM, I, OT, S> for SloptFeedback<F, SL>
where
    F: Feedback<EM, I, OT, S>,
    S: HasMetadata,
    SL: Slopt,
{
    fn is_interesting(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        input: &I,
        observers: &OT,
        exit_kind: &libafl::executors::ExitKind,
    ) -> Result<bool, libafl::Error> {
        let interesting = self
            .inner
            .is_interesting(state, manager, input, observers, exit_kind)?;
        let slopt = state.metadata_mut::<SL>()?;
        let reward = if interesting { 1 } else { 0 } as u8;
        let choices = slopt.last_choices();

        slopt.op_bandit_mut().add_reward(choices.0, reward);
        slopt.bs_bandits_mut()[choices.1][choices.0].add_reward(choices.2, reward);

        Ok(interesting)
    }

    fn append_metadata(
        &mut self,
        state: &mut S,
        manager: &mut EM,
        observers: &OT,
        testcase: &mut libafl::corpus::Testcase<I>,
    ) -> Result<(), libafl::Error> {
        self
            .inner
            .append_metadata(state, manager, observers, testcase)
    }
}
