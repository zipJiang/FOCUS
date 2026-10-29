"""The MATH collector both trainer tutorials hand to their FOCUS integration."""
from focus import Collector, Critic, Selective

from math_quickstart import make_agent


def search_config():
    """A FIFO warm-up, then selective allocation by expected critic-error reduction."""
    return Selective(budget=32, warmup=16, max_concurrency=1)


def math_context(policy):
    """Give the critic the exact MATH prefix, including unfinished reasoning."""
    def render(rollout):
        if rollout.turns:
            last = rollout.turns[-1]
            tokens = last.prefix + last.tokens
        else:
            tokens = policy.prepare(rollout.messages).tokens
        return policy.decode(tokens)
    return render


def make_collector(policy, solution, *, critic=None, search=None, execution=None):
    if critic is None:
        # A uniform starting prediction for the first collection round.
        critic = Critic.constant(0.5, context=math_context(policy))
    return Collector(
        make_agent(policy, solution, policy.decode, execution=execution),
        critic=critic,
        search=search if search is not None else search_config(),
    )


def collector_for(policy, problem):
    """What both trainers ask for: the collector for one MATH problem under a policy.

    ``problem.prompt`` is the question the policy sees; ``problem.label`` is the
    reference solution, which stays inside the environment for grading.
    """
    return make_collector(policy, problem.label)
