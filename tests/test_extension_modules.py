"""Custom components compose through their public, responsibility-specific modules."""

from focus import Collector, Critic, Phase, Search
from focus.branching import BranchPoints
from focus.context import AppendOnly, ContextManager
from focus.estimation import Estimator
from focus.priority import NodeInfo, Priority
from tests.helpers import ScriptedGenerator, ToyEnv, toy_runner


async def test_custom_extension_modules_collect_and_prepare_together():
    prepared = []
    eligible = []
    ranked = []
    estimated: list[int] = []

    class RememberContext(AppendOnly):
        def prepare(self, rs, agent, *, tools=None):
            prepared.append(rs.state.n)
            return super().prepare(rs, agent, tools=tools)

    class AfterFirstStep:
        def eligible(self, state):
            eligible.append(state.state.n)
            return state.state.n == 1

    class DeepestFirst(Priority):
        def __call__(self, node: NodeInfo) -> float:
            ranked.append(node.id)
            return float(node.depth)

    class ConstantAdvantage(Estimator):
        def advantages(self, edges, values):
            estimated.extend(edge.node.id for edge in edges)
            return [0.7 for _ in edges]

    context: ContextManager[object] = RememberContext()
    points: BranchPoints = AfterFirstStep()
    agent = toy_runner(
        ScriptedGenerator(["go"] * 8, logprobs=-0.5),
        env=ToyEnv(done_at=2),
        context=context,
        branch_points=points,
        max_steps=2,
        finish_prompt=None,
    )
    async with Collector(
        agent,
        search=Search([Phase(DeepestFirst(), 4, max_children=2)], max_concurrency=1),
        critic=Critic.constant(0.5),
        estimator=ConstantAdvantage(),
    ) as collector:
        tree = await collector.search("task")
        batch = collector.prepare(tree)

    assert prepared
    assert eligible
    assert ranked
    assert estimated
    assert batch.actor
    assert batch.critic
    assert all(sample.weight == 0.7 for sample in batch.actor)
    assert any(
        node.parent_id is not None and node.parent_id != tree.root_id
        for node in tree.nodes.values()
    )
    for node in tree.nodes.values():
        if node.id != tree.root_id and node.payload.forkable:
            assert node.payload.state.n == 1
    assert batch.to_records(0)
