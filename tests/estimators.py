"""Small custom estimators used to isolate export and integration tests."""

from focus.preparation import AdvantageEstimator
from focus.scheduler.core.returns import value_baseline


class FixedWeight(AdvantageEstimator):
    name = "test_fixed"

    def __init__(self, value=1.0):
        self.value = value

    def assign(self, records, *, view, reward_config):
        for record in records:
            record.weight = self.value


class RootReturn(AdvantageEstimator):
    name = "test_root_return"

    def assign(self, records, *, view, reward_config):
        for record in records:
            node = view.node(record.node_id)
            baseline = value_baseline(view.root, reward_config)
            record.weight = (
                reward_config.configured_return(
                    node.payload.turns, len(view.root.payload.turns)
                )
                - baseline
            )
            record.diagnostics = {"advantage": record.weight, "baseline": baseline}
