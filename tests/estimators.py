"""Small custom estimators used to isolate export and integration tests."""

from focus.estimation import Advantage, Estimator


class FixedWeight(Estimator):
    name = "test_fixed"

    def __init__(self, value=1.0):
        self.value = value

    def advantages(self, edges, values):
        return [self.value for _ in edges]


class RootReturn(Estimator):
    """The whole return from the root's prefix, against the root's critic estimate."""

    name = "test_root_return"

    def advantages(self, edges, values):
        root = values.tree.root
        baseline = 0.0 if root.critic is None else root.critic
        result = []
        for edge in edges:
            weight = (
                values.reward.configured_return(
                    edge.node.payload.turns, len(root.payload.turns)
                )
                - baseline
            )
            result.append(
                Advantage(weight, {"advantage": weight, "baseline": baseline})
            )
        return result
