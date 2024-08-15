import abc

from ellm.rm.ensemble import EnsembleModel


class RewardModel(abc.ABC):

    @abc.abstractmethod
    def get_duel_actions(self, features):
        """Get duel actions based on rewards of given features."""


class EnnDTS(RewardModel):
    """Double Thompson Sampling based on ensemble."""

    def __init__(self, model: EnsembleModel) -> None:
        super().__init__()
        self.model = model

    def get_duel_actions(self, features):
        return super().get_duel_actions(features)
