import torch

from .base import BaseTrainer


class StatefulTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._feature_storage = torch.zeros(
            self.data.num_nodes, self.num_classes, device=self.device
        )
        self._last_update = torch.zeros(
            self.data.num_nodes, dtype=torch.long, device=self.device
        )

    @property
    def feature_storage(self) -> torch.Tensor:
        return self._feature_storage

    @property
    def last_update(self) -> torch.Tensor:
        return self._last_update

    def reset_state(self):
        if self.warm_start_snapshot is not None:
            self._restore_state_from_snapshot(self.warm_start_snapshot)
        else:
            self.neighbor_loader.reset_state()
            self._feature_storage.zero_()
            self._last_update.zero_()

    def _restore_state_from_snapshot(self, snapshot: dict):
        self._feature_storage.copy_(snapshot["feature_storage"])
        self._last_update.copy_(snapshot["last_update"])
        self.neighbor_loader.load_state_dict(snapshot["neighbor_loader_state"])
