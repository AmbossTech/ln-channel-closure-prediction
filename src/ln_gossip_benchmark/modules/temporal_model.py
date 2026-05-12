import torch
import torch.nn as nn


class TemporalModel(nn.Module):
    """Wraps a predictor with stateful per-node feature buffers.

    Buffers (registered, not trained):
      - `feature_storage` [num_nodes, num_classes] — running event counts
        per class (OPEN / FORCED / MUTUAL).
      - `last_update`     [num_nodes]              — last time `feature_storage`
        was touched for that node, in raw seconds.

    `get_node_features` exposes either the raw counts or an enriched feature
    vector (counts + current degree + 3 ratios), matching what the predictor
    consumes downstream.
    """

    def __init__(
        self,
        predictor: nn.Module,
        num_nodes: int,
        num_classes: int,
        enrich_node_features: bool = True,
    ):
        super().__init__()
        self.predictor = predictor
        self.num_classes = num_classes
        self.enrich_node_features = enrich_node_features
        self.register_buffer("feature_storage", torch.zeros(num_nodes, num_classes))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))

    def reset_state(self):
        self.feature_storage.zero_()
        self.last_update.zero_()

    def get_node_features(
        self, node_ids: torch.Tensor, neighbor_loader=None
    ) -> torch.Tensor:
        """Per-node feature vector for `node_ids`.

        Raw mode:    [count_OPEN, count_FORCED, count_MUTUAL].
        Enriched:    raw + [current_degree, forced_ratio, mutual_ratio, closure_ratio].
        """
        counts = self.feature_storage[node_ids]
        if not self.enrich_node_features:
            return counts

        total_events = counts.sum(dim=-1, keepdim=True).clamp(min=1)
        total_closures = counts[:, 1:].sum(dim=-1, keepdim=True).clamp(min=1)
        forced_ratio = counts[:, 1:2] / total_closures
        mutual_ratio = counts[:, 2:3] / total_closures
        closure_ratio = total_closures / total_events

        if neighbor_loader is not None:
            current_degree = neighbor_loader.get_node_degrees(node_ids).unsqueeze(-1)
        else:
            current_degree = counts[:, 0:1]
        return torch.cat(
            [counts, current_degree, forced_ratio, mutual_ratio, closure_ratio],
            dim=-1,
        )
