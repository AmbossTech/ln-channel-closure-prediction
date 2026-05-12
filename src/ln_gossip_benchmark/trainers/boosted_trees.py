import logging
from abc import abstractmethod

import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from ln_gossip_benchmark.utils.time import timestamp_ms_to_string

from .stateful import StatefulTrainer

log = logging.getLogger(__name__)


class _BoostedTreeTrainer(StatefulTrainer):
    library_name: str = "boosted-tree"

    def __init__(
        self,
        neighbor_loader,
        ground_truth_oracle,
        data,
        device: str,
        num_classes: int,
        cfg: omegaconf.DictConfig,
        run,
        train_end_snapshot: dict,
        train_loader,
        all_chan_ids,
        t_min,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        class_weights: list = None,
        enrich_node_features: bool = True,
        warm_start_snapshot: dict = None,
        **estimator_kwargs,
    ):
        super().__init__(
            neighbor_loader=neighbor_loader,
            ground_truth_oracle=ground_truth_oracle,
            data=data,
            device=device,
            num_classes=num_classes,
            cfg=cfg,
            run=run,
            train_end_snapshot=train_end_snapshot,
            train_loader=train_loader,
            all_chan_ids=all_chan_ids,
            t_min=t_min,
            warm_start_snapshot=warm_start_snapshot,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.class_weights = [float(w) for w in (class_weights or [1.0, 5.0, 5.0])]
        self.enrich_node_features = enrich_node_features
        self._estimator_kwargs = estimator_kwargs
        self.classifier = None

    @abstractmethod
    def _make_classifier(self): ...

    def _get_node_features(self, node_ids: torch.Tensor) -> torch.Tensor:
        counts = self._feature_storage[node_ids]
        if not self.enrich_node_features:
            return counts
        total_events = counts.sum(dim=-1, keepdim=True).clamp(min=1)
        total_closures = counts[:, 1:].sum(dim=-1, keepdim=True).clamp(min=1)
        forced_ratio = counts[:, 1:2] / total_closures
        mutual_ratio = counts[:, 2:3] / total_closures
        closure_ratio = counts[:, 1:].sum(dim=-1, keepdim=True) / total_events
        current_degree = self.neighbor_loader.get_node_degrees(node_ids).unsqueeze(-1)
        return torch.cat(
            [counts, current_degree, forced_ratio, mutual_ratio, closure_ratio], dim=-1
        )

    def _extract_features(
        self, current_t, edge_original_indices, open_src, open_dst
    ) -> np.ndarray:
        edge_features = self.data.msg[edge_original_indices].to(self.device)
        edge_timestamps = self.data.t[edge_original_indices].to(self.device)
        src_nf = torch.log1p(self._get_node_features(open_src))
        dst_nf = torch.log1p(self._get_node_features(open_dst))
        edge_age = torch.log1p((current_t - edge_timestamps).float()).unsqueeze(-1)
        src_recency = torch.log1p(
            (current_t - self._last_update[open_src]).float()
        ).unsqueeze(-1)
        dst_recency = torch.log1p(
            (current_t - self._last_update[open_dst]).float()
        ).unsqueeze(-1)
        return (
            torch.cat(
                [edge_features, src_nf, dst_nf, edge_age, src_recency, dst_recency],
                dim=-1,
            )
            .cpu()
            .numpy()
        )

    def train_loop(self, train_loader, val_loader):
        log.info(f"{self.library_name}: replaying training data to build state...")
        self.reset_state()
        global_step = 0
        pbar = tqdm(total=self.max_steps, desc=f"{self.library_name} - Replaying")

        for epoch in range(1, self.cfg.model.num_epochs + 1):
            for batch in train_loader:
                batch = batch.to(self.device)
                current_ts_ms = (batch.t.max().item() + self.t_min.item()) * 1000
                current_date = timestamp_ms_to_string(int(current_ts_ms))

                closing_edges = torch.atleast_1d(
                    (batch.edge_status == 0).nonzero().squeeze()
                )
                opening_edges = self._process_batch_insertions(batch)
                self._process_batch_removals(batch, closing_edges)

                self._log_step_metrics(
                    {
                        "loss": 0.0,
                        "date": current_date,
                        "time_ms": 0.0,
                        "num_open_edges": 0,
                        "num_active_nodes": 0,
                        "num_opening_batch": opening_edges.numel(),
                        "num_closing_batch": closing_edges.numel(),
                    },
                    global_step,
                    epoch,
                )
                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch, "date": current_date})
        pbar.close()

        log.info(
            f"{self.library_name}: collecting features from final training snapshot..."
        )
        _, edge_index_unique, edge_original_indices = self.neighbor_loader()
        if edge_index_unique.size(1) == 0:
            raise RuntimeError(
                f"No open edges at end of training — cannot fit {self.library_name}"
            )

        current_ts_ms = (self._last_update.max().item() + self.t_min.item()) * 1000
        ground_truth_labels, open_src, open_dst = self._get_ground_truth_for_open_edges(
            edge_original_indices, current_ts_ms
        )

        current_t = self.data.t[edge_original_indices].max()
        X_train = self._extract_features(
            current_t, edge_original_indices, open_src, open_dst
        )
        y_train = ground_truth_labels.cpu().numpy()
        sample_weight = np.array([self.class_weights[c] for c in y_train])

        log.info(
            f"{self.library_name}: collected {len(X_train)} samples, {X_train.shape[1]} features; "
            f"class distribution {np.bincount(y_train, minlength=self.num_classes)}"
        )
        log.info(
            f"{self.library_name}: fitting "
            f"(n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.learning_rate})..."
        )
        self.classifier = self._make_classifier()
        self.classifier.fit(X_train, y_train, sample_weight=sample_weight)
        log.info(f"{self.library_name}: fitting complete.")

    def _get_predictions(
        self,
        batch,
        edge_original_indices,
        ground_truth_labels,
        open_edges_original_src,
        open_edges_original_dst,
        node_ids_gnn=None,
        edge_index_unique=None,
    ) -> torch.Tensor:
        features = self._extract_features(
            batch.t.max(),
            edge_original_indices,
            open_edges_original_src,
            open_edges_original_dst,
        )
        return torch.tensor(
            self.classifier.predict(features), dtype=torch.long, device=self.device
        )


class XGBoostTrainer(_BoostedTreeTrainer):
    """XGBoost classifier on the end-of-training snapshot."""

    library_name = "XGBoost"

    def _make_classifier(self):
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="multi:softprob",
            num_class=self.num_classes,
            tree_method="hist",
            random_state=self.cfg.model.seed,
            n_jobs=-1,
        )


class LightGBMTrainer(_BoostedTreeTrainer):
    """LightGBM classifier on the end-of-training snapshot."""

    library_name = "LightGBM"

    def __init__(self, *args, num_leaves: int = 63, **kwargs):
        self.num_leaves = num_leaves
        super().__init__(*args, **kwargs)

    def _make_classifier(self):
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            objective="multiclass",
            num_class=self.num_classes,
            random_state=self.cfg.model.seed,
            n_jobs=-1,
            verbose=-1,
        )
