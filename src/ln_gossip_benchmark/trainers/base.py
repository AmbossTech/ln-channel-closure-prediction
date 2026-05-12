import logging
from abc import ABC, abstractmethod

import numpy as np
import omegaconf
import torch
from tqdm import tqdm

from ._metrics import compute_classification_metrics, format_step_log

log = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer with the shared replay/evaluation pipeline."""

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
        warm_start_snapshot: dict = None,
    ):
        self.neighbor_loader = neighbor_loader
        self.ground_truth_oracle = ground_truth_oracle
        self.data = data
        self.device = device
        self.num_classes = num_classes
        self.cfg = cfg
        self.run = run
        self.train_end_snapshot = train_end_snapshot
        self.warm_start_snapshot = warm_start_snapshot
        self.all_chan_ids = all_chan_ids
        self.t_min = t_min

        self.max_steps = cfg.model.num_epochs * len(train_loader)
        self.eval_every_steps = cfg.model.eval_every_steps

    @property
    @abstractmethod
    def feature_storage(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def last_update(self) -> torch.Tensor: ...

    @abstractmethod
    def reset_state(self): ...

    @abstractmethod
    def _restore_state_from_snapshot(self, snapshot: dict): ...

    @abstractmethod
    def train_loop(self, train_loader, val_loader): ...

    @abstractmethod
    def _get_predictions(
        self,
        batch,
        edge_original_indices,
        ground_truth_labels,
        open_edges_original_src,
        open_edges_original_dst,
        node_ids_gnn=None,
        edge_index_unique=None,
    ) -> torch.Tensor: ...

    def _process_batch_insertions(self, batch):
        opening_edges = torch.atleast_1d(batch.edge_status.nonzero().squeeze())
        if opening_edges.numel() > 0:
            self.neighbor_loader.insert(
                batch.src[opening_edges],
                batch.dst[opening_edges],
                batch.global_id[opening_edges],
            )
            one_hot_open = torch.zeros(self.num_classes, device=self.device)
            one_hot_open[0] = 1
            self.feature_storage[batch.src[opening_edges]] += one_hot_open
            self.feature_storage[batch.dst[opening_edges]] += one_hot_open
        return opening_edges

    def _process_batch_removals(self, batch, closing_edges):
        if closing_edges.numel() == 0:
            return
        self.neighbor_loader.remove(batch.src[closing_edges], batch.dst[closing_edges])
        one_hot_y = torch.eye(self.num_classes, device=self.device)[
            batch.y[closing_edges]
        ]
        self.feature_storage[batch.src[closing_edges]] += one_hot_y
        self.feature_storage[batch.dst[closing_edges]] += one_hot_y
        self.last_update[batch.src[closing_edges]] = batch.t[closing_edges]
        self.last_update[batch.dst[closing_edges]] = batch.t[closing_edges]

    def _get_ground_truth_for_open_edges(
        self, edge_original_indices, current_timestamp_raw_ms
    ):
        open_src = self.data.src[edge_original_indices].to(self.device)
        open_dst = self.data.dst[edge_original_indices].to(self.device)
        open_ts_ms = self.data.raw_t[edge_original_indices].to(self.device) * 1000
        chan_ids = self.all_chan_ids[edge_original_indices.cpu().numpy()]
        labels = self.ground_truth_oracle.get_ground_truth_for_open_edges(
            open_src,
            open_dst,
            open_ts_ms,
            current_timestamp_raw_ms,
            edge_chan_ids=chan_ids,
        ).to(self.device)
        return labels, open_src, open_dst

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_prefix: str,
        log_confusion_matrix: bool = False,
    ) -> dict:
        return compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            num_classes=self.num_classes,
            metric_prefix=metric_prefix,
            target_names=self.cfg.model.target_names,
            log_confusion_matrix=log_confusion_matrix,
        )

    def _log_step_metrics(
        self,
        step_metrics: dict,
        global_step: int,
        epoch: int,
        extra_metrics: dict = None,
    ):
        self.run.log(
            format_step_log(step_metrics, global_step, epoch, extra_metrics),
            step=global_step,
        )

    def _run_step_validation(self, val_loader):
        """Run validation against the train-end snapshot, restoring state afterwards."""
        state_backup = {
            "feature_storage": self.feature_storage.clone(),
            "last_update": self.last_update.clone(),
        }
        neighbor_loader_state = self.neighbor_loader.state_dict()

        self._restore_state_from_snapshot(self.train_end_snapshot)
        val_metrics = self._run_evaluation(val_loader, "val", metric_prefix="val_step")

        self.feature_storage.copy_(state_backup["feature_storage"])
        self.last_update.copy_(state_backup["last_update"])
        self.neighbor_loader.load_state_dict(neighbor_loader_state)
        return val_metrics

    def build_val_end_context(self, val_loader):
        """Replay validation events from the train-end snapshot."""
        log.info("Building Validation End Snapshot for Testing...")
        self._restore_state_from_snapshot(self.train_end_snapshot)
        for batch in tqdm(val_loader, desc="Replaying Val Context"):
            batch = batch.to(self.device)
            closing_edges = torch.atleast_1d(
                (batch.edge_status == 0).nonzero().squeeze()
            )
            self._process_batch_insertions(batch)
            self._process_batch_removals(batch, closing_edges)

    def _run_evaluation(
        self,
        loader,
        split_mode: str,
        metric_prefix: str = None,
        save_predictions_path: str = None,
    ) -> dict:
        if metric_prefix is None:
            metric_prefix = split_mode

        all_y_true = []
        all_y_pred = []
        for batch in loader:
            batch = batch.to(self.device)
            current_ts_ms = (batch.t.max().item() + self.t_min.item()) * 1000
            closing_edges = torch.atleast_1d(
                (batch.edge_status == 0).nonzero().squeeze()
            )
            self._process_batch_insertions(batch)

            node_ids_gnn, edge_index_unique, edge_original_indices = (
                self.neighbor_loader()
            )
            if node_ids_gnn.numel() > 0 and edge_index_unique.size(1) > 0:
                edge_messages_unique = self.data.msg[edge_original_indices].to(
                    self.device
                )
                if hasattr(self, "_set_predictor_graph_structure"):
                    self._set_predictor_graph_structure(
                        edge_index_unique, edge_attr=edge_messages_unique
                    )
                ground_truth_labels, open_src, open_dst = (
                    self._get_ground_truth_for_open_edges(
                        edge_original_indices,
                        current_ts_ms,
                    )
                )
                predictions = self._get_predictions(
                    batch,
                    edge_original_indices,
                    ground_truth_labels,
                    open_src,
                    open_dst,
                    node_ids_gnn=node_ids_gnn,
                    edge_index_unique=edge_index_unique,
                )
                all_y_true.extend(ground_truth_labels.cpu().numpy())
                all_y_pred.extend(predictions.cpu().numpy())

            self._process_batch_removals(batch, closing_edges)

        if save_predictions_path and all_y_true:
            np.savez(
                save_predictions_path,
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
            )
            log.info(f"Saved predictions to {save_predictions_path}")

        if not (all_y_true and all_y_pred):
            return {f"{metric_prefix}/accuracy": 0.0}
        # Confusion matrices are heavy; only log them for epoch-level or test metrics.
        log_cm = "epoch" in metric_prefix or metric_prefix == "test"
        return self._compute_metrics(
            np.array(all_y_true),
            np.array(all_y_pred),
            metric_prefix,
            log_confusion_matrix=log_cm,
        )

    def test(self, test_loader, val_loader, save_predictions_path: str = None):
        log.info("Running final test evaluation...")
        self.build_val_end_context(val_loader)
        perf_metrics_test = self._run_evaluation(
            test_loader,
            "test",
            metric_prefix="test",
            save_predictions_path=save_predictions_path,
        )
        self.run.log(perf_metrics_test, step=self.max_steps)
        log.info(f"Test metrics: {perf_metrics_test}")
