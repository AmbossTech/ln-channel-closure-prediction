import logging
import time

import numpy as np
import omegaconf
import torch
import torch.nn as nn
from tqdm import tqdm

from ln_gossip_benchmark.utils.time import timestamp_ms_to_string

from ._class_freq import class_frequencies_from_oracle
from .stateful import StatefulTrainer

log = logging.getLogger(__name__)


class RandomTrainer(StatefulTrainer):
    """Baseline trainer that replays the temporal pipeline but predicts via
    a fixed strategy:

    - "uniform"    — uniform random class
    - "majority"   — always the most-frequent class in training
    - "stratified" — sample from training class frequencies
    """

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
        strategy: str = "uniform",
        warm_start_snapshot: dict = None,
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
        self.criterion = nn.CrossEntropyLoss()
        self.strategy = strategy
        self.class_frequencies = None
        self.majority_class = 0

        log.info(f"Baseline strategy: {strategy}")
        if strategy in ("majority", "stratified"):
            self._compute_class_frequencies(train_loader)

    def _compute_class_frequencies(self, train_loader):
        self.class_frequencies, self.majority_class, class_counts = (
            class_frequencies_from_oracle(
                train_loader=train_loader,
                neighbor_loader_factory=type(self.neighbor_loader),
                ground_truth_oracle=self.ground_truth_oracle,
                data=self.data,
                all_chan_ids=self.all_chan_ids,
                num_nodes=self.data.num_nodes,
                num_classes=self.num_classes,
                t_min=self.t_min,
                device=self.device,
            )
        )
        target_names = self.cfg.model.target_names
        log.info(f"Ground truth class counts: {class_counts.cpu().numpy()}")
        log.info(
            f"Ground truth class frequencies: {self.class_frequencies.cpu().numpy()}"
        )
        log.info(
            f"Majority class: {self.majority_class} ({target_names[self.majority_class]})"
        )

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
        n = len(ground_truth_labels)
        if self.strategy == "uniform":
            return torch.randint(0, self.num_classes, (n,), device=self.device)
        if self.strategy == "majority":
            return torch.full(
                (n,), self.majority_class, dtype=torch.long, device=self.device
            )
        if self.strategy == "stratified":
            return torch.multinomial(
                self.class_frequencies.expand(n, -1), num_samples=1
            ).squeeze(-1)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_random_logits(self, num_edges: int) -> torch.Tensor:
        if self.strategy == "uniform":
            return torch.randn(num_edges, self.num_classes, device=self.device)
        if self.strategy == "majority":
            logits = torch.zeros(num_edges, self.num_classes, device=self.device)
            logits[:, self.majority_class] = 10.0
            return logits
        if self.strategy == "stratified":
            log_freqs = torch.log(self.class_frequencies + 1e-8)
            return log_freqs.unsqueeze(0).expand(num_edges, -1)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def train_loop(self, train_loader, val_loader):
        log.info(
            f"Baseline ({self.strategy}): replaying training data to build state..."
        )
        self.reset_state()
        global_step = 0
        pbar = tqdm(
            total=self.max_steps, desc=f"Baseline ({self.strategy}) - Replaying"
        )

        for epoch in range(1, self.cfg.model.num_epochs + 1):
            epoch_step_metrics = []
            for batch in train_loader:
                batch = batch.to(self.device)
                t0 = time.time()
                current_ts_ms = (batch.t.max().item() + self.t_min.item()) * 1000
                current_date = timestamp_ms_to_string(int(current_ts_ms))

                closing_edges = torch.atleast_1d(
                    (batch.edge_status == 0).nonzero().squeeze()
                )
                opening_edges = self._process_batch_insertions(batch)

                node_ids_gnn, edge_index_unique, edge_original_indices = (
                    self.neighbor_loader()
                )
                num_open_edges = edge_original_indices.numel()
                num_active_nodes = node_ids_gnn.numel()

                step_classification_metrics = None
                step_loss = 0.0
                if num_active_nodes > 0 and num_open_edges > 0:
                    ground_truth_labels, _, _ = self._get_ground_truth_for_open_edges(
                        edge_original_indices, current_ts_ms
                    )
                    y_true = ground_truth_labels.cpu().numpy()
                    logits = self._get_random_logits(len(y_true))
                    step_loss = self.criterion(logits, ground_truth_labels).item()
                    y_pred = logits.argmax(dim=-1).cpu().numpy()
                    step_classification_metrics = self._compute_metrics(
                        y_true, y_pred, "train_step", log_confusion_matrix=False
                    )
                    epoch_step_metrics.append(
                        {"y_true": y_true, "y_pred": y_pred, "loss": step_loss}
                    )

                self._process_batch_removals(batch, closing_edges)

                self._log_step_metrics(
                    {
                        "loss": step_loss,
                        "date": current_date,
                        "time_ms": (time.time() - t0) * 1000,
                        "num_open_edges": num_open_edges,
                        "num_active_nodes": num_active_nodes,
                        "num_opening_batch": opening_edges.numel(),
                        "num_closing_batch": closing_edges.numel(),
                    },
                    global_step,
                    epoch,
                    extra_metrics=step_classification_metrics,
                )

                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch, "date": current_date})

                if global_step % self.eval_every_steps == 0:
                    val_step_metrics = self._run_step_validation(val_loader)
                    self.run.log(val_step_metrics, step=global_step)

            epoch_metrics = {"epoch": epoch}
            if epoch_step_metrics:
                all_y_true = np.concatenate([m["y_true"] for m in epoch_step_metrics])
                all_y_pred = np.concatenate([m["y_pred"] for m in epoch_step_metrics])
                train_epoch_metrics = self._compute_metrics(
                    all_y_true, all_y_pred, "train_epoch", log_confusion_matrix=True
                )
                train_epoch_metrics["train_epoch/loss"] = sum(
                    m["loss"] for m in epoch_step_metrics
                ) / len(epoch_step_metrics)
                epoch_metrics.update(train_epoch_metrics)

            epoch_metrics.update(self._run_epoch_evaluation("val_epoch"))
            self.run.log(epoch_metrics, step=global_step)

        pbar.close()
        log.info(f"Baseline ({self.strategy}): replay complete.")

    def _run_epoch_evaluation(self, metric_prefix: str) -> dict:
        node_ids_gnn, edge_index_unique, edge_original_indices = self.neighbor_loader()
        if node_ids_gnn.numel() == 0 or edge_index_unique.size(1) == 0:
            return {f"{metric_prefix}/accuracy": 0.0}

        current_ts_ms = (self._last_update.max().item() + self.t_min.item()) * 1000
        ground_truth_labels, _, _ = self._get_ground_truth_for_open_edges(
            edge_original_indices, current_ts_ms
        )
        y_true = ground_truth_labels.cpu().numpy()
        logits = self._get_random_logits(len(y_true))
        loss = self.criterion(logits, ground_truth_labels).item()
        y_pred = logits.argmax(dim=-1).cpu().numpy()

        log_cm = "epoch" in metric_prefix or metric_prefix == "test"
        metrics = self._compute_metrics(
            y_true, y_pred, metric_prefix, log_confusion_matrix=log_cm
        )
        metrics[f"{metric_prefix}/loss"] = loss
        return metrics
