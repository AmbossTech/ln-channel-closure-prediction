import logging

import omegaconf
import torch
import torch.nn as nn
from tqdm import tqdm

from ln_gossip_benchmark.modules import TemporalModel
from ln_gossip_benchmark.modules.spectral import compute_spectral_features

from ._neural_eval import _NeuralEvalMixin
from ._neural_step import _NeuralStepMixin
from .base import BaseTrainer

log = logging.getLogger(__name__)


class Trainer(_NeuralStepMixin, _NeuralEvalMixin, BaseTrainer):
    """Backprop trainer for the MLP/GNN/spectral predictors."""

    def __init__(
        self,
        model: TemporalModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        neighbor_loader,
        ground_truth_oracle,
        data,
        assoc: torch.Tensor,
        device: str,
        num_classes: int,
        cfg: omegaconf.DictConfig,
        run,
        train_end_snapshot: dict,
        train_loader,
        all_chan_ids,
        t_min,
        scheduler,
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
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.assoc = assoc
        self.downsample_open_ratio = cfg.model.downsample_open_ratio
        self.ablation_disable_gnn = cfg.model.ablation_disable_gnn
        self.ablation_disable_raw_node_features = (
            cfg.model.ablation_disable_raw_node_features
        )
        self.spectral_dim = cfg.model.spectral_dim
        self.spectral_recompute_every = cfg.model.spectral_recompute_every
        self._spectral_step_count = 0
        self._spectral_buffer = (
            torch.zeros(data.num_nodes, self.spectral_dim, device=device)
            if self.spectral_dim > 0
            else None
        )

    @property
    def feature_storage(self) -> torch.Tensor:
        return self.model.feature_storage

    @property
    def last_update(self) -> torch.Tensor:
        return self.model.last_update

    def reset_state(self):
        if self.warm_start_snapshot is not None:
            self._restore_state_from_snapshot(self.warm_start_snapshot)
        else:
            self.model.reset_state()
            self.neighbor_loader.reset_state()
        self._spectral_step_count = 0
        if self._spectral_buffer is not None:
            self._spectral_buffer.zero_()

    def _restore_state_from_snapshot(self, snapshot: dict):
        self.model.feature_storage.copy_(snapshot["feature_storage"])
        self.model.last_update.copy_(snapshot["last_update"])
        self.neighbor_loader.load_state_dict(snapshot["neighbor_loader_state"])

    def _apply_ablation_flags(self, src_nf, dst_nf, full_nf):
        """Zero out feature groups per ablation config; preserves tensor dims."""
        if self.ablation_disable_gnn and full_nf is not None:
            full_nf = None
        if self.ablation_disable_raw_node_features:
            if src_nf is not None:
                src_nf = torch.zeros_like(src_nf)
            if dst_nf is not None:
                dst_nf = torch.zeros_like(dst_nf)
        return src_nf, dst_nf, full_nf

    def _get_spectral_kwargs(self, edge_index_unique, node_ids_gnn, open_src, open_dst):
        """Return src/dst spectral kwargs, recomputing eigvecs every N steps."""
        if self.spectral_dim == 0:
            return {}
        if self._spectral_step_count % self.spectral_recompute_every == 0:
            spectral_local = compute_spectral_features(
                edge_index_unique,
                node_ids_gnn.size(0),
                k=self.spectral_dim,
                device=self.device,
            )
            self._spectral_buffer[node_ids_gnn] = spectral_local
        self._spectral_step_count += 1
        return {
            "src_spectral": self._spectral_buffer[open_src],
            "dst_spectral": self._spectral_buffer[open_dst],
        }

    def _set_predictor_graph_structure(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor = None
    ):
        if hasattr(self.model.predictor, "set_graph_structure"):
            self.model.predictor.set_graph_structure(
                edge_index=edge_index,
                num_nodes=self.data.num_nodes,
                edge_attr=edge_attr,
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
        edge_messages_unique = self.data.msg[edge_original_indices].to(self.device)
        edge_timestamps = self.data.t[edge_original_indices].to(self.device)

        src_node_features = torch.log1p(
            self.model.get_node_features(open_edges_original_src, self.neighbor_loader)
        )
        dst_node_features = torch.log1p(
            self.model.get_node_features(open_edges_original_dst, self.neighbor_loader)
        )

        full_node_features = None
        if node_ids_gnn is not None:
            self.assoc[node_ids_gnn] = torch.arange(
                node_ids_gnn.size(0), device=self.device
            )
            full_node_features = torch.log1p(
                self.model.get_node_features(node_ids_gnn, self.neighbor_loader)
            )

        spectral_kwargs = {}
        if edge_index_unique is not None and node_ids_gnn is not None:
            spectral_kwargs = self._get_spectral_kwargs(
                edge_index_unique,
                node_ids_gnn,
                open_edges_original_src,
                open_edges_original_dst,
            )

        current_t = batch.t.max()
        edge_age = (current_t - edge_timestamps).float()
        src_recency = (
            current_t - self.model.last_update[open_edges_original_src]
        ).float()
        dst_recency = (
            current_t - self.model.last_update[open_edges_original_dst]
        ).float()

        abl_src_nf, abl_dst_nf, abl_full_nf = self._apply_ablation_flags(
            src_node_features,
            dst_node_features,
            full_node_features,
        )
        predictions = self.model.predictor(
            src_node_features=abl_src_nf,
            dst_node_features=abl_dst_nf,
            edge_features=edge_messages_unique,
            edge_age=edge_age,
            src_recency=src_recency,
            dst_recency=dst_recency,
            src_indices=open_edges_original_src,
            dst_indices=open_edges_original_dst,
            full_node_features=abl_full_nf,
            assoc=self.assoc,
            **spectral_kwargs,
        )
        return predictions.argmax(dim=-1)

    def train_loop(self, train_loader, val_loader):
        log.info(f"Starting training for {self.max_steps} steps...")
        pbar = tqdm(total=self.max_steps, desc="Training")
        global_step = 0
        epoch = 0
        while global_step < self.max_steps:
            epoch += 1
            global_step, epoch_step_losses = self._train_epoch(
                train_loader,
                val_loader,
                pbar,
                global_step,
                epoch,
            )
            epoch_train_metrics = self._get_epoch_train_metrics(
                metric_prefix="train_epoch"
            )
            epoch_train_metrics["train_epoch/loss"] = sum(epoch_step_losses) / max(
                len(epoch_step_losses), 1
            )
            epoch_val_metrics = self._run_evaluation(
                val_loader, "val", metric_prefix="val_epoch"
            )

            epoch_metrics = {"epoch": epoch}
            epoch_metrics.update(epoch_train_metrics)
            epoch_metrics.update(epoch_val_metrics)
            self.run.log(epoch_metrics, step=global_step)
        pbar.close()
        log.info("Training complete.")

    def _run_step_validation(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            return super()._run_step_validation(val_loader)

    def build_val_end_context(self, val_loader):
        log.info("Building Validation End Snapshot for Testing...")
        self._restore_state_from_snapshot(self.train_end_snapshot)
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Replaying Val Context"):
                batch = batch.to(self.device)
                closing_edges = torch.atleast_1d(
                    (batch.edge_status == 0).nonzero().squeeze()
                )
                self._process_batch_insertions(batch)
                self._process_batch_removals(batch, closing_edges)
