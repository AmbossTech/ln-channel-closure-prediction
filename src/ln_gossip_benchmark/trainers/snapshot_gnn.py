import logging

import omegaconf
import torch
import torch.nn as nn
from tqdm import tqdm

from ln_gossip_benchmark.modules._layers import swap_directional_features
from ln_gossip_benchmark.modules.snapshot_gnn import SnapshotGNN
from ln_gossip_benchmark.utils.time import timestamp_ms_to_string

from .stateful import StatefulTrainer

log = logging.getLogger(__name__)


class SnapshotGNNTrainer(StatefulTrainer):
    """GNN trained on a single end-of-training graph snapshot.

    Replays training events to build the same per-node state the other
    trainers use, then takes a frozen snapshot of (graph, labels) and trains
    the GNN with backprop on that snapshot for `snapshot_epochs` rounds.
    Acts as the graph counterpart to the gradient-boosted-tree baselines.
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
        hidden_dim: int = 128,
        num_layers: int = 2,
        conv_type: str = "sage",
        mlp_num_hidden_layers: int = 1,
        mlp_hidden_dim: int = 128,
        snapshot_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        class_weights: list = None,
        directional_feature_pairs: list = None,
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.snapshot_epochs = snapshot_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = [float(w) for w in (class_weights or [1.0, 5.0, 5.0])]
        self.directional_feature_pairs = (
            [tuple(p) for p in directional_feature_pairs]
            if directional_feature_pairs
            else []
        )

        # Match the rest of the pipeline: 3 counts + degree + 3 ratios = 7 dims.
        self.enrich_node_features = True
        node_dim = num_classes + 4 if self.enrich_node_features else num_classes

        self.model = SnapshotGNN(
            hidden_dim=hidden_dim,
            out_channels=num_classes,
            node_dim=node_dim,
            edge_dim=data.msg.size(-1),
            num_layers=num_layers,
            conv_type=conv_type,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_hidden_layers=mlp_num_hidden_layers,
        ).to(device)

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

    def _build_snapshot_inputs(
        self,
        current_t,
        edge_index_unique,
        edge_original_indices,
        open_src,
        open_dst,
    ):
        edge_features = self.data.msg[edge_original_indices].to(self.device)
        edge_index_bidir = torch.cat(
            [edge_index_unique, edge_index_unique.flip(0)], dim=1
        )
        edge_attr_reversed = swap_directional_features(
            edge_features, self.directional_feature_pairs
        )
        edge_attr_bidir = torch.cat([edge_features, edge_attr_reversed], dim=0)

        all_node_ids = torch.arange(self.data.num_nodes, device=self.device)
        node_features = torch.log1p(self._get_node_features(all_node_ids))

        edge_timestamps = self.data.t[edge_original_indices].to(self.device)
        edge_age = (current_t - edge_timestamps).float()
        src_recency = (current_t - self._last_update[open_src]).float()
        dst_recency = (current_t - self._last_update[open_dst]).float()

        return (
            edge_index_bidir,
            node_features,
            edge_features,
            edge_attr_bidir,
            edge_age,
            src_recency,
            dst_recency,
        )

    def train_loop(self, train_loader, val_loader):
        log.info(
            f"Snapshot GNN ({self.conv_type}): replaying training data to build state..."
        )
        self.reset_state()
        global_step = 0
        pbar = tqdm(
            total=self.max_steps, desc=f"Snapshot GNN ({self.conv_type}) - Replaying"
        )

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

        log.info("Snapshot GNN: extracting end-of-training snapshot...")
        node_ids_gnn, edge_index_unique, edge_original_indices = self.neighbor_loader()
        if node_ids_gnn.numel() == 0 or edge_index_unique.size(1) == 0:
            raise RuntimeError(
                "No open edges at end of training — cannot fit snapshot GNN"
            )

        current_ts_ms = (self._last_update.max().item() + self.t_min.item()) * 1000
        ground_truth_labels, open_src, open_dst = self._get_ground_truth_for_open_edges(
            edge_original_indices,
            current_ts_ms,
        )

        current_t = self.data.t[edge_original_indices].max()
        ei_bidir, node_features, edge_features, ea_bidir, edge_age, src_rec, dst_rec = (
            self._build_snapshot_inputs(
                current_t,
                edge_index_unique,
                edge_original_indices,
                open_src,
                open_dst,
            )
        )

        log.info(
            f"Snapshot GNN: training set has {open_src.size(0)} edges, "
            f"class dist = {torch.bincount(ground_truth_labels, minlength=self.num_classes).tolist()}"
        )

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights, device=self.device)
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        log.info(
            f"Snapshot GNN: training {self.conv_type} for {self.snapshot_epochs} epochs..."
        )
        self.model.train()
        for epoch in range(self.snapshot_epochs):
            optimizer.zero_grad()
            logits = self.model(
                edge_index=ei_bidir,
                node_features=node_features,
                src_indices=open_src,
                dst_indices=open_dst,
                edge_features=edge_features,
                edge_age=edge_age,
                src_recency=src_rec,
                dst_recency=dst_rec,
                edge_attr=ea_bidir,
            )
            loss = criterion(logits, ground_truth_labels)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                with torch.no_grad():
                    acc = (
                        (logits.argmax(dim=-1) == ground_truth_labels)
                        .float()
                        .mean()
                        .item()
                    )
                log.info(
                    f"  epoch {epoch + 1}/{self.snapshot_epochs} loss={loss.item():.4f} train_acc={acc:.4f}"
                )

        log.info("Snapshot GNN: training done.")

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
        self.model.eval()
        with torch.no_grad():
            (
                ei_bidir,
                node_features,
                edge_features,
                ea_bidir,
                edge_age,
                src_rec,
                dst_rec,
            ) = self._build_snapshot_inputs(
                batch.t.max(),
                edge_index_unique,
                edge_original_indices,
                open_edges_original_src,
                open_edges_original_dst,
            )
            logits = self.model(
                edge_index=ei_bidir,
                node_features=node_features,
                src_indices=open_edges_original_src,
                dst_indices=open_edges_original_dst,
                edge_features=edge_features,
                edge_age=edge_age,
                src_recency=src_rec,
                dst_recency=dst_rec,
                edge_attr=ea_bidir,
            )
        return logits.argmax(dim=-1)
