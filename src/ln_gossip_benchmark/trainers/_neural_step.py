import logging
import time

import torch

from ln_gossip_benchmark.utils.time import timestamp_ms_to_string

log = logging.getLogger(__name__)


class _NeuralStepMixin:
    def _train_step(self, batch, global_step_idx) -> dict:
        self.model.train()
        batch = batch.to(self.device)

        current_ts_ms = (batch.t.max().item() + self.t_min.item()) * 1000
        current_date = timestamp_ms_to_string(int(current_ts_ms))

        opening_edges = torch.atleast_1d(batch.edge_status.nonzero().squeeze())
        closing_edges = torch.atleast_1d((batch.edge_status == 0).nonzero().squeeze())
        num_opening = opening_edges.numel()
        num_closing = closing_edges.numel()

        if num_opening > 0:
            self.neighbor_loader.insert(
                batch.src[opening_edges],
                batch.dst[opening_edges],
                batch.global_id[opening_edges],
            )

        node_ids_gnn, edge_index_unique, edge_original_indices = self.neighbor_loader()
        num_open_edges = edge_original_indices.numel()
        num_active_nodes = node_ids_gnn.numel()

        step_loss = 0.0
        if num_active_nodes > 0 and num_open_edges > 0:
            self.optimizer.zero_grad()
            self.assoc[node_ids_gnn] = torch.arange(
                node_ids_gnn.size(0), device=self.device
            )

            edge_timestamps = self.data.t[edge_original_indices].to(self.device)
            edge_messages = self.data.msg[edge_original_indices].to(self.device)
            self._set_predictor_graph_structure(
                edge_index_unique, edge_attr=edge_messages
            )

            open_src = self.data.src[edge_original_indices].to(self.device)
            open_dst = self.data.dst[edge_original_indices].to(self.device)
            open_ts_ms = self.data.raw_t[edge_original_indices].to(self.device) * 1000
            chan_ids = self.all_chan_ids[edge_original_indices.cpu().numpy()]

            ground_truth_labels = (
                self.ground_truth_oracle.get_ground_truth_for_open_edges(
                    open_src,
                    open_dst,
                    open_ts_ms,
                    current_ts_ms,
                    edge_chan_ids=chan_ids,
                ).to(self.device)
            )

            src_node_features = torch.log1p(
                self.model.get_node_features(open_src, self.neighbor_loader)
            )
            dst_node_features = torch.log1p(
                self.model.get_node_features(open_dst, self.neighbor_loader)
            )
            full_node_features = torch.log1p(
                self.model.get_node_features(node_ids_gnn, self.neighbor_loader)
            )

            current_t = batch.t.max()
            edge_age = (current_t - edge_timestamps).float()
            src_recency = (current_t - self.model.last_update[open_src]).float()
            dst_recency = (current_t - self.model.last_update[open_dst]).float()

            spectral_kwargs = self._get_spectral_kwargs(
                edge_index_unique,
                node_ids_gnn,
                open_src,
                open_dst,
            )
            abl_src_nf, abl_dst_nf, abl_full_nf = self._apply_ablation_flags(
                src_node_features,
                dst_node_features,
                full_node_features,
            )
            predictions = self.model.predictor(
                src_node_features=abl_src_nf,
                dst_node_features=abl_dst_nf,
                edge_features=edge_messages,
                edge_age=edge_age,
                src_recency=src_recency,
                dst_recency=dst_recency,
                src_indices=open_src,
                dst_indices=open_dst,
                full_node_features=abl_full_nf,
                assoc=self.assoc,
                **spectral_kwargs,
            )

            # Optional class-imbalance downsampling: keep all closure edges plus
            # `downsample_open_ratio * n_close` randomly chosen OPEN edges.
            label_mask = None
            if self.downsample_open_ratio is not None:
                open_mask = ground_truth_labels == 0
                close_mask = ~open_mask
                n_close = close_mask.sum().item()
                if n_close > 0 and open_mask.sum().item() > 0:
                    n_open_keep = min(
                        open_mask.sum().item(),
                        max(1, int(self.downsample_open_ratio * n_close)),
                    )
                    open_indices = open_mask.nonzero(as_tuple=True)[0]
                    perm = torch.randperm(open_indices.size(0), device=self.device)[
                        :n_open_keep
                    ]
                    keep = torch.cat(
                        [open_indices[perm], close_mask.nonzero(as_tuple=True)[0]]
                    )
                    label_mask = torch.zeros_like(ground_truth_labels, dtype=torch.bool)
                    label_mask[keep] = True

            if label_mask is not None:
                loss = self.criterion(
                    predictions[label_mask], ground_truth_labels[label_mask]
                )
            else:
                loss = self.criterion(predictions, ground_truth_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            step_loss = loss.item()

        if num_closing > 0:
            self.neighbor_loader.remove(
                batch.src[closing_edges], batch.dst[closing_edges]
            )

        with torch.no_grad():
            if num_opening > 0:
                one_hot_open = torch.zeros(self.num_classes, device=self.device)
                one_hot_open[0] = 1
                self.model.feature_storage[batch.src[opening_edges]] += one_hot_open
                self.model.feature_storage[batch.dst[opening_edges]] += one_hot_open
            if num_closing > 0:
                one_hot_y = torch.eye(self.num_classes, device=self.device)[
                    batch.y[closing_edges]
                ]
                self.model.feature_storage[batch.src[closing_edges]] += one_hot_y
                self.model.feature_storage[batch.dst[closing_edges]] += one_hot_y
                self.model.last_update[batch.src[closing_edges]] = batch.t[
                    closing_edges
                ]
                self.model.last_update[batch.dst[closing_edges]] = batch.t[
                    closing_edges
                ]

        return {
            "loss": step_loss,
            "date": current_date,
            "num_open_edges": num_open_edges,
            "num_active_nodes": num_active_nodes,
            "num_opening_batch": num_opening,
            "num_closing_batch": num_closing,
        }

    def _train_epoch(
        self, train_loader, val_loader, pbar, global_step: int, epoch: int
    ):
        self.reset_state()
        epoch_step_losses = []
        for batch in train_loader:
            if global_step >= self.max_steps:
                break
            t0 = time.time()
            step_metrics = self._train_step(batch, global_step)
            step_metrics["time_ms"] = (time.time() - t0) * 1000

            step_loss = step_metrics["loss"]
            self._log_step_metrics(
                step_metrics,
                global_step,
                epoch,
                extra_metrics={"train_step/lr": self.optimizer.param_groups[0]["lr"]},
            )
            epoch_step_losses.append(step_loss)
            global_step += 1

            pbar.update(1)
            pbar.set_postfix(
                {
                    "epoch": epoch,
                    "step_loss": f"{step_loss:.4f}",
                    "date": step_metrics["date"],
                }
            )

            if global_step % self.eval_every_steps == 0:
                val_step_metrics = self._run_step_validation(val_loader)
                self.run.log(val_step_metrics, step=global_step)
                val_macro = val_step_metrics.get("val_step/macro avg/f1-score", 0.0)
                pbar.set_postfix({"epoch": epoch, "val_macro_f1": f"{val_macro:.4f}"})

        return global_step, epoch_step_losses
