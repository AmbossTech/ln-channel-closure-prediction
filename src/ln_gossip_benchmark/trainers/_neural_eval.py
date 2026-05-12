import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class _NeuralEvalMixin:
    def _get_epoch_train_metrics(self, metric_prefix: str = "train_epoch") -> dict:
        self.model.eval()
        with torch.no_grad():
            node_ids_gnn, edge_index_unique, edge_original_indices = (
                self.neighbor_loader()
            )
            if node_ids_gnn.numel() == 0 or edge_index_unique.size(1) == 0:
                return {}

            current_ts_ms = (
                self.model.last_update.max().item() + self.t_min.item()
            ) * 1000
            ground_truth_labels, open_src, open_dst = (
                self._get_ground_truth_for_open_edges(
                    edge_original_indices,
                    current_ts_ms,
                )
            )

            edge_messages = self.data.msg[edge_original_indices].to(self.device)
            edge_timestamps = self.data.t[edge_original_indices].to(self.device)
            self._set_predictor_graph_structure(
                edge_index_unique, edge_attr=edge_messages
            )
            self.assoc[node_ids_gnn] = torch.arange(
                node_ids_gnn.size(0), device=self.device
            )

            src_nf = torch.log1p(
                self.model.get_node_features(open_src, self.neighbor_loader)
            )
            dst_nf = torch.log1p(
                self.model.get_node_features(open_dst, self.neighbor_loader)
            )
            full_nf = torch.log1p(
                self.model.get_node_features(node_ids_gnn, self.neighbor_loader)
            )

            current_t = self.model.last_update.max()
            edge_age = (current_t - edge_timestamps).float()
            src_recency = (current_t - self.model.last_update[open_src]).float()
            dst_recency = (current_t - self.model.last_update[open_dst]).float()

            spectral_kwargs = self._get_spectral_kwargs(
                edge_index_unique,
                node_ids_gnn,
                open_src,
                open_dst,
            )
            abl_src, abl_dst, abl_full = self._apply_ablation_flags(
                src_nf, dst_nf, full_nf
            )
            predictions = self.model.predictor(
                src_node_features=abl_src,
                dst_node_features=abl_dst,
                edge_features=edge_messages,
                edge_age=edge_age,
                src_recency=src_recency,
                dst_recency=dst_recency,
                src_indices=open_src,
                dst_indices=open_dst,
                full_node_features=abl_full,
                assoc=self.assoc,
                **spectral_kwargs,
            )
            y_true = ground_truth_labels.cpu().numpy()
            y_pred = predictions.argmax(dim=-1).cpu().numpy()
            if len(y_true) == 0:
                return {}
            return self._compute_metrics(
                y_true, y_pred, metric_prefix, log_confusion_matrix=True
            )

    def _run_evaluation(
        self,
        loader,
        split_mode: str,
        metric_prefix: str = None,
        save_predictions_path: str = None,
    ) -> dict:
        self.model.eval()
        if metric_prefix is None:
            metric_prefix = split_mode
        compute_loss = split_mode != "test"

        all_y_true, all_y_pred, all_edge_ages = [], [], []
        total_loss = 0.0
        num_prediction_batches = 0

        with torch.no_grad():
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
                    ground_truth_labels, open_src, open_dst = (
                        self._get_ground_truth_for_open_edges(
                            edge_original_indices,
                            current_ts_ms,
                        )
                    )
                    edge_messages = self.data.msg[edge_original_indices].to(self.device)
                    edge_timestamps = self.data.t[edge_original_indices].to(self.device)
                    self._set_predictor_graph_structure(
                        edge_index_unique, edge_attr=edge_messages
                    )
                    self.assoc[node_ids_gnn] = torch.arange(
                        node_ids_gnn.size(0), device=self.device
                    )

                    src_nf = torch.log1p(
                        self.model.get_node_features(open_src, self.neighbor_loader)
                    )
                    dst_nf = torch.log1p(
                        self.model.get_node_features(open_dst, self.neighbor_loader)
                    )
                    full_nf = torch.log1p(
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
                    abl_src, abl_dst, abl_full = self._apply_ablation_flags(
                        src_nf, dst_nf, full_nf
                    )
                    predictions = self.model.predictor(
                        src_node_features=abl_src,
                        dst_node_features=abl_dst,
                        edge_features=edge_messages,
                        edge_age=edge_age,
                        src_recency=src_recency,
                        dst_recency=dst_recency,
                        src_indices=open_src,
                        dst_indices=open_dst,
                        full_node_features=abl_full,
                        assoc=self.assoc,
                        **spectral_kwargs,
                    )

                    if compute_loss:
                        total_loss += self.criterion(
                            predictions, ground_truth_labels
                        ).item()
                        num_prediction_batches += 1
                    all_y_true.extend(ground_truth_labels.cpu().numpy())
                    all_y_pred.extend(predictions.argmax(dim=-1).cpu().numpy())
                    all_edge_ages.extend(edge_age.cpu().numpy())

                self._process_batch_removals(batch, closing_edges)

        if save_predictions_path and all_y_true:
            np.savez(
                save_predictions_path,
                y_true=np.array(all_y_true),
                y_pred=np.array(all_y_pred),
                edge_ages=np.array(all_edge_ages),
            )
            log.info(f"Saved predictions to {save_predictions_path}")

        if all_y_true and all_y_pred:
            log_cm = "epoch" in metric_prefix or metric_prefix == "test"
            perf_metrics = self._compute_metrics(
                np.array(all_y_true),
                np.array(all_y_pred),
                metric_prefix,
                log_confusion_matrix=log_cm,
            )
        else:
            perf_metrics = {f"{metric_prefix}/accuracy": 0.0}

        if compute_loss:
            perf_metrics[f"{metric_prefix}/loss"] = (
                total_loss / num_prediction_batches
                if num_prediction_batches > 0
                else 0.0
            )
        return perf_metrics
