import numpy as np
import pandas as pd
import torch


class GroundTruthOracle:
    """
    Manages ground truth information for open edges without time leakage.
    Computes the future closing status of specific dominant channels.
    """

    def __init__(self, data_df: pd.DataFrame, time_window_days: int = 180):
        """
        Initialize the oracle with preprocessed data.

        Args:
            data_df: DataFrame including 'src', 'dst', 'chan_id', etc.
            time_window_days: Time window in days to look for future closing events.
        """
        self.time_window = (
            time_window_days * 24 * 60 * 60 * 1000
        )  # Convert to milliseconds

        self._build_closing_index(data_df)

    def _build_closing_index(self, data_df: pd.DataFrame):
        """
        Build a dict index from closing events for O(1) lookup + binary search.
        Key: (src, dst, chan_id) -> (sorted gossip_ts array, label array)
        """
        closing_df = data_df[data_df["channel_status"] == 0][
            ["src", "dst", "gossip_ts", "edge_label", "chan_id"]
        ].copy()

        closing_df.sort_values(["src", "dst", "chan_id", "gossip_ts"], inplace=True)

        self._closing_index = {}
        for (src, dst, chan_id), group in closing_df.groupby(["src", "dst", "chan_id"]):
            ts_arr = group["gossip_ts"].values.astype(np.float64)
            label_arr = group["edge_label"].values.astype(np.int64)
            self._closing_index[(int(src), int(dst), chan_id)] = (ts_arr, label_arr)

    def get_ground_truth_for_open_edges(
        self,
        original_src_nodes: torch.Tensor,
        original_dst_nodes: torch.Tensor,
        edge_opening_timestamps: torch.Tensor,
        current_timestamp: float,
        edge_chan_ids: np.ndarray = None,
    ) -> torch.Tensor:
        """
        Get ground truth labels for specific open edges using strict ID matching.
        """
        num_queries = len(original_src_nodes)
        if num_queries == 0:
            return torch.empty(0, dtype=torch.long, device=original_src_nodes.device)

        if edge_chan_ids is None:
            raise ValueError("edge_chan_ids must be provided for strict matching.")

        src_np = original_src_nodes.cpu().numpy()
        dst_np = original_dst_nodes.cpu().numpy()
        open_ts_np = edge_opening_timestamps.cpu().numpy()

        horizon_end = current_timestamp + self.time_window
        final_labels = np.zeros(num_queries, dtype=np.int64)

        for i in range(num_queries):
            open_ts = open_ts_np[i]
            # Skip edges not yet active at current_timestamp
            if open_ts > current_timestamp:
                continue

            key = (int(src_np[i]), int(dst_np[i]), edge_chan_ids[i])
            entry = self._closing_index.get(key)
            if entry is None:
                continue

            ts_arr, label_arr = entry

            # Find the first closing event where gossip_ts > open_ts
            # using binary search (ts_arr is sorted)
            idx_start = np.searchsorted(ts_arr, open_ts, side="right")

            # Now scan from idx_start for the first event in [current_timestamp, horizon_end]
            for j in range(idx_start, len(ts_arr)):
                ts = ts_arr[j]
                if ts > horizon_end:
                    break
                if ts >= current_timestamp:
                    final_labels[i] = label_arr[j]
                    break

        return torch.tensor(
            final_labels, dtype=torch.long, device=original_src_nodes.device
        )
