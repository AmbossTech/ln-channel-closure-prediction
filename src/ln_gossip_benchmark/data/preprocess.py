from typing import Tuple

import numpy as np
import pandas as pd

# Mapping from gossip `closing_info` strings to integer labels.
# OPEN rows (channel still open at the time of the event) get class 0;
# closing rows get the actual closure type (PENALTY collapses into FORCED
# since they're functionally identical at the trainer level).
LABEL_MAP = {"OPEN": 0, "FORCED": 1, "PENALTY": 1, "MUTUAL": 2}

EDGE_FEATURE_COLS = [
    "status",
    "src_implementation",
    "dst_implementation",
    "capacity",
    "transaction_vout",
    "src_time_lock_delta",
    "dst_time_lock_delta",
    "src_min_htlc",
    "dst_min_htlc",
    "src_fee_base_msat",
    "dst_fee_base_msat",
    "src_fee_rate_milli_msat",
    "dst_fee_rate_milli_msat",
    "src_disabled",
    "dst_disabled",
    "src_last_update",
    "dst_last_update",
    "ts",
    "block_avg_fee_rate",
    "height",
]


def compute_event_labels(df: pd.DataFrame) -> np.ndarray:
    """Map each row's own `closing_info` to a 3-class label.

    The ground-truth oracle already builds its own forward-looking index
    from closing events, so we don't need a forward window here — at OPEN
    events the trainer ignores `data.y` and at closing events it just needs
    the row's own closure type, which is exactly what this returns.
    """
    return df["closing_info"].map(LABEL_MAP).fillna(0).astype(int).to_numpy()


def chronological_masks(
    timestamps: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split events chronologically using timestamp quantiles."""
    val_time, test_time = np.quantile(
        timestamps, [1 - val_ratio - test_ratio, 1 - test_ratio]
    )
    train_mask = timestamps <= val_time
    val_mask = (timestamps > val_time) & (timestamps <= test_time)
    test_mask = timestamps > test_time
    return train_mask, val_mask, test_mask
