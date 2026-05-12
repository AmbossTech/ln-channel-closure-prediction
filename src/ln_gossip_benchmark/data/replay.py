import logging
from typing import Optional

import hydra
import torch

log = logging.getLogger(__name__)

# 2022-06-09 first day; 2022-06-10 00:00 UTC = first "real" day timestamp.
FIRST_REAL_DAY_TS = 1654819200


def build_train_end_snapshot(
    train_loader,
    neighbor_loader_conf,
    num_nodes: int,
    num_classes: int,
    device,
    initial_snapshot: Optional[dict] = None,
) -> dict:
    """Replay `train_loader` through a fresh neighbor loader.

    Returns the snapshot dict consumed by trainers (`feature_storage`,
    `last_update`, `neighbor_loader_state`). If `initial_snapshot` is set,
    starts from that state instead of zeros — used for warm-start where the
    first-day events are pre-loaded.
    """
    nl = hydra.utils.instantiate(
        neighbor_loader_conf, num_nodes=num_nodes, device=device
    )
    if initial_snapshot is not None:
        nl.load_state_dict(initial_snapshot["neighbor_loader_state"])
        feature_storage = initial_snapshot["feature_storage"].clone()
        last_update = initial_snapshot["last_update"].clone()
    else:
        feature_storage = torch.zeros(num_nodes, num_classes, device=device)
        last_update = torch.zeros(num_nodes, dtype=torch.long, device=device)

    one_hot_open = torch.zeros(num_classes, device=device)
    one_hot_open[0] = 1
    one_hot_mapping = torch.eye(num_classes, device=device)

    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            opening = torch.atleast_1d(batch.edge_status.nonzero().squeeze())
            closing = torch.atleast_1d((batch.edge_status == 0).nonzero().squeeze())

            if opening.numel() > 0:
                nl.insert(
                    batch.src[opening], batch.dst[opening], batch.global_id[opening]
                )
                feature_storage[batch.src[opening]] += one_hot_open
                feature_storage[batch.dst[opening]] += one_hot_open

            if closing.numel() > 0:
                nl.remove(batch.src[closing], batch.dst[closing])
                one_hot_y = one_hot_mapping[batch.y[closing]]
                feature_storage[batch.src[closing]] += one_hot_y
                feature_storage[batch.dst[closing]] += one_hot_y
                last_update[batch.src[closing]] = batch.t[closing]
                last_update[batch.dst[closing]] = batch.t[closing]

    return {
        "feature_storage": feature_storage,
        "last_update": last_update,
        "neighbor_loader_state": nl.state_dict(),
    }


def apply_first_day_filter(data, dataset, threshold_ts: int = FIRST_REAL_DAY_TS):
    """Remove all events strictly before `threshold_ts` from `data` and `dataset`.

    Returns the filtered `data`. Mutates `dataset._{train,val,test}_mask` and
    `dataset.dataset._full_data` in place to stay aligned with `data`.
    """
    valid_mask = data.t >= threshold_ts
    original_len = len(data.src)
    excluded = (~valid_mask).sum().item()

    data = data[valid_mask]

    dataset._train_mask = dataset._train_mask[valid_mask]
    dataset._val_mask = dataset._val_mask[valid_mask]
    dataset._test_mask = dataset._test_mask[valid_mask]

    valid_indices = valid_mask.cpu().numpy()
    for key in list(dataset.dataset._full_data.keys()):
        arr = dataset.dataset._full_data[key]
        if hasattr(arr, "__len__") and len(arr) == original_len:
            dataset.dataset._full_data[key] = arr[valid_indices]

    log.info(
        f"Excluded first day (initial snapshot): {excluded:,} events "
        f"({excluded / original_len * 100:.1f}%); remaining {len(data.src):,}"
    )
    return data


def apply_warm_start_mask(
    data, dataset, threshold_ts: int = FIRST_REAL_DAY_TS
) -> torch.Tensor:
    """Zero the train/val/test masks for first-day events but keep the data.

    Returns the boolean tensor identifying first-day events (used later to
    build the warm-start snapshot from those very events).
    """
    first_day_mask = data.t < threshold_ts
    dataset._train_mask = dataset._train_mask & ~first_day_mask
    dataset._val_mask = dataset._val_mask & ~first_day_mask
    dataset._test_mask = dataset._test_mask & ~first_day_mask
    log.info(
        f"Warm-start: {first_day_mask.sum().item():,} first-day events "
        "excluded from masks but kept for initialization"
    )
    return first_day_mask
