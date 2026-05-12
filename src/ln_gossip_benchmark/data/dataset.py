import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import TemporalData

from .preprocess import (
    EDGE_FEATURE_COLS,
    chronological_masks,
    compute_event_labels,
)

log = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    base = os.environ.get("LN_GOSSIP_CACHE_DIR")
    if base:
        return Path(base)
    return Path(os.path.expanduser("~/.cache/ln-gossip-benchmark"))


class LightningGossipDataset:
    """In-memory representation of the LN gossip CSV.

    Mirrors the surface of TGB's PyGLinkPropPredDataset (`get_TemporalData`,
    `train_mask`/`val_mask`/`test_mask`, nested `dataset._full_data` /
    `dataset.meta_dict`) so existing code can use it as a drop-in.
    """

    def __init__(
        self,
        name: str = "tgbl-ln",
        hf_repo_id: str = "Amboss-Tech/ln-gossip",
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        cache_dir: Optional[str] = None,
    ):
        self.name = name
        self.hf_repo_id = hf_repo_id
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self._fetch_csv()
        self._cache_path = self._cache_dir / f"{name}_processed.pkl"
        self.meta_dict = {"fname": str(self._csv_path)}

        cached = self._load_cache()
        if cached is None:
            cached = self._build_from_csv()
            self._save_cache(cached)
        self._init_from_dict(cached)

    # ----------------- Public, TGB-compatible surface -----------------

    @property
    def dataset(self):
        return self  # `self.dataset.foo` aliases `self.foo`

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    def get_TemporalData(self) -> TemporalData:
        return TemporalData(
            src=self._src,
            dst=self._dst,
            t=self._ts,
            msg=self._msg,
            y=self._edge_label,
            edge_status=self._edge_status,
        )

    @property
    def edge_label(self) -> torch.Tensor:
        return self._edge_label

    @property
    def train_mask(self) -> torch.Tensor:
        return self._train_mask

    @property
    def val_mask(self) -> torch.Tensor:
        return self._val_mask

    @property
    def test_mask(self) -> torch.Tensor:
        return self._test_mask

    # --------------------------- HF download ---------------------------

    def _fetch_csv(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required. Install with `uv add huggingface_hub`."
            ) from e
        log.info(f"Resolving gossip CSV from HF dataset '{self.hf_repo_id}' ...")
        local = hf_hub_download(
            repo_id=self.hf_repo_id,
            filename=f"{self.name}_edgelist.csv",
            repo_type="dataset",
        )
        return Path(local)

    # ------------------------ Build pipeline --------------------------

    def _build_from_csv(self) -> dict:
        log.info(f"Loading gossip CSV from {self._csv_path} ...")
        df = pd.read_csv(self._csv_path, true_values=["True"], false_values=["False"])

        unique_nodes = pd.concat([df["src"], df["dst"]]).unique()
        node_ids = {node: idx for idx, node in enumerate(unique_nodes)}
        df["u"] = df["src"].map(node_ids)
        df["i"] = df["dst"].map(node_ids)
        df["status"] = (df["channel_status"] == "OPEN").astype(int)

        edge_feat = df[EDGE_FEATURE_COLS].astype(float).to_numpy()
        sources = df["u"].to_numpy().astype(int)
        destinations = df["i"].to_numpy().astype(int)
        timestamps = (df["gossip_ts"] // 1000).to_numpy().astype(np.int64)
        edge_label = compute_event_labels(df).astype(np.int64)
        edge_status = df["status"].to_numpy().astype(np.int64)
        chan_id = df["chan_id"].to_numpy() if "chan_id" in df.columns else None

        train_mask, val_mask, test_mask = chronological_masks(
            timestamps,
            val_ratio=self._val_ratio,
            test_ratio=self._test_ratio,
        )

        return {
            "sources": sources,
            "destinations": destinations,
            "timestamps": timestamps,
            "edge_label": edge_label,
            "edge_status": edge_status,
            "edge_feat": edge_feat,
            "chan_id": chan_id,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "num_nodes": len(node_ids),
        }

    def _save_cache(self, payload: dict) -> None:
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Cached processed dataset at {self._cache_path}")
        except OSError as e:
            log.warning(f"Could not write dataset cache: {e}")

    def _load_cache(self) -> Optional[dict]:
        if not self._cache_path.exists():
            return None
        if self._cache_path.stat().st_mtime < self._csv_path.stat().st_mtime:
            log.info("CSV newer than cache; reprocessing.")
            return None
        try:
            with open(self._cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            log.warning(f"Cache unreadable ({e}); reprocessing.")
            return None

    def _init_from_dict(self, payload: dict) -> None:
        self._src = torch.from_numpy(payload["sources"]).long()
        self._dst = torch.from_numpy(payload["destinations"]).long()
        self._ts = torch.from_numpy(payload["timestamps"]).long()
        self._msg = torch.from_numpy(payload["edge_feat"]).float()
        self._edge_label = torch.from_numpy(payload["edge_label"]).long()
        self._edge_status = torch.from_numpy(payload["edge_status"]).long()
        self._train_mask = torch.from_numpy(payload["train_mask"])
        self._val_mask = torch.from_numpy(payload["val_mask"])
        self._test_mask = torch.from_numpy(payload["test_mask"])
        self._num_nodes = payload["num_nodes"]

        # Numpy-array view of the payload, kept for legacy `_full_data` access
        self._full_data = {
            "sources": payload["sources"],
            "destinations": payload["destinations"],
            "timestamps": payload["timestamps"],
            "edge_label": payload["edge_label"],
            "edge_status": payload["edge_status"],
            "chan_id": payload["chan_id"],
        }
