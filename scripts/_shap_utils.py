from typing import List

import numpy as np
import torch

EDGE_FEATURE_NAMES = [
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
RAW_NODE_FEATURE_NAMES = ["count_open", "count_forced", "count_mutual"]
ENRICHED_NODE_EXTRAS = [
    "current_degree",
    "forced_ratio",
    "mutual_ratio",
    "closure_ratio",
]


def feature_names_for_pipeline(enrich_node_features: bool) -> List[str]:
    node_names = list(RAW_NODE_FEATURE_NAMES)
    if enrich_node_features:
        node_names += ENRICHED_NODE_EXTRAS
    return (
        list(EDGE_FEATURE_NAMES)
        + [f"src_{n}" for n in node_names]
        + [f"dst_{n}" for n in node_names]
        + ["edge_age", "src_recency", "dst_recency"]
    )


class WrappedMLP(torch.nn.Module):
    """Predictor wrapper that exposes a single flat-tensor forward.

    SHAP's GradientExplainer wants a callable on a single input tensor; the
    predictor expects multiple kwargs. This unpacks `x` into the predictor's
    expected fields. `src_indices`/`dst_indices` are passed as zeros since
    they're only consulted when `use_gnn_embeddings=True`, which we don't
    use for the MLP we're explaining.
    """

    def __init__(self, predictor: torch.nn.Module, edge_dim: int, node_dim: int):
        super().__init__()
        self.predictor = predictor
        self.edge_dim = edge_dim
        self.node_dim = node_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ed, ns = self.edge_dim, self.node_dim
        edge_features = x[:, :ed]
        src_nf = x[:, ed : ed + ns]
        dst_nf = x[:, ed + ns : ed + 2 * ns]
        edge_age = x[:, ed + 2 * ns]
        src_recency = x[:, ed + 2 * ns + 1]
        dst_recency = x[:, ed + 2 * ns + 2]
        dummy = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.predictor(
            dummy,
            dummy,
            src_node_features=src_nf,
            dst_node_features=dst_nf,
            edge_features=edge_features,
            edge_age=edge_age,
            src_recency=src_recency,
            dst_recency=dst_recency,
        )


def compute_mean_abs_shap(
    wrapped: WrappedMLP,
    flat: torch.Tensor,
    n_background: int = 100,
    n_sample: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Run shap.GradientExplainer over a sample and return per-feature mean |SHAP|."""
    import shap

    rng = np.random.RandomState(seed)
    n_total = flat.size(0)
    bg_idx = rng.choice(n_total, size=min(n_background, n_total), replace=False)
    sample_idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)

    explainer = shap.GradientExplainer(wrapped, flat[bg_idx].detach())
    shap_values = explainer.shap_values(flat[sample_idx].detach())
    if isinstance(shap_values, list):
        shap_arr = np.stack([np.asarray(s) for s in shap_values], axis=0)
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr.transpose(2, 0, 1)
    # Mean over samples and classes -> per-feature importance.
    return np.mean(np.abs(shap_arr), axis=(0, 1))
