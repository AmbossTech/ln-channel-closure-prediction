from typing import Iterable, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, TransformerConv


def build_mlp_head(
    input_dim: int,
    hidden_dim: int,
    out_channels: int,
    num_hidden_layers: int,
    dropout: float = 0.1,
) -> nn.Module:
    """Build the prediction head shared by all predictors.

    `num_hidden_layers == 0` collapses to a plain linear layer; each hidden
    layer is `Linear → ReLU → Dropout`.
    """
    if num_hidden_layers == 0:
        return nn.Linear(input_dim, out_channels)
    layers = []
    in_dim = input_dim
    for _ in range(num_hidden_layers):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, out_channels))
    return nn.Sequential(*layers)


def build_conv_stack(
    num_layers: int,
    in_dim: int,
    hidden_dim: int,
    *,
    use_edge_features: bool,
    edge_dim: int = None,
) -> nn.ModuleList:
    """Build a stack of GNN conv layers — TransformerConv if edges carry
    features (so attention can condition on them), otherwise SAGEConv.
    """
    convs = nn.ModuleList()
    if use_edge_features:
        convs.append(
            TransformerConv(
                in_dim, hidden_dim, heads=1, concat=False, edge_dim=edge_dim
            )
        )
        for _ in range(num_layers - 1):
            convs.append(
                TransformerConv(
                    hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=edge_dim
                )
            )
    else:
        convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden_dim, hidden_dim))
    return convs


def swap_directional_features(
    edge_attr: torch.Tensor,
    pairs: Iterable[Tuple[int, int]],
) -> torch.Tensor:
    """Swap (src_idx, dst_idx) feature pairs.

    Used when reversing a directed LN edge for bidirectional GNN message
    passing — pairs like (src_fee, dst_fee) need to flip so that the
    reversed edge's "src" sees what the forward edge's "dst" had.
    """
    if edge_attr.size(1) == 0:
        return edge_attr
    swapped = edge_attr.clone()
    for src_idx, dst_idx in pairs:
        if src_idx < edge_attr.size(1) and dst_idx < edge_attr.size(1):
            swapped[:, src_idx] = edge_attr[:, dst_idx]
            swapped[:, dst_idx] = edge_attr[:, src_idx]
    return swapped
