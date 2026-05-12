import torch

from ._layers import build_mlp_head
from .time_enc import TimeEncoder


class MLPPredictor(torch.nn.Module):
    """
    Flexible MLP-based predictor for edge classification.

    Combines functionality of EdgePredictor, NodeFeaturePredictor, and LogisticRegressionPredictor
    through configurable options:
    - use_edge_features: Include edge/message features
    - use_time_encoding: Include temporal encodings (edge_age, src/dst recency)
    - use_mlp: Use MLP (True) or single linear layer (False, = logistic regression)
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        out_channels: int,
        edge_dim: int = None,
        time_dim: int = 32,
        use_edge_features: bool = True,
        use_time_encoding: bool = True,
        use_mlp: bool = True,
        num_hidden_layers: int = None,
        spectral_dim: int = 0,
    ):
        """
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            out_channels: Number of output classes
            edge_dim: Dimension of edge features (required if use_edge_features=True)
            time_dim: Dimension for time encoding
            use_edge_features: Whether to include edge features
            use_time_encoding: Whether to include time encodings
            use_mlp: Deprecated, use num_hidden_layers instead. Kept for backward compat.
            num_hidden_layers: Number of hidden layers (0=linear, 1, 2, 3, ...). If None, inferred from use_mlp (True=2, False=0).
            spectral_dim: Number of spectral encoding dimensions per node (0 = disabled)
        """
        super().__init__()

        # Resolve num_hidden_layers from use_mlp for backward compatibility
        if num_hidden_layers is None:
            num_hidden_layers = 2 if use_mlp else 0

        self.out_channels = out_channels
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.use_edge_features = use_edge_features
        self.use_time_encoding = use_time_encoding
        self.num_hidden_layers = num_hidden_layers
        self.edge_dim = edge_dim if use_edge_features else 0
        self.spectral_dim = spectral_dim

        # Time encoder
        if use_time_encoding:
            self.time_enc = TimeEncoder(time_dim)
            time_features_dim = 3 * time_dim  # edge_age, src_recency, dst_recency
        else:
            self.time_enc = None
            time_features_dim = 0

        # Calculate input dimension
        input_dim = 2 * node_dim + time_features_dim
        if use_edge_features and edge_dim is not None:
            input_dim += edge_dim
        if spectral_dim > 0:
            input_dim += 2 * spectral_dim  # src + dst spectral encodings

        self.predictor = build_mlp_head(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(
        self,
        src_node_features: torch.Tensor,
        dst_node_features: torch.Tensor,
        **kwargs,
    ):
        """
        Args:
            src_node_features: [num_edges, node_dim] - Source node features
            dst_node_features: [num_edges, node_dim] - Destination node features
            **kwargs: Optional arguments:
                - edge_features: [num_edges, edge_dim] - Edge features (if use_edge_features=True)
                - edge_age, src_recency, dst_recency: Time features (if use_time_encoding=True)

        Returns:
            predictions: [num_edges, out_channels] - Class logits
        """
        features = [src_node_features, dst_node_features]

        if self.use_edge_features:
            features.insert(0, kwargs["edge_features"])

        if self.use_time_encoding:
            features.append(self.time_enc(torch.log1p(kwargs["edge_age"])))
            features.append(self.time_enc(torch.log1p(kwargs["src_recency"])))
            features.append(self.time_enc(torch.log1p(kwargs["dst_recency"])))

        if self.spectral_dim > 0:
            src_spectral = kwargs.get("src_spectral")
            dst_spectral = kwargs.get("dst_spectral")
            if src_spectral is not None and dst_spectral is not None:
                features.extend([src_spectral, dst_spectral])

        combined = torch.cat(features, dim=-1)
        return self.predictor(combined)
