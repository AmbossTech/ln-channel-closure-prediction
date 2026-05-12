import torch
import torch.nn.functional as F

from ._layers import build_conv_stack, build_mlp_head, swap_directional_features
from .time_enc import TimeEncoder


class GNNPredictor(torch.nn.Module):
    """
    GNN-based predictor for edge classification.

    Uses GraphSAGE to aggregate neighborhood information before prediction.
    Configurable options:
    - use_node_history: Use accumulated node counts (temporal) vs just current degree (static)
    - use_edge_features: Include edge features in final prediction
    - use_time_encoding: Include temporal encodings
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        node_dim: int = None,
        edge_dim: int = None,
        num_layers: int = 2,
        time_dim: int = 32,
        use_node_history: bool = False,
        use_edge_features: bool = False,
        use_time_encoding: bool = False,
        use_gnn_embeddings: bool = True,
        mlp_hidden_dim: int = None,
        mlp_num_hidden_layers: int = 2,
        directional_feature_pairs: list = None,
    ):
        """
        Args:
            hidden_dim: Hidden layer dimension for GNN conv layers
            out_channels: Number of output classes
            node_dim: Dimension of node features (used if use_node_history=True)
            edge_dim: Dimension of edge features (used if use_edge_features=True)
            num_layers: Number of GNN conv layers
            time_dim: Dimension for time encoding
            use_node_history: Use accumulated node history (True) or just current degree (False)
            use_edge_features: Include edge features in prediction
            use_time_encoding: Include time encodings in prediction
            use_gnn_embeddings: Include GNN embeddings in prediction MLP (False = skip GNN entirely)
            mlp_hidden_dim: Hidden dim for prediction MLP (defaults to hidden_dim if None)
            mlp_num_hidden_layers: Number of hidden layers in prediction MLP
            directional_feature_pairs: Pairs of feature indices to swap for bidirectional edges
        """
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim

        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.time_dim = time_dim
        self.use_node_history = use_node_history
        self.use_edge_features = use_edge_features
        self.use_time_encoding = use_time_encoding
        self.use_gnn_embeddings = use_gnn_embeddings
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        # Convert to list of tuples if provided as list of lists (from yaml)
        if directional_feature_pairs:
            self.directional_feature_pairs = [
                tuple(p) for p in directional_feature_pairs
            ]
        else:
            self.directional_feature_pairs = []

        if self.use_edge_features and self.edge_dim is None:
            raise ValueError("edge_dim must be set when use_edge_features is True")

        if use_gnn_embeddings:
            gnn_input_dim = node_dim if use_node_history else 1
            self.convs = build_conv_stack(
                num_layers=num_layers,
                in_dim=gnn_input_dim,
                hidden_dim=hidden_dim,
                use_edge_features=self.use_edge_features,
                edge_dim=edge_dim,
            )
        else:
            self.convs = None

        # Time encoder
        if use_time_encoding:
            self.time_enc = TimeEncoder(time_dim)
            time_features_dim = 3 * time_dim
        else:
            self.time_enc = None
            time_features_dim = 0

        # Final MLP input: optional GNN emb + optional features
        mlp_input_dim = time_features_dim
        if use_gnn_embeddings:
            mlp_input_dim += 2 * hidden_dim  # src_emb + dst_emb
        if use_edge_features and edge_dim is not None:
            mlp_input_dim += edge_dim
        if use_node_history and node_dim is not None:
            mlp_input_dim += 2 * node_dim  # raw src + dst node features

        self.mlp = build_mlp_head(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            out_channels=out_channels,
            num_hidden_layers=mlp_num_hidden_layers,
        )

        # Store graph state
        self._edge_index = None
        self._node_features = None
        self._num_nodes = None
        self._edge_attr = None

    def set_graph_structure(
        self,
        edge_index: torch.Tensor,
        num_nodes: int = None,
        edge_attr: torch.Tensor = None,
    ):
        """Set the current graph structure (and optional edge features) for message passing.

        Note: Input edge_index may be unidirectional (e.g., only edges where src < dst).
        This method automatically makes edges bidirectional for proper GNN message passing.
        For edge features with directional properties (like Lightning Network's src/dst fees),
        the directional pairs are swapped when reversing edge direction.
        """
        # Make edges bidirectional for undirected graph message passing
        # SAGEConv/TransformerConv only pass messages in the direction of edges provided
        if edge_index is not None and edge_index.size(1) > 0:
            edge_index_reversed = edge_index.flip(0)
            self._edge_index = torch.cat([edge_index, edge_index_reversed], dim=1)

            # For reversed edges, swap directional feature pairs
            # e.g., src_fee <-> dst_fee, src_disabled <-> dst_disabled
            if edge_attr is not None:
                edge_attr_reversed = swap_directional_features(
                    edge_attr, self.directional_feature_pairs
                )
                self._edge_attr = torch.cat([edge_attr, edge_attr_reversed], dim=0)
            else:
                self._edge_attr = None
        else:
            self._edge_index = edge_index
            self._edge_attr = edge_attr

        if num_nodes is not None:
            self._num_nodes = num_nodes

        if not self.use_node_history:
            # Compute degrees for static mode
            if edge_index is not None and edge_index.size(1) > 0:
                if num_nodes is None:
                    num_nodes = edge_index.max().item() + 1
                self._num_nodes = num_nodes

                degrees = torch.zeros(num_nodes, device=edge_index.device)
                src, dst = edge_index[0], edge_index[1]
                degrees.scatter_add_(0, src, torch.ones_like(src, dtype=degrees.dtype))
                degrees.scatter_add_(0, dst, torch.ones_like(dst, dtype=degrees.dtype))
                self._node_features = (degrees / 2).unsqueeze(-1)  # [num_nodes, 1]
            else:
                self._node_features = None

    def compute_node_embeddings(self, node_features: torch.Tensor = None):
        """
        Compute node embeddings via GNN message passing.

        Args:
            node_features: [num_nodes, node_dim] - Node features (used if use_node_history=True)

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
        """
        if self.use_node_history:
            if node_features is None:
                return None
            x = node_features
        else:
            if self._node_features is None:
                return None
            x = torch.log1p(self._node_features)  # Normalize degrees

        if self._edge_index is None or self._edge_index.size(1) == 0:
            # No edges - apply layers without message passing
            for conv in self.convs:
                empty_edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
                x = conv(x, empty_edge_index, None)
                x = F.relu(x)
            return x

        if self.use_edge_features and self._edge_attr is None:
            raise ValueError(
                "edge_attr must be set (via set_graph_structure) when use_edge_features is True"
            )

        for conv in self.convs:
            x = conv(
                x, self._edge_index, self._edge_attr if self.use_edge_features else None
            )
            x = F.relu(x)

        return x

    def forward(
        self,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        **kwargs,
    ):
        """
        Args:
            src_indices: [num_edges] - Source node indices
            dst_indices: [num_edges] - Destination node indices
            **kwargs: Optional arguments:
                - edge_features: [num_edges, edge_dim] - (used upstream for set_graph_structure when edge features are enabled)
                - edge_age, src_recency, dst_recency: Time features (if use_time_encoding=True)
                - full_node_features: [num_nodes, node_dim] - Node features (if use_node_history=True)

        Returns:
            predictions: [num_edges, out_channels]
        """
        num_edges = src_indices.size(0)
        device = src_indices.device

        # Build combined features
        features = []

        # GNN embeddings (only if enabled)
        if self.use_gnn_embeddings:
            if self.use_node_history:
                full_node_features = kwargs.get("full_node_features")
                node_embeddings = self.compute_node_embeddings(full_node_features)
            else:
                node_embeddings = self.compute_node_embeddings()

            if node_embeddings is not None:
                assoc = kwargs.get("assoc")
                if assoc is not None:
                    src_emb = node_embeddings[assoc[src_indices]]
                    dst_emb = node_embeddings[assoc[dst_indices]]
                else:
                    src_emb = node_embeddings[src_indices]
                    dst_emb = node_embeddings[dst_indices]
            else:
                src_emb = torch.zeros(num_edges, self.hidden_dim, device=device)
                dst_emb = torch.zeros(num_edges, self.hidden_dim, device=device)
            features.extend([src_emb, dst_emb])

        if self.use_edge_features:
            edge_features = kwargs.get("edge_features")
            if edge_features is not None:
                features.append(edge_features)

        if self.use_node_history:
            src_nf = kwargs.get("src_node_features")
            dst_nf = kwargs.get("dst_node_features")
            if src_nf is not None and dst_nf is not None:
                features.extend([src_nf, dst_nf])

        if self.use_time_encoding:
            features.append(self.time_enc(torch.log1p(kwargs["edge_age"])))
            features.append(self.time_enc(torch.log1p(kwargs["src_recency"])))
            features.append(self.time_enc(torch.log1p(kwargs["dst_recency"])))

        combined = torch.cat(features, dim=-1)
        return self.mlp(combined)
