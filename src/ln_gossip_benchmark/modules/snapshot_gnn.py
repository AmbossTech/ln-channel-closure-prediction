import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv

from ._layers import build_mlp_head
from .time_enc import TimeEncoder


class SnapshotGNN(torch.nn.Module):
    """
    GNN trained on a single graph snapshot, using the same input features as
    the tabular baseline (XGBoost) plus graph aggregation.

    Inputs:
      - edge_index: [2, E] bidirectional edge index of the snapshot graph
      - node_features: [N, node_dim] per-node features (counts, degree, ratios)
      - src_indices, dst_indices: [Q] node ids for query edges
      - edge_features: [Q, edge_dim] per-edge raw features
      - edge_age, src_recency, dst_recency: [Q] scalar time features

    Architecture: stacked GNN conv layers (sage/gcn/transformer) operate on the
    node features (transformer also uses edge features as edge_attr in attention),
    then for each query edge we concatenate
        [gnn_src_emb, gnn_dst_emb, edge_features, raw_src_nf, raw_dst_nf,
         time_encoding(edge_age, src_recency, dst_recency)]
    and feed through a small MLP head.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        node_dim: int,
        edge_dim: int,
        num_layers: int = 2,
        conv_type: str = "sage",
        time_dim: int = 128,
        mlp_hidden_dim: int = None,
        mlp_num_hidden_layers: int = 1,
        dropout: float = 0.1,
        heads: int = 1,
    ):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type.lower()
        self.dropout = dropout
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.uses_edge_attr = self.conv_type == "transformer"

        self.convs = torch.nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            if self.conv_type == "sage":
                self.convs.append(SAGEConv(in_dim, hidden_dim))
            elif self.conv_type == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_dim))
            elif self.conv_type == "transformer":
                self.convs.append(
                    TransformerConv(
                        in_dim,
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        edge_dim=edge_dim,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            in_dim = hidden_dim

        self.time_enc = TimeEncoder(time_dim)
        # Head input = [gnn_src, gnn_dst, edge_feats, raw_src_nf, raw_dst_nf, time_enc × 3].
        self.mlp = build_mlp_head(
            input_dim=2 * hidden_dim + edge_dim + 2 * node_dim + 3 * time_dim,
            hidden_dim=mlp_hidden_dim,
            out_channels=out_channels,
            num_hidden_layers=mlp_num_hidden_layers,
            dropout=dropout,
        )

    def forward(
        self,
        edge_index,
        node_features,
        src_indices,
        dst_indices,
        edge_features,
        edge_age,
        src_recency,
        dst_recency,
        edge_attr=None,
    ):
        """
        Args:
            edge_index: [2, E] bidirectional edge index
            node_features: [N, node_dim]
            src_indices, dst_indices: [Q]
            edge_features: [Q, edge_dim]
            edge_age, src_recency, dst_recency: [Q]
            edge_attr: [E, edge_dim] per-edge attributes for the message-passing
                graph (used by TransformerConv as attention input). Caller is
                responsible for symmetrising it to match `edge_index`.
        Returns:
            logits: [Q, out_channels]
        """
        x = node_features
        for conv in self.convs:
            if self.uses_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        src_emb = x[src_indices]
        dst_emb = x[dst_indices]
        raw_src_nf = node_features[src_indices]
        raw_dst_nf = node_features[dst_indices]

        edge_age_enc = self.time_enc(torch.log1p(edge_age))
        src_rec_enc = self.time_enc(torch.log1p(src_recency))
        dst_rec_enc = self.time_enc(torch.log1p(dst_recency))

        feat = torch.cat(
            [
                src_emb,
                dst_emb,
                edge_features,
                raw_src_nf,
                raw_dst_nf,
                edge_age_enc,
                src_rec_enc,
                dst_rec_enc,
            ],
            dim=-1,
        )
        return self.mlp(feat)
