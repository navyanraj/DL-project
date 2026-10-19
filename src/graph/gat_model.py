"""
Graph Attention Network (GAT) model for lung disease classification.

Takes a graph of detected lesions (from build_graph.py) and classifies
the overall disease pattern using attention-weighted message passing.
Returns logits and attention weights for explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class LungGAT(nn.Module):
    """
    Graph Attention Network for lung disease classification.

    Architecture:
      Input (node features) → GATConv Layer 1 → GATConv Layer 2
      → Global Pooling → MLP Classifier → Disease Logits

    Attention weights from GAT layers are returned for explainability.
    """

    def __init__(
        self,
        input_dim: int = 22,
        hidden_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        num_classes: int = 15,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        """
        Args:
            input_dim:   Node feature dimension (22 from build_graph).
            hidden_dim:  Hidden dimension per attention head.
            num_heads:   Number of attention heads.
            num_layers:  Number of GAT layers (2-3 recommended).
            num_classes: Output classes (14 diseases + 1 normal = 15).
            dropout:     Dropout rate.
            pooling:     Graph-level pooling ('mean', 'max', or 'both').
        """
        super(LungGAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer: input_dim → hidden_dim * num_heads
        self.gat_layers.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,  # Output is hidden_dim * num_heads
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim * num_heads))

        # Final GAT layer: concat=False to reduce dimensions
        if num_layers >= 2:
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=False,  # Output is hidden_dim (averaged over heads)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # MLP classifier
        pool_dim = hidden_dim * 2 if pooling == "both" else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index, and batch.

        Returns:
            logits:           Tensor of shape (batch_size, num_classes).
            attention_weights: List of attention weight tensors from each GAT layer.
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        attention_weights = []

        # Pass through GAT layers
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x, (edge_idx, attn_w) = gat_layer(
                x, edge_index, return_attention_weights=True
            )
            attention_weights.append(attn_w)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        if self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_embedding = global_max_pool(x, batch)
        elif self.pooling == "both":
            graph_embedding = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
            )

        # Classification
        logits = self.classifier(graph_embedding)

        return logits, attention_weights


class LungGATWithEdgeFeatures(LungGAT):
    """
    Extended GAT that incorporates edge features (distance, IoU, same_lung).
    Projects edge features and adds them to attention computation.
    """

    def __init__(self, edge_dim: int = 3, **kwargs):
        super().__init__(**kwargs)
        # Edge feature projection
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, kwargs.get("hidden_dim", 64)),
            nn.ReLU(),
            nn.Linear(kwargs.get("hidden_dim", 64), kwargs.get("num_heads", 8)),
        )

    def forward(self, data):
        """Forward pass with edge features."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        attention_weights = []

        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x, (edge_idx, attn_w) = gat_layer(
                x, edge_index, return_attention_weights=True
            )
            attention_weights.append(attn_w)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == "mean":
            graph_embedding = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_embedding = global_max_pool(x, batch)
        elif self.pooling == "both":
            graph_embedding = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
            )

        logits = self.classifier(graph_embedding)
        return logits, attention_weights


def build_model(config: dict) -> LungGAT:
    """
    Build a LungGAT model from config dict.

    Expected config keys: input_dim, hidden_dim, num_heads, num_layers, num_classes, dropout.
    """
    model = LungGAT(
        input_dim=config.get("input_dim", 22),
        hidden_dim=config.get("hidden_dim", 64),
        num_heads=config.get("num_heads", 8),
        num_layers=config.get("num_layers", 2),
        num_classes=config.get("num_classes", 15),
        dropout=config.get("dropout", 0.3),
        pooling=config.get("pooling", "mean"),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LungGAT model created: {total_params:,} params ({trainable_params:,} trainable)")

    return model
