from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class TemporalGestureNet(nn.Module):
        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int = 192,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()

            if input_dim <= 0:
                raise ValueError(f"input_dim must be > 0, got {input_dim}")
            if num_classes <= 1:
                raise ValueError(f"num_classes must be > 1, got {num_classes}")
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
                )

            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, frames, features]
            if x.ndim != 3:
                raise RuntimeError("Expected input [batch, frames, features]")

            encoded = self.input_projection(x)
            encoded = self.encoder(encoded)
            pooled = encoded.mean(dim=1)
            return self.head(pooled)

else:

    class TemporalGestureNet:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for sequence training/export. "
                "Install with: pip install torch"
            )


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> TemporalGestureNet:
    return TemporalGestureNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
