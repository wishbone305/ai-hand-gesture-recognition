from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

# ---------------------------------------------------------------------------
# Size presets — each entry overrides hidden_dim / num_heads / num_layers /
# dropout when passed as model_size to build_model().
# ---------------------------------------------------------------------------
MODEL_SIZE_PRESETS: dict[str, dict] = {
    "small":  dict(hidden_dim=128, num_heads=4, num_layers=2, dropout=0.2),
    "medium": dict(hidden_dim=192, num_heads=4, num_layers=2, dropout=0.2),  # original default
    "large":  dict(hidden_dim=384, num_heads=6, num_layers=4, dropout=0.15),
    "xlarge": dict(hidden_dim=512, num_heads=8, num_layers=6, dropout=0.1),
}


if nn is not None:

    class LearnedPositionalEncoding(nn.Module):
        """Learnable positional embedding added to the projected input sequence."""

        def __init__(self, max_len: int, hidden_dim: int) -> None:
            super().__init__()
            self.pos_embed = nn.Embedding(max_len, hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, frames, hidden_dim]
            positions = torch.arange(x.size(1), device=x.device)
            return x + self.pos_embed(positions).unsqueeze(0)

    class TemporalGestureNet(nn.Module):
        """Transformer-based gesture sequence classifier.

        Architecture:
          1. Linear projection  → LayerNorm → GELU → Dropout
          2. Learned positional encoding
          3. TransformerEncoder (num_layers × num_heads)
          4. Mean-pool over frames
          5. MLP classification head
             - 2 layers  for small / medium
             - 3 layers with residual skip for large / xlarge
        """

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int = 192,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            max_seq_len: int = 512,
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

            self.pos_encoding = LearnedPositionalEncoding(max_seq_len, hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Deep residual head for large / xlarge; standard head otherwise
            self._deep_head = hidden_dim >= 384
            if self._deep_head:
                self.head_norm = nn.LayerNorm(hidden_dim)
                self.head_fc1 = nn.Linear(hidden_dim, hidden_dim)
                self.head_act1 = nn.GELU()
                self.head_drop1 = nn.Dropout(dropout)
                self.head_fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.head_act2 = nn.GELU()
                self.head_drop2 = nn.Dropout(dropout)
                self.head_out = nn.Linear(hidden_dim, num_classes)
            else:
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

            projected = self.input_projection(x)
            projected = self.pos_encoding(projected)
            encoded = self.encoder(projected)
            pooled = encoded.mean(dim=1)

            if self._deep_head:
                h = self.head_norm(pooled)
                residual = h
                h = self.head_drop1(self.head_act1(self.head_fc1(h)))
                h = self.head_drop2(self.head_act2(self.head_fc2(h)))
                h = h + residual  # skip connection
                return self.head_out(h)
            else:
                return self.head(pooled)

else:

    class TemporalGestureNet:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required for sequence training/export. "
                "Install with: pip install torch"
            )

    class LearnedPositionalEncoding:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required.")


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 192,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    max_seq_len: int = 512,
    model_size: str | None = None,
) -> "TemporalGestureNet":
    """Instantiate a TemporalGestureNet.

    Args:
        model_size: One of "small", "medium", "large", "xlarge".  When given,
            overrides hidden_dim / num_heads / num_layers / dropout with the
            preset values from MODEL_SIZE_PRESETS.
    """
    if model_size is not None:
        if model_size not in MODEL_SIZE_PRESETS:
            raise ValueError(
                f"Unknown model_size '{model_size}'. "
                f"Choose from: {list(MODEL_SIZE_PRESETS)}"
            )
        preset = MODEL_SIZE_PRESETS[model_size]
        hidden_dim = preset["hidden_dim"]
        num_heads = preset["num_heads"]
        num_layers = preset["num_layers"]
        dropout = preset["dropout"]

    return TemporalGestureNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=max_seq_len,
    )
