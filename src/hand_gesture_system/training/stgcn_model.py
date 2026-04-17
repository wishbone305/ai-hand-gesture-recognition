"""Spatial-Temporal Graph Convolutional Network (ST-GCN) for hand gesture recognition.

Architecture reference:
  Yan et al. 2018 — "Spatial Temporal Graph Convolutional Networks
  for Skeleton-Based Action Recognition"
  https://arxiv.org/abs/1801.07455

Key differences from TemporalGestureNet (the existing transformer model):
  - Treats the 21 hand landmarks as *graph nodes* connected by the hand skeleton
  - Spatial conv: message-passing along bone edges (graph convolution)
  - Temporal conv: dilated 1-D convolution over the time axis
  - No attention / no positional encoding — locality is modelled by the graph
  - Input layout: [B, C=3, T, V=21]  (C=xyz, T=frames, V=joints)
    vs. [B, T, F] in the transformer

Size presets
  small  — 32→64→128  channels, ~1 M params   (fast experiments)
  medium — 64→128→256 channels, ~5 M params   (default)
  large  — 128→256→512 channels, ~20 M params  (full training)
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

# ---------------------------------------------------------------------------
# Hand skeleton graph (MediaPipe 21-landmark layout)
# ---------------------------------------------------------------------------
NUM_JOINTS = 21

HAND_EDGES: list[tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),   (3, 4),   # thumb
    (0, 5),  (5, 6),  (6, 7),   (7, 8),   # index
    (0, 9),  (5, 9),  (9, 10),  (10, 11), (11, 12), # middle (+ direct wrist link)
    (0, 13), (9, 13), (13, 14), (14, 15), (15, 16), # ring   (+ direct wrist link)
    (0, 17), (13, 17),(17, 18), (18, 19), (19, 20), # pinky
]

STGCN_SIZE_PRESETS: dict[str, dict] = {
    # channels: feature channels per ST-GCN block (length = number of blocks)
    # downsample_at: block indices where temporal stride=2 is applied
    "small":  dict(channels=[32,  32,  32,  64,  64,  64,  128, 128, 128],
                   downsample_at=[3, 6], dropout=0.3),
    "medium": dict(channels=[64,  64,  64,  128, 128, 128, 256, 256, 256],
                   downsample_at=[3, 6], dropout=0.5),
    "large":  dict(channels=[128, 128, 128, 256, 256, 256, 512, 512, 512],
                   downsample_at=[3, 6], dropout=0.5),
    # jumbo: 10 blocks, ~52 M parameters — the "at least 50M" preset
    # Spatial: graph conv on 21-joint hand skeleton
    # Temporal: dilated 1-D conv, kernel=9, over the frame axis
    # Channels grow 256 → 512 → 1024 × 4 blocks
    "jumbo":  dict(channels=[256, 256, 256, 512, 512, 512, 1024, 1024, 1024, 1024],
                   downsample_at=[3, 6], dropout=0.5),
}


def _build_adjacency(num_joints: int, edges: list[tuple[int, int]]) -> "torch.Tensor":
    """Symmetric, degree-normalised adjacency matrix with self-loops."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)  # self-loops
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-8)))
    return torch.from_numpy(D_inv_sqrt @ A @ D_inv_sqrt)


if nn is not None:

    class _SpatialGCN(nn.Module):
        """Single spatial graph convolution: A × X × W."""

        def __init__(self, in_ch: int, out_ch: int, A: "torch.Tensor") -> None:
            super().__init__()
            self.register_buffer("A", A)            # [V, V]
            self.fc = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, C, T, V]
            # matmul along V axis: x @ A  →  [B, C, T, V]
            # faster than einsum on MPS/CUDA
            x = torch.matmul(x, self.A)
            return self.fc(x)                       # [B, C_out, T, V]

    class STGCNBlock(nn.Module):
        """One ST-GCN block: spatial GCN → BN/ReLU → temporal conv → BN + residual."""

        def __init__(
            self,
            in_ch: int,
            out_ch: int,
            A: "torch.Tensor",
            stride: int = 1,
            dropout: float = 0.0,
            temporal_kernel: int = 9,
        ) -> None:
            super().__init__()
            pad = (temporal_kernel - 1) // 2

            self.spatial = _SpatialGCN(in_ch, out_ch, A)
            self.bn1 = nn.BatchNorm2d(out_ch)

            self.temporal = nn.Sequential(
                nn.Conv2d(
                    out_ch, out_ch,
                    kernel_size=(temporal_kernel, 1),
                    padding=(pad, 0),
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_ch),
            )
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            if in_ch != out_ch or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.residual = nn.Identity()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            res = self.residual(x)
            x = self.act(self.bn1(self.spatial(x)))
            x = self.drop(self.temporal(x))
            return self.act(x + res)

    class STGCNGestureNet(nn.Module):
        """ST-GCN hand gesture classifier.

        Input  : [batch, 3, frames, 21]   — (x, y, z) per frame per joint
        Output : [batch, num_classes]      — logits

        The graph captures hand bone connectivity; spatial convolutions
        propagate information along edges (bones) while temporal convolutions
        model motion over time.  This is architecturally distinct from the
        transformer-based TemporalGestureNet — no attention mechanism is used.
        """

        def __init__(
            self,
            num_classes: int,
            in_channels: int = 3,
            model_size: str = "medium",
        ) -> None:
            super().__init__()

            if model_size not in STGCN_SIZE_PRESETS:
                raise ValueError(
                    f"Unknown model_size '{model_size}'. "
                    f"Choose from: {list(STGCN_SIZE_PRESETS)}"
                )
            cfg = STGCN_SIZE_PRESETS[model_size]
            channels: list[int] = cfg["channels"]
            dropout: float = cfg["dropout"]
            downsample_at: set[int] = set(cfg.get("downsample_at", [3, 6]))

            A = _build_adjacency(NUM_JOINTS, HAND_EDGES)

            # Input batch normalisation (applied to flattened joint×channel dim)
            self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

            # ST-GCN blocks — temporal resolution halved at blocks in downsample_at
            self.blocks = nn.ModuleList()
            prev = in_channels
            first_ds = min(downsample_at) if downsample_at else len(channels)
            for i, c in enumerate(channels):
                stride = 2 if i in downsample_at else 1
                blk_drop = dropout if i >= first_ds else 0.0
                self.blocks.append(STGCNBlock(prev, c, A, stride=stride, dropout=blk_drop))
                prev = c

            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(channels[-1], num_classes),
            )

            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: [B, C, T, V]
            B, C, T, V = x.shape

            # Normalise across joint×channel dim: reshape to [B, C*V, T]
            x_norm = x.permute(0, 3, 1, 2).contiguous().view(B, V * C, T)
            x_norm = self.data_bn(x_norm)
            x = x_norm.view(B, V, C, T).permute(0, 2, 3, 1).contiguous()

            for block in self.blocks:
                x = block(x)

            # Global average pool over time and joints → [B, C_last]
            x = x.mean(dim=(2, 3))
            return self.head(x)

    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

else:

    class STGCNGestureNet:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "PyTorch is required. Install with: pip install torch"
            )

    class STGCNBlock:  # pragma: no cover
        pass

    def count_parameters(model) -> int:  # pragma: no cover
        return 0


def build_stgcn(
    num_classes: int,
    in_channels: int = 3,
    model_size: str = "medium",
) -> "STGCNGestureNet":
    """Convenience factory — mirrors build_model() in model.py."""
    return STGCNGestureNet(
        num_classes=num_classes,
        in_channels=in_channels,
        model_size=model_size,
    )
