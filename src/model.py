"""
src/model.py

Spatial-Temporal Transformer (ST-Transformer) for AirSketch.

Adapted from:
    Plizzari, C., Cannici, M., & Matteucci, M. (2021).
    Spatial Temporal Transformer Network for Skeleton-Based Action Recognition.
    ICPR 2021.

Modified for:
    - Fingertip trajectory regression (not action classification)
    - 21-joint hand skeleton (not full-body pose)
    - Joint training with a binary gesture classifier

Input contract (from issue #6 AirSketchDataset):
    sequence:  (B, T=16, 42)   float32  normalized landmark coords
    → internally reshaped to (B, T, 21, 2) then projected to (B, T, 21, d)

Output contract (consumed by issue #10 training loop):
    pred_xy:   (B, 2)          float32  index fingertip (x, y) at T+1
    gesture_logits: (B, 2)     float32  raw logits for {idle=0, draw=1}
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# ── Config dataclass ───────────────────────────────────────────────────────────


@dataclass
class STTransformerConfig:
    """
    All model hyperparameters in one place.
    Matches the keys in configs/default.yaml model block.

    Args:
        hidden_dim:          Feature dimension d throughout the model.
                             Must be divisible by num_heads.
        num_layers:          Number of stacked ST-Blocks.
        num_heads:           Number of attention heads in both spatial and
                             temporal self-attention. hidden_dim // num_heads
                             must be >= 8 for reasonable head dimension.
        dropout:             Dropout probability applied after attention
                             and FFN layers.
        window_size:         T — number of input frames (default: 16).
        num_landmarks:       Number of hand joints (default: 21).
        input_dim:           Features per frame = num_landmarks * 2 (default: 42).
        gesture_loss_weight: Lambda for gesture loss in joint training.
                             Stored here for reference; used in train.py.
    """

    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    window_size: int = 16
    num_landmarks: int = 21
    input_dim: int = 42  # num_landmarks * 2
    gesture_loss_weight: float = 0.5

    def __post_init__(self):
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        assert self.input_dim == self.num_landmarks * 2, (
            f"input_dim ({self.input_dim}) must equal num_landmarks*2 "
            f"({self.num_landmarks * 2})"
        )

    @classmethod
    def from_yaml(cls, config: dict) -> "STTransformerConfig":
        """Build config from the 'model' block of default.yaml."""
        m = config.get("model", {})
        return cls(
            hidden_dim=m.get("hidden_dim", 128),
            num_layers=m.get("num_layers", 4),
            num_heads=m.get("num_heads", 4),
            dropout=m.get("dropout", 0.1),
            window_size=config.get("data", {}).get("window_size", 16),
            gesture_loss_weight=m.get("gesture_loss_weight", 0.5),
        )


# ── Positional encoding ────────────────────────────────────────────────────────


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding applied along the temporal dimension.

    Added to the temporal sequence before temporal self-attention so the model
    can distinguish frame 1 from frame 16. Not applied to the spatial dimension
    since joint indices are already encoded via the input projection weights.

    Args:
        d_model:  Feature dimension.
        max_len:  Maximum sequence length (default: 512, covers T=16 easily).
        dropout:  Dropout applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x + positional encoding, same shape.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Feed-forward network ───────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network: Linear → GELU → Dropout → Linear.
    Applied after each attention sub-layer in an ST-Block.

    The inner dimension is 4 × d_model following the original transformer
    paper (Vaswani et al., 2017).

    Args:
        d_model:  Input and output feature dimension.
        dropout:  Dropout probability.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Spatial self-attention ─────────────────────────────────────────────────────


class SpatialSelfAttention(nn.Module):
    """
    Multi-head self-attention applied across the 21 joints within each frame.

    Operates on the spatial dimension (J=21) independently for each
    (batch, time) position. Captures intra-frame joint relationships:
    how the wrist position constrains the knuckle, how the knuckle
    constrains the fingertip.

    Input is reshaped from (B, T, J, d) → (B*T, J, d) so that standard
    nn.MultiheadAttention can process all frames in one batch dimension.

    Args:
        d_model:   Feature dimension per joint.
        num_heads: Number of attention heads.
        dropout:   Dropout on attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, d)
        Returns:
            (B, T, J, d) — spatial attention applied within each frame.
        """
        B, T, J, d = x.shape

        # Merge batch and time → treat each frame independently
        x_flat = x.reshape(B * T, J, d)  # (B*T, J, d)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.dropout(attn_out)

        # Residual + LayerNorm
        out = self.norm(x_flat + attn_out)  # (B*T, J, d)

        return out.reshape(B, T, J, d)  # (B, T, J, d)


# ── Temporal self-attention ────────────────────────────────────────────────────


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention applied across the T=16 frames for each joint.

    Operates on the temporal dimension (T) independently for each
    (batch, joint) position. Captures motion patterns over time:
    the trajectory of a joint across the window. This is the component
    that learns to distinguish deliberate drawing motion from idle tremor.

    Input is reshaped from (B, T, J, d) → (B*J, T, d) so that standard
    nn.MultiheadAttention processes all joints in one batch dimension.

    Args:
        d_model:   Feature dimension per joint.
        num_heads: Number of attention heads.
        dropout:   Dropout on attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, d)
        Returns:
            (B, T, J, d) — temporal attention applied across frames per joint.
        """
        B, T, J, d = x.shape

        # Permute to (B, J, T, d) then merge batch and joint dimensions
        x_perm = x.permute(0, 2, 1, 3)  # (B, J, T, d)
        x_flat = x_perm.reshape(B * J, T, d)  # (B*J, T, d)

        # Add temporal positional encoding
        x_flat = self.pos_enc(x_flat)

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = self.dropout(attn_out)

        out = self.norm(x_flat + attn_out)  # (B*J, T, d)

        # Reshape back
        out = out.reshape(B, J, T, d)  # (B, J, T, d)
        return out.permute(0, 2, 1, 3)  # (B, T, J, d)


# ── ST-Block ───────────────────────────────────────────────────────────────────


class STBlock(nn.Module):
    """
    One Spatial-Temporal Block: spatial attention → temporal attention → FFN.

    Each sub-layer uses a pre-norm residual connection (LayerNorm before
    attention/FFN rather than after). Pre-norm is more stable during early
    training and is the de facto standard in modern transformer implementations.

    Args:
        d_model:   Feature dimension.
        num_heads: Number of attention heads.
        dropout:   Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_attn = SpatialSelfAttention(d_model, num_heads, dropout)

        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attn = TemporalSelfAttention(d_model, num_heads, dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, d)
        Returns:
            (B, T, J, d)
        """
        # Spatial attention with pre-norm residual
        x = x + self.spatial_attn(self.spatial_norm(x))

        # Temporal attention with pre-norm residual
        x = x + self.temporal_attn(self.temporal_norm(x))

        # FFN with pre-norm residual (apply over last dim, broadcast over B,T,J)
        x = x + self.ffn(self.ffn_norm(x))

        return x


# ── Regression head ────────────────────────────────────────────────────────────


class RegressionHead(nn.Module):
    """
    2-layer MLP that maps the pooled sequence representation to
    the predicted index fingertip (x, y) coordinate at frame T+1.

    Output is passed through sigmoid to constrain predictions to [0, 1],
    matching the normalized coordinate space of the training labels.

    Architecture: Linear(d → d//2) → GELU → Dropout → Linear(d//2 → 2) → Sigmoid

    Args:
        d_model: Input feature dimension.
        dropout: Dropout between the two linear layers.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid(),  # constrain output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model) — pooled sequence representation.
        Returns:
            (B, 2) — predicted (x, y) fingertip coordinates in [0, 1].
        """
        return self.net(x)


# ── Gesture head ───────────────────────────────────────────────────────────────


class GestureHead(nn.Module):
    """
    Binary classifier (draw=1 / idle=0) operating on the CLS token output.

    Returns raw logits (not softmax) for use with nn.CrossEntropyLoss,
    which applies log-softmax internally. Do not apply softmax here.

    Architecture: Linear(d → 2)

    Args:
        d_model: Input feature dimension (CLS token dimension).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_token: (B, d_model) — CLS token from the transformer output.
        Returns:
            (B, 2) — raw logits for {idle=0, draw=1}.
        """
        return self.classifier(cls_token)


# ── Main model ─────────────────────────────────────────────────────────────────


class STTransformer(nn.Module):
    """
    Spatial-Temporal Transformer for AirSketch fingertip trajectory prediction.

    Takes a T=16 frame window of 21-joint hand landmarks and predicts:
      1. The (x, y) index fingertip position at frame T+1 (regression)
      2. Whether the window is a drawing gesture or idle (classification)

    Both outputs are produced in a single forward pass and trained jointly
    with a weighted loss in issue #10.

    Args:
        config: STTransformerConfig instance. Build from YAML with
                STTransformerConfig.from_yaml(config_dict).

    Input shape:
        sequence: (B, T, 42) — T=16 frames, 42 = 21 joints × 2 coords.

    Output:
        dict with keys:
            "pred_xy":        (B, 2)  float32  fingertip prediction in [0, 1]
            "gesture_logits": (B, 2)  float32  raw logits for gesture classifier
    """

    def __init__(self, config: STTransformerConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        # J = config.num_landmarks  # 21
        # T = config.window_size  # 16

        # ── Input projection: (B, T, J, 2) → (B, T, J, d) ───────────────────
        # Projects raw (x, y) coordinates per joint into d-dimensional space.
        # Applied per-joint: same linear weights for all 21 joints.
        self.input_proj = nn.Linear(2, d)

        # ── CLS token (learnable, prepended to the temporal sequence) ─────────
        # Shape (1, 1, 1, d) — broadcast over batch and joint dimensions.
        # After temporal attention, the CLS position aggregates global
        # sequence information and feeds the gesture classifier.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, d))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Stacked ST-Blocks ─────────────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [
                STBlock(d, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )

        # ── Output layer norm (applied before heads) ──────────────────────────
        self.norm = nn.LayerNorm(d)

        # ── Prediction heads ──────────────────────────────────────────────────
        self.regression_head = RegressionHead(d, config.dropout)
        self.gesture_head = GestureHead(d)

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize all linear layers with truncated normal (std=0.02)
        and all biases to zero. This matches the ViT / BERT initialization
        convention and provides stable gradients at the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            sequence: (B, T, 42) float32 — flattened landmark window.

        Returns:
            dict:
                "pred_xy":        (B, 2) — fingertip prediction in [0, 1].
                "gesture_logits": (B, 2) — raw logits for CrossEntropyLoss.
        """
        B, T, _ = sequence.shape
        J = self.config.num_landmarks
        d = self.config.hidden_dim

        # ── Reshape: (B, T, 42) → (B, T, J, 2) ──────────────────────────────
        x = sequence.reshape(B, T, J, 2)  # (B, T, 21, 2)

        # ── Input projection: (B, T, J, 2) → (B, T, J, d) ───────────────────
        x = self.input_proj(x)  # (B, T, 21, d)

        # ── Prepend CLS token along temporal dimension ────────────────────────
        # CLS has shape (1, 1, 1, d) → expand to (B, 1, J, d)
        # Prepend at T=0: sequence becomes (B, T+1, J, d)
        cls = self.cls_token.expand(B, 1, J, d)  # (B, 1, 21, d)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, 21, d)

        # ── Stacked ST-Blocks ─────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)  # (B, T+1, 21, d)

        # ── Layer norm ────────────────────────────────────────────────────────
        x = self.norm(x)  # (B, T+1, 21, d)

        # ── Extract CLS token (temporal position 0) ───────────────────────────
        # Average CLS over the 21 joint dimension → single vector per sample
        cls_out = x[:, 0, :, :].mean(dim=1)  # (B, d)

        # ── Pool the non-CLS positions for regression ─────────────────────────
        # Average over both T and J dimensions of the main sequence frames
        seq_out = x[:, 1:, :, :].mean(dim=(1, 2))  # (B, d)

        # ── Heads ─────────────────────────────────────────────────────────────
        pred_xy = self.regression_head(seq_out)  # (B, 2) in [0, 1]
        gesture_logits = self.gesture_head(cls_out)  # (B, 2) raw logits

        return {
            "pred_xy": pred_xy,
            "gesture_logits": gesture_logits,
        }

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self) -> None:
        """Print a human-readable model summary."""
        cfg = self.config
        total = self.count_parameters()

        print("── AirSketch ST-Transformer ────────────────────────────────────────")
        print(f"  hidden_dim:          {cfg.hidden_dim}")
        print(f"  num_layers:          {cfg.num_layers}")
        print(f"  num_heads:           {cfg.num_heads}")
        print(f"  dropout:             {cfg.dropout}")
        print(f"  window_size (T):     {cfg.window_size}")
        print(f"  num_landmarks (J):   {cfg.num_landmarks}")
        print(f"  input_dim:           {cfg.input_dim}")
        print(f"  input shape:         (B, {cfg.window_size}, {cfg.input_dim})")
        print("  output pred_xy:      (B, 2)")
        print("  output gesture:      (B, 2)")
        print()

        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:<22s} {params:>10,} params")

        print(f"  {'─'*36}")
        print(f"  {'TOTAL':<22s} {total:>10,} params")
        print(f"  {'─'*36}")

        # Sanity check — transformers in this size range should be 0.5–5M params
        if total < 100_000:
            print(f"  WARNING: Model may be too small ({total:,} params).")
        elif total > 20_000_000:
            print(
                f"  WARNING: Model may be too large for CPU inference ({total:,} params)."
            )
        else:
            print("  ✓ Parameter count looks reasonable for CPU deployment.")


def build_model(config: dict, device: str = "cpu") -> STTransformer:
    """
    Build and return an STTransformer from the full config dict.

    Args:
        config: Full config dict loaded from default.yaml.
        device: Device string — "cpu", "cuda", "cuda:0", etc.

    Returns:
        STTransformer on the specified device, in training mode.

    Usage:
        import yaml
        with open("configs/default.yaml") as f:
            config = yaml.safe_load(f)
        model = build_model(config, device="cuda")
    """
    cfg = STTransformerConfig.from_yaml(config)
    model = STTransformer(cfg)
    model = model.to(device)
    model.print_summary()
    return model


class AirSketchLoss(nn.Module):
    """
    Joint loss for ST-Transformer training.

    Combines:
        L = L1(pred_xy, gt_xy) + λ * CrossEntropy(gesture_logits, gesture_labels)

    L1 loss is used for regression (rather than MSE) because it is less
    sensitive to outlier frames — MediaPipe occasionally produces large
    landmark errors on occluded or fast-moving hands, and L1's linear
    penalty prevents these from dominating the gradient.

    Args:
        gesture_loss_weight: Lambda (λ) weighting the gesture classification
                             term. Default: 0.5. Tune via hyperparameter sweep.
        class_weights:       Optional (2,) tensor of inverse-frequency weights
                             for the gesture classes (idle, draw). Pass the
                             output of dataset.get_class_weights() here to
                             handle FreiHAND's class imbalance.
    """

    def __init__(
        self,
        gesture_loss_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lam = gesture_loss_weight
        self.regression_loss = nn.L1Loss()
        self.gesture_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        pred_xy: torch.Tensor,
        gt_xy: torch.Tensor,
        gesture_logits: torch.Tensor,
        gesture_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the joint loss.

        Args:
            pred_xy:        (B, 2)  — predicted fingertip coords in [0, 1].
            gt_xy:          (B, 2)  — ground-truth fingertip coords in [0, 1].
            gesture_logits: (B, 2)  — raw gesture classifier logits.
            gesture_labels: (B,)    — int64 labels {0=idle, 1=draw}.

        Returns:
            dict with keys:
                "loss":           Total combined loss (scalar).
                "regression_loss": L1 component (scalar).
                "gesture_loss":    CrossEntropy component (scalar).
        """
        reg_loss = self.regression_loss(pred_xy, gt_xy)
        ges_loss = self.gesture_loss(gesture_logits, gesture_labels)
        total = reg_loss + self.lam * ges_loss

        return {
            "loss": total,
            "regression_loss": reg_loss,
            "gesture_loss": ges_loss,
        }
