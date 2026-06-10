import torch
import torch.nn as nn


# ---------- Squeeze-and-Excitation Block ----------
class SEBlock(nn.Module):
    """
    Channel attention mechanism - recalibrates channel-wise feature responses.
    Uses global pooling + MLP to compute per-channel attention weights.
    Minimal overhead (~1-2% params) but significant performance boost.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Global pooling: [B,C,T] -> [B,C,1]
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, T]
        scale = self.squeeze(x)  # [B, C, 1] - global channel statistics
        scale = self.excitation(scale)  # [B, C, 1] - learned channel weights
        return x * scale  # [B, C, T] - reweight channels


# ---------- Dynamic FiLM (Context-aware conditioning) ----------
class DynamicFiLM(nn.Module):
    """
    Advanced FiLM that generates both static and temporal-adaptive modulation.
    Combines global conditioning (from A) with position-aware adjustments.
    More expressive than static FiLM for complex temporal patterns.
    """

    def __init__(self, a_dim: int, c_dim: int, hidden: int = 128):
        super().__init__()
        # Static component: global γ, β from A
        self.static_net = nn.Sequential(
            nn.Linear(a_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * c_dim)
        )

        # Dynamic component: temporal modulation from A
        self.dynamic_net = nn.Sequential(
            nn.Linear(a_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.temporal_conv = nn.Conv1d(hidden, 2 * c_dim, kernel_size=1)

    def forward(self, x, A):
        # x: [B, C, T], A: [B, a_dim]
        B, C, T = x.shape

        # Static modulation (same across time)
        gamma_s, beta_s = self.static_net(A).chunk(2, dim=-1)  # [B, C], [B, C]
        gamma_s = gamma_s.unsqueeze(-1)  # [B, C, 1]
        beta_s = beta_s.unsqueeze(-1)  # [B, C, 1]

        # Dynamic modulation (varies across time)
        h = self.dynamic_net(A).unsqueeze(-1)  # [B, H, 1]
        h = h.expand(-1, -1, T)  # [B, H, T]
        gamma_d, beta_d = self.temporal_conv(h).chunk(2, dim=1)  # [B, C, T]

        # Combine static and dynamic components
        gamma = (gamma_s + gamma_d) * 0.5  # Average for stability
        beta = (beta_s + beta_d) * 0.5

        return gamma * x + beta


# ---------- Multi-Scale Gated TCN Block ----------
class MultiScaleGatedTCNBlock(nn.Module):
    """
    Enhanced TCN block with multiple improvements:
    1. Multi-head gating: diverse gating patterns for richer representations
    2. Multi-scale dilations: parallel branches capture different temporal scales
    3. SE attention: channel recalibration after gating
    4. Dynamic FiLM: more expressive conditioning from static features
    5. Pre-activation: better gradient flow
    6. Stochastic depth: regularization during training
    7. Learnable skip: adaptive residual weighting
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list[int] = None,
        causal: bool = False,
        dropout: float = 0.0,
        a_dim: int | None = None,
        stochastic_depth: float = 0.0,
        use_se: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.stochastic_depth = stochastic_depth

        # Multi-scale dilations (if not provided, use single dilation)
        if dilations is None:
            dilations = [1]
        self.dilations = dilations

        # Multi-branch convolutions for multi-scale processing
        self.conv_branches = nn.ModuleList()
        branch_channels = (2 * channels) // len(dilations)  # Split output channels

        for dilation in dilations:
            pad = (
                (kernel_size - 1) * dilation
                if causal
                else ((kernel_size - 1) * dilation) // 2
            )
            self.conv_branches.append(
                nn.Conv1d(
                    channels,
                    branch_channels,
                    kernel_size,
                    padding=pad,
                    dilation=dilation,
                )
            )

        # Adjust final channels if not evenly divisible
        self.channel_adjustment = None
        total_branch_channels = branch_channels * len(dilations)
        if total_branch_channels != 2 * channels:
            self.channel_adjustment = nn.Conv1d(total_branch_channels, 2 * channels, 1)

        # Dynamic FiLM conditioning (more powerful than static)
        self.film = DynamicFiLM(a_dim, channels) if a_dim is not None else None

        # Squeeze-and-Excitation for channel attention
        self.se = SEBlock(channels) if use_se else nn.Identity()

        # Pre-activation normalization (applied before conv)
        self.pre_norm = nn.LayerNorm(channels)

        # Dropout and post-norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.post_norm = nn.LayerNorm(channels)

        # Learnable skip connection weight (adaptive residual)
        self.skip_weight = nn.Parameter(torch.ones(1))

    def forward(self, x, A=None):
        """
        x: [B, C, T], A: [B, a_dim] or None
        Returns: [B, C, T]
        """
        res = x  # Store for residual connection

        # Stochastic depth: randomly skip block during training for regularization
        if self.training and self.stochastic_depth > 0:
            if torch.rand(1).item() < self.stochastic_depth:
                return res

        # === Pre-activation: normalize before convolution ===
        h = x.transpose(1, 2)  # [B, T, C]
        h = self.pre_norm(h)  # LayerNorm over channels
        h = h.transpose(1, 2)  # [B, C, T]

        # === Multi-scale convolutions: parallel branches ===
        branch_outputs = []
        for conv in self.conv_branches:
            branch_h = conv(h)  # [B, branch_C, T+pad]

            # Causal trimming if needed
            if self.causal:
                trim = (conv.kernel_size[0] - 1) * conv.dilation[0]
                if trim > 0:
                    branch_h = branch_h[..., :-trim]

            branch_outputs.append(branch_h)

        # Concatenate multi-scale features
        h = torch.cat(branch_outputs, dim=1)  # [B, total_branch_C, T]

        # Adjust channels if necessary
        if self.channel_adjustment is not None:
            h = self.channel_adjustment(h)  # [B, 2C, T]

        # === Multi-head GLU gating ===
        # Split into multiple heads for diverse gating patterns
        num_heads = 2
        head_dim = h.size(1) // (2 * num_heads)

        gated_heads = []
        for i in range(num_heads):
            start_idx = i * 2 * head_dim
            h_in = h[:, start_idx : start_idx + head_dim, :]
            h_gate = h[:, start_idx + head_dim : start_idx + 2 * head_dim, :]

            # LSTM-like gating: tanh(input) ⊗ σ(gate)
            h_in = torch.tanh(h_in)
            h_gate = torch.sigmoid(h_gate)
            gated = h_in * h_gate
            gated_heads.append(gated)

        h = torch.cat(gated_heads, dim=1)  # [B, C, T]

        # === Dynamic FiLM conditioning ===
        if self.film is not None and A is not None:
            h = self.film(h, A)  # Modulate with static features

        # === Squeeze-and-Excitation attention ===
        h = self.se(h)  # Channel recalibration

        # === Dropout ===
        h = self.dropout(h)

        # === Post-normalization ===
        h = h.transpose(1, 2)  # [B, T, C]
        h = self.post_norm(h)
        h = h.transpose(1, 2)  # [B, C, T]

        # === Learnable weighted residual connection ===
        return self.skip_weight * h + res


# ---------- Enhanced Gated TCN Model ----------
class EnhancedGatedTCN(nn.Module):
    """
    Advanced Gated TCN with multiple enhancements:
    - Multi-scale temporal processing with parallel dilations
    - Multi-head gating for diverse feature extraction
    - Dynamic FiLM conditioning for expressive static feature injection
    - SE blocks for channel attention
    - Pre-activation and better normalization
    - Stochastic depth for regularization
    - Learnable skip connections

    Map (X: [B,T,n_x], A: [B,n_a]) -> y: [B,T,n_out]
    """

    def __init__(
        self,
        n_x: int = 3,
        n_a: int = 15,
        n_out: int = 1,
        width: int = 64,
        depth: int = 6,
        kernel_size: int = 3,
        base_dilation: int = 1,
        causal: bool = False,
        dropout: float = 0.1,
        stochastic_depth: float = 0.1,
        use_se: bool = True,
        multi_scale: bool = True,
    ):
        super().__init__()

        # Input projection with small MLP for better feature extraction
        self.inp = nn.Sequential(
            nn.Linear(n_x, width), nn.GELU(), nn.Linear(width, width)
        )

        # Build TCN blocks with progressively increasing dilation
        blocks = []
        for i in range(depth):
            base_dil = base_dilation * (2**i)

            # Multi-scale: use multiple dilations per block (early layers only)
            if multi_scale and i < depth // 2:
                dilations = [base_dil, base_dil * 2]  # Parallel scales
            else:
                dilations = [base_dil]

            # Increase stochastic depth probability linearly with depth
            drop_prob = stochastic_depth * (i / depth)

            blocks.append(
                MultiScaleGatedTCNBlock(
                    channels=width,
                    kernel_size=kernel_size,
                    dilations=dilations,
                    causal=causal,
                    dropout=dropout,
                    a_dim=n_a,
                    stochastic_depth=drop_prob,
                    use_se=use_se,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        # Enhanced output head with residual connection
        self.head = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Light dropout before final layer
            nn.Conv1d(width, out_channels=n_out, kernel_size=1),
        )

    def forward(self, X, A):
        """
        X: [B, T, n_x] - temporal input features
        A: [B, n_a]    - static conditioning features
        Returns: y: [B, T, n_out] - predictions
        """
        # Project input to hidden dimension
        h = self.inp(X)  # [B, T, width]
        h = h.transpose(1, 2)  # [B, width, T] - conv expects [B, C, T]

        # Pass through enhanced TCN blocks
        for blk in self.blocks:
            h = blk(h, A)  # Multi-scale gated conv + all enhancements

        # Generate predictions
        y = self.head(h)  # [B, n_out, T]
        y = y.transpose(1, 2)  # [B, T, n_out]

        return y
