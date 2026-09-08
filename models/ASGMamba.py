import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ================================================================
# 0. Mamba dependency
# ================================================================
# The reported model uses the actual Mamba implementation.  Do not silently
# replace it with a Linear layer: that would produce a different model while
# preserving the same tensor shapes.
try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - depends on the local runtime
    raise ImportError(
        "ASGMamba requires the 'mamba_ssm' package. Install it before "
        "training or evaluating the paper model."
    ) from exc


# ================================================================
# 1. Reversible Instance Normalization
# ================================================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        self._mean = None
        self._stdev = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Args:
            x:
                norm mode:   [B, L, C]
                denorm mode: [B, T, C]
            mode: "norm" or "denorm"
        """
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.eps
            ).detach()

            x = (x - self._mean) / self._stdev

            if self.affine:
                x = x * self.affine_weight + self.affine_bias

            return x

        if mode == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError(
                    "RevIN denorm was called before norm in the same forward pass."
                )

            if self.affine:
                safe_weight = self.affine_weight + self.eps
                x = (x - self.affine_bias) / safe_weight

            return x * self._stdev + self._mean

        raise ValueError(f"Unsupported RevIN mode: {mode!r}")


# ================================================================
# 2. ASG spectral router
#    Paper implementation:
#      G = 1 + 0.5 * tanh(logits), G in (0.5, 1.5)
#      The final projection is zero-initialized, so G starts at one.
# ================================================================
class PatchFrequencyRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        gate_amplitude: float = 0.5,
    ):
        super().__init__()

        if not 0.0 < gate_amplitude <= 1.0:
            raise ValueError("gate_amplitude must be in (0, 1].")

        hidden_dim = max(d_model // 4, 8)
        self.gate_amplitude = gate_amplitude

        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)

        # Identity initialization:
        # logits = 0 -> tanh(0) = 0 -> gate = 1.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _three_band_energy(x_patch: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized low/mid/high spectral energy.

        Args:
            x_patch: [B*C, P, patch_len]

        Returns:
            freq_features: [B*C, P, 3]
        """
        x_fft = torch.fft.rfft(x_patch, dim=-1)
        energy = x_fft.abs().pow(2)

        # The paper defines bands from the normalized rFFT frequency
        # nu_j = 2j/P, rather than splitting the available bins equally.
        patch_len = x_patch.size(-1)
        freq_len = energy.size(-1)
        nu = 2.0 * torch.arange(
            freq_len,
            device=x_patch.device,
            dtype=energy.dtype,
        ) / float(patch_len)

        low_mask = nu <= (1.0 / 3.0)
        mid_mask = (nu > (1.0 / 3.0)) & (nu <= (2.0 / 3.0))
        high_mask = nu > (2.0 / 3.0)

        e_low = energy[..., low_mask].sum(dim=-1, keepdim=True)
        e_mid = energy[..., mid_mask].sum(dim=-1, keepdim=True)
        e_high = energy[..., high_mask].sum(dim=-1, keepdim=True)

        freq_features = torch.cat([e_low, e_mid, e_high], dim=-1)
        total_energy = energy.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return freq_features / total_energy

    def forward(self, x_patch: torch.Tensor) -> torch.Tensor:
        freq_features = self._three_band_energy(x_patch)

        logits = self.fc2(F.silu(self.fc1(freq_features)))

        # Residual gate centered at 1.
        # With alpha=0.5, the range is approximately (0.5, 1.5).
        gate = 1.0 + self.gate_amplitude * torch.tanh(logits)
        return gate


# ================================================================
# 3. Patch-scale layer
#    Paper implementation:
#      - non-overlapping zero-padded patches
#      - identity-centered spectral gate
#      - Mamba with d_conv=configs.d_conv (=4 in the paper setting)
#      - dropout only after Mamba, before the residual addition
# ================================================================
class PatchScaleLayer(nn.Module):
    def __init__(self, configs, patch_len: int):
        super().__init__()

        self.patch_len = patch_len
        self.d_model = configs.d_model

        # Keep the original non-overlapping patching protocol.
        pad_len = (patch_len - configs.seq_len % patch_len) % patch_len
        self.num_patches = (configs.seq_len + pad_len) // patch_len

        self.patch_embed = nn.Linear(patch_len, configs.d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                1,
                self.num_patches,
                configs.d_model,
            ) * 0.02
        )

        self.router = PatchFrequencyRouter(
            d_model=configs.d_model,
            gate_amplitude=0.5,
        )

        self.mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )

        self.norm = nn.LayerNorm(configs.d_model)
        self.output_dropout = nn.Dropout(configs.dropout)

        self.head = nn.Linear(
            self.num_patches * configs.d_model,
            configs.pred_len,
        )

        # Optional diagnostic cache. It is detached and does not affect training.
        self.last_gate_mean = None

    def forward(
        self,
        x: torch.Tensor,
        node_embed_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
            node_embed_map: [1, C, 1, D]

        Returns:
            pred: [B, C, T]
        """
        batch_size, num_channels, seq_len = x.shape

        # --------------------------------------------------------
        # Non-overlapping patching
        # --------------------------------------------------------
        pad_len = (self.patch_len - seq_len % self.patch_len) % self.patch_len

        # Preserve the original zero-padding behavior for checkpoint/protocol
        # compatibility with the uploaded implementation.
        x_pad = F.pad(x, (0, pad_len))

        x_patches = x_pad.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.patch_len,
        )
        # [B, C, P, patch_len]
        x_patches = x_patches.reshape(
            batch_size,
            num_channels,
            self.num_patches,
            self.patch_len,
        )

        # --------------------------------------------------------
        # Patch, positional, and variable-identity embeddings
        # --------------------------------------------------------
        x_encoded = self.patch_embed(x_patches)
        x_encoded = x_encoded + self.pos_embedding

        if node_embed_map is not None:
            x_encoded = x_encoded + node_embed_map

        # [B*C, P, D]
        x_encoded_flat = x_encoded.reshape(
            batch_size * num_channels,
            self.num_patches,
            self.d_model,
        )

        # --------------------------------------------------------
        # Identity-centered adaptive spectral gating
        # --------------------------------------------------------
        x_patches_flat = x_patches.reshape(
            batch_size * num_channels,
            self.num_patches,
            self.patch_len,
        )
        gate_weights = self.router(x_patches_flat)

        residual = x_encoded_flat
        x_normalized = self.norm(x_encoded_flat)
        x_gated = x_normalized * gate_weights

        # --------------------------------------------------------
        # Mamba and residual path
        # --------------------------------------------------------
        x_out = self.mamba(x_gated)
        x_out = self.output_dropout(x_out)
        x_out = x_out + residual

        # --------------------------------------------------------
        # Forecasting head
        # --------------------------------------------------------
        x_out = x_out.reshape(batch_size, num_channels, -1)
        pred = self.head(x_out)

        self.last_gate_mean = gate_weights.detach().mean()

        return pred


# ================================================================
# 4. ASGMamba A2
#    Keep the original global learnable Softmax fusion.
# ================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # Default-value protection.
        if not hasattr(configs, "d_model"):
            configs.d_model = 128
        if not hasattr(configs, "d_state"):
            configs.d_state = 16
        if not hasattr(configs, "d_conv"):
            configs.d_conv = 4
        if not hasattr(configs, "expand"):
            configs.expand = 2
        if not hasattr(configs, "dropout"):
            configs.dropout = 0.1
        required_fields = ["seq_len", "pred_len", "enc_in"]
        missing = [name for name in required_fields if not hasattr(configs, name)]
        if missing:
            raise AttributeError(
                f"Missing required config fields: {', '.join(missing)}"
            )

        self.configs = configs
        self.revin = RevIN(configs.enc_in)

        self.patch_sizes = [8, 16, 32]

        self.scales = nn.ModuleList(
            [
                PatchScaleLayer(configs, patch_len)
                for patch_len in self.patch_sizes
            ]
        )

        # Original dataset-level global fusion.
        self.scale_weights = nn.Parameter(
            torch.ones(len(self.patch_sizes))
        )

        self.node_embed = nn.Parameter(
            torch.randn(
                1,
                configs.enc_in,
                1,
                configs.d_model,
            ) * 0.02
        )

        # Optional diagnostic cache.
        self.last_scale_weights = None

    def forward(
        self,
        x: torch.Tensor,
        x_mark=None,
        y_true=None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, C]

        Returns:
            forecast: [B, T, C]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B, L, C], got {tuple(x.shape)}."
            )

        # RevIN normalization.
        x = self.revin(x, "norm")

        # [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1).contiguous()

        outputs = [
            scale_layer(x, node_embed_map=self.node_embed)
            for scale_layer in self.scales
        ]

        # Original global Softmax scale fusion.
        weights = F.softmax(self.scale_weights, dim=0)

        final_pred = torch.zeros_like(outputs[0])
        for scale_index, scale_output in enumerate(outputs):
            final_pred = final_pred + weights[scale_index] * scale_output

        self.last_scale_weights = weights.detach()

        # [B, C, T] -> [B, T, C]
        final_pred = final_pred.permute(0, 2, 1).contiguous()

        return self.revin(final_pred, "denorm")


# ================================================================
# 5. Minimal shape test
# ================================================================
if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    )

    model = Model(config)
    sample = torch.randn(2, config.seq_len, config.enc_in)
    output = model(sample)

    print("Input shape: ", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print(
        "Scale weights:",
        model.last_scale_weights.cpu().tolist()
        if model.last_scale_weights is not None
        else None,
    )

    assert output.shape == (
        sample.size(0),
        config.pred_len,
        config.enc_in,
    )
