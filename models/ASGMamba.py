import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ================================================================
# 0. Mamba 依赖 (保持不变)
# ================================================================
try:
    from mamba_ssm import Mamba
    print("[Info] Using mamba_ssm acceleration.")
except ImportError:
    print("[Warning] 'mamba_ssm' not found. Falling back to nn.Linear.")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x): return self.linear(x)

# ================================================================
# 1. 基础组件
# ================================================================
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / (self.affine_weight + self.eps*1e-5)
            x = x * self.stdev + self.mean
        return x

# ================================================================
# 2. Patch Router (频域门控)
# ================================================================
class PatchFrequencyRouter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )

    def forward(self, x_patch):
        x_fft = torch.fft.rfft(x_patch, dim=-1)
        energy = torch.abs(x_fft) ** 2
        freq_len = energy.shape[-1]
        
        if freq_len < 3:
            e_low = energy[..., 0:1]
            e_mid = energy[..., 1:2] if freq_len > 1 else torch.zeros_like(e_low)
            e_high = energy[..., 2:] if freq_len > 2 else torch.zeros_like(e_low)
        else:
            split1 = freq_len // 3
            split2 = 2 * freq_len // 3
            e_low  = energy[..., :split1].sum(dim=-1, keepdim=True)
            e_mid  = energy[..., split1:split2].sum(dim=-1, keepdim=True)
            e_high = energy[..., split2:].sum(dim=-1, keepdim=True)
            
        freq_feats = torch.cat([e_low, e_mid, e_high], dim=-1)
        total = freq_feats.sum(dim=-1, keepdim=True) + 1e-6
        freq_feats = freq_feats / total
        
        return self.mlp(freq_feats)

# ================================================================
# 3. Patch Scale Layer (优化：位置编码 + 残差连接)
# ================================================================
class PatchScaleLayer(nn.Module):
    def __init__(self, configs, patch_len):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = configs.d_model
        
        # 1. Embedding
        self.patch_embed = nn.Linear(patch_len, configs.d_model)
        
        # 计算 Patch 数量
        pad_len = (patch_len - configs.seq_len % patch_len) % patch_len
        self.num_patches = (configs.seq_len + pad_len) // patch_len
        
        # >>> [优化点1] 位置编码 (Positional Embedding) <<<
        # 让模型知道每个 Patch 是“开头”还是“结尾”，对预测未来至关重要
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, configs.d_model) * 0.02)
        
        # 2. Router & Mamba
        self.router = PatchFrequencyRouter(configs.d_model)
        
        self.mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand
        )
        
        self.norm = nn.LayerNorm(configs.d_model)
        self.dropout_layer = nn.Dropout(configs.dropout)
        
        # 3. Head
        self.head = nn.Linear(self.num_patches * configs.d_model, configs.pred_len)

    def forward(self, x, node_embed_map=None):
        """
        x: [B, N, L]
        node_embed_map: [1, N, 1, D]
        """
        B, N, L = x.shape
        
        # --- Patching ---
        pad_len = (self.patch_len - L % self.patch_len) % self.patch_len
        x_pad = F.pad(x, (0, pad_len))
        x_patches = x_pad.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x_patches_bn = x_patches.reshape(B, N, -1, self.patch_len) # [B, N, P, S]
        
        # --- Embedding ---
        x_enc = self.patch_embed(x_patches_bn) # [B, N, P, D]
        
        # >>> [优化点1] 注入位置编码 (Broadcasting) <<<
        x_enc = x_enc + self.pos_embedding
        
        # 注入 Node Embedding
        if node_embed_map is not None:
            x_enc = x_enc + node_embed_map
            
        # Reshape for Mamba: [B*N, P, D]
        x_enc_flat = x_enc.reshape(B * N, -1, self.d_model)
        
        # --- Routing (频域门控) ---
        # 获取原始 Patch 用于 FFT
        x_patches_flat = x_patches.reshape(B * N, -1, self.patch_len)
        gate_weights = self.router(x_patches_flat) # [B*N, P, D]
        
        # --- [优化点2] 残差连接结构 (Pre-Norm Residual) ---
        # 保存残差路径
        residual = x_enc_flat 
        
        # Norm
        x_norm = self.norm(x_enc_flat)
        
        # Gating + Mamba
        x_gated = x_norm * gate_weights
        x_out = self.mamba(x_gated)
        
        # Dropout
        x_out = self.dropout_layer(x_out)
        
        # 残差相加：即使 Mamba/Router 表现不好，至少保留原始 Embedding 信息
        x_out = x_out + residual 
        
        # --- Head ---
        out_flat = x_out.reshape(B, N, -1) # [B, N, P*D]
        pred = self.head(out_flat) # [B, N, T]
        
        return pred

# ================================================================
# 4. 主模型
# ================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 默认参数保护
        if not hasattr(configs, 'd_model'): configs.d_model = 128
        if not hasattr(configs, 'd_state'): configs.d_state = 16
        if not hasattr(configs, 'd_conv'): configs.d_conv = 4
        if not hasattr(configs, 'expand'): configs.expand = 2
        
        self.configs = configs
        self.revin = RevIN(configs.enc_in)
        
        # 多尺度设定
        self.patch_sizes = [8, 16, 32]
        
        self.scales = nn.ModuleList([
            PatchScaleLayer(configs, p_len) for p_len in self.patch_sizes
        ])
        
        # 简单的可学习权重融合
        self.scale_weights = nn.Parameter(torch.ones(len(self.patch_sizes)))
        
        # Node Embedding: 给每个变量一个独立的身份标识
        self.node_embed = nn.Parameter(torch.randn(1, configs.enc_in, 1, configs.d_model) * 0.02)

    def forward(self, x, x_mark=None, y_true=None):
        """
        x: [B, Length, Variables]
        """
        # 1. RevIN Normalization
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1) # [B, N, L]
        
        # 2. Multi-Scale Processing
        outputs = []
        for layer in self.scales:
            # 显式传入 node_embed
            out = layer(x, node_embed_map=self.node_embed)
            outputs.append(out)
            
        # 3. Weighted Fusion
        # 使用 Softmax 保证权重之和为 1，数值更稳定
        weights = F.softmax(self.scale_weights, dim=0)
        
        final_pred = torch.zeros_like(outputs[0])
        for i, out in enumerate(outputs):
            final_pred += out * weights[i]
            
        final_pred = final_pred.permute(0, 2, 1) # [B, T, N]
        
        # 4. RevIN Denormalization
        return self.revin(final_pred, 'denorm')


 