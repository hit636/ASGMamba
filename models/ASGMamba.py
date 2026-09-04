import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ================================================================
# 0. Mamba 依赖 (双向扫描基础)
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
# 1. 基础组件 (RevIN)
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
# 2. Spectral AdaLN Router (频域自适应层归一化)
# ================================================================
class SpectralAdaLNRouter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # >>> [优化点2: AdaLN] 输出 2*d_model，分别作为 gamma 和 beta
        self.mlp = nn.Sequential(
            nn.Linear(3, d_model // 4),
            nn.SiLU(), # SiLU 比 ReLU 在生成调节参数时更平滑稳定
            nn.Linear(d_model // 4, d_model * 2) 
        )
        
        # 初始化最后一步为0，使得初始状态下相当于标准的 LayerNorm
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

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
        
        # 拆分为 Scale(gamma) 和 Shift(beta)
        ada_params = self.mlp(freq_feats) # [..., 2 * d_model]
        gamma, beta = ada_params.chunk(2, dim=-1) # 各自 [..., d_model]
        return gamma, beta

# ================================================================
# 3. Patch Scale Layer (Bi-Mamba + AdaLN)
# ================================================================
class PatchScaleLayer(nn.Module):
    def __init__(self, configs, patch_len):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = configs.d_model
        
        self.patch_embed = nn.Linear(patch_len, configs.d_model)
        
        # 恢复非重叠切片，防止最后全连接层参数暴增导致过拟合
        pad_len = (patch_len - configs.seq_len % patch_len) % patch_len
        self.num_patches = (configs.seq_len + pad_len) // patch_len
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, configs.d_model) * 0.02)
        
        self.router = SpectralAdaLNRouter(configs.d_model)
        self.norm = nn.LayerNorm(configs.d_model)
        
        # >>> [优化点1: Bi-Mamba] 前向和后向两个 Mamba 块
        self.mamba_fwd = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.mamba_bwd = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        
        self.dropout_layer = nn.Dropout(configs.dropout)
        self.head = nn.Linear(self.num_patches * configs.d_model, configs.pred_len)

    def forward(self, x, node_embed_map=None):
        B, N, L = x.shape
        
        pad_len = (self.patch_len - L % self.patch_len) % self.patch_len
        x_pad = F.pad(x, (0, pad_len))
        # 保持 step=patch_len
        x_patches = x_pad.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x_patches_bn = x_patches.reshape(B, N, -1, self.patch_len)
        
        x_enc = self.patch_embed(x_patches_bn) + self.pos_embedding
        if node_embed_map is not None:
            x_enc = x_enc + node_embed_map
            
        x_enc_flat = x_enc.reshape(B * N, -1, self.d_model)
        
        # --- Spectral AdaLN ---
        x_patches_flat = x_patches.reshape(B * N, -1, self.patch_len)
        gamma, beta = self.router(x_patches_flat) 
        
        residual = x_enc_flat 
        x_norm = self.norm(x_enc_flat)
        # 用 AdaLN 代替直接乘以 Gate，训练极其稳定
        x_modulated = x_norm * (1.0 + gamma) + beta 
        
        # --- Bi-Mamba Scan ---
        out_fwd = self.mamba_fwd(x_modulated)
        # 将序列在时间维度(dim=1)翻转后反向扫描，再翻转回来
        out_bwd = torch.flip(self.mamba_bwd(torch.flip(x_modulated, dims=[1])), dims=[1])
        
        x_out = out_fwd + out_bwd
        x_out = self.dropout_layer(x_out) + residual 
        
        # --- Head ---
        out_flat = x_out.reshape(B, N, -1) 
        pred = self.head(out_flat) 
        
        return pred

# ================================================================
# 4. 主模型
# ================================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        for attr, val in zip(['d_model','d_state','d_conv','expand','dropout'], [128, 16, 4, 2, 0.1]):
            if not hasattr(configs, attr): setattr(configs, attr, val)
            
        self.configs = configs
        self.revin = RevIN(configs.enc_in)
        
        self.patch_sizes = [8, 16, 32]
        self.scales = nn.ModuleList([PatchScaleLayer(configs, p) for p in self.patch_sizes])
        
        # >>> [优化点3: 变量特异性融合] 
        # 形状为 [1, 变量数, 1, 尺度数]，让每个变量自己学习最需要哪个尺度
        self.scale_weights = nn.Parameter(torch.ones(1, configs.enc_in, 1, len(self.patch_sizes)))
        
        self.node_embed = nn.Parameter(torch.randn(1, configs.enc_in, 1, configs.d_model) * 0.02)

    def forward(self, x, x_mark=None, y_true=None):
        x = self.revin(x, 'norm')
        x = x.permute(0, 2, 1) # [B, N, L]
        
        outputs = []
        for layer in self.scales:
            outputs.append(layer(x, node_embed_map=self.node_embed))
            
        # 将三个尺度的输出堆叠: [B, N, T, 3]
        outputs_stack = torch.stack(outputs, dim=-1) 
        
        # 变量特异性 Softmax 权重: [1, N, 1, 3]
        weights = F.softmax(self.scale_weights, dim=-1)
        
        # 加权融合: [B, N, T]
        final_pred = (outputs_stack * weights).sum(dim=-1) 
        
        final_pred = final_pred.permute(0, 2, 1) # [B, T, N]
        return self.revin(final_pred, 'denorm')
