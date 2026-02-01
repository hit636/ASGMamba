import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

# ----------------- 配置类 -----------------
class Configs:
    """
    用于存储 AHEWN 模型超参数的配置类
    """
    def __init__(self, seq_len=96, pred_len=96, enc_in=7, 
                 wavelet_levels=3, d_model=128, mixer_layers=2):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in  # C: 通道数
        self.wavelet_levels = wavelet_levels # L: 分解层数
        self.d_model = d_model
        self.mixer_layers = mixer_layers

# ----------------- 模块二：尺度专用预测器 (的核心组件) -----------------
class MLPMixer(nn.Module):
    """
    MLP-Mixer 模块，用于时间和通道混合。
    """
    def __init__(self, input_len, output_len, num_channels, d_model, mixer_layers):
        super(MLPMixer, self).__init__()
        
        self.input_proj = nn.Linear(input_len, d_model)
        
        self.temporal_mixers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            ) for _ in range(mixer_layers)
        ])
        
        self.channel_mixers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
            ) for _ in range(mixer_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, output_len)

    def forward(self, x):
        # x: [B, C, L_in]
        x = self.input_proj(x)
        
        for temporal_mixer, channel_mixer in zip(self.temporal_mixers, self.channel_mixers):
            x = x + temporal_mixer(x)
            x = x + channel_mixer(x.permute(0, 2, 1)).permute(0, 2, 1)
            
        x = self.output_proj(x)
        return x

# ----------------- 主模型 AHEWN (已修复 forward 签名) -----------------
class AHEWN(nn.Module):
    """
    Adaptive Hierarchical Ensemble Wavelet Network (AHEWN)
    """
    def __init__(self, configs):
        super(AHEWN, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.wavelet_levels = configs.wavelet_levels
        self.wavelet = 'db4'

        self.in_coeffs_lens = self._get_coeffs_lengths(self.seq_len)
        self.out_coeffs_lens = self._get_coeffs_lengths(self.pred_len)
        
        self.cA_predictors = nn.ModuleList()
        self.cD_predictors = nn.ModuleList()

        # 为 cA_L (最粗) 创建预测器
        self.cA_predictors.append(MLPMixer(
            input_len=self.in_coeffs_lens[0], output_len=self.out_coeffs_lens[0],
            num_channels=self.channels, d_model=configs.d_model, mixer_layers=configs.mixer_layers
        ))

        # 为 cA_{L-1} ... cA_0 创建预测器 (输入为拼接后)
        for i in range(1, self.wavelet_levels + 1):
            self.cA_predictors.append(MLPMixer(
                input_len=self.in_coeffs_lens[i] * 2, output_len=self.out_coeffs_lens[i],
                num_channels=self.channels, d_model=configs.d_model, mixer_layers=configs.mixer_layers
            ))
        
        # 为 cD_L ... cD_1 创建预测器
        for i in range(self.wavelet_levels):
            self.cD_predictors.append(MLPMixer(
                input_len=self.in_coeffs_lens[i+1], output_len=self.out_coeffs_lens[i+1],
                num_channels=self.channels, d_model=configs.d_model, mixer_layers=configs.mixer_layers
            ))
            
        context_dim = configs.d_model * self.channels
        self.gating_network = nn.Sequential(
            nn.Linear(context_dim, self.wavelet_levels + 1),
            nn.Softmax(dim=-1)
        )

    def _get_coeffs_lengths(self, length):
        coeffs = pywt.wavedec(np.random.randn(length), self.wavelet, level=self.wavelet_levels)
        return [len(c) for c in coeffs]

    def _decompose(self, x):
        coeffs_tuple = pywt.wavedec(x.cpu().numpy(), self.wavelet, level=self.wavelet_levels, axis=-1)
        return [torch.from_numpy(c).float().to(x.device) for c in coeffs_tuple]

    def _reconstruct(self, coeffs_list):
        numpy_coeffs = [c.cpu().numpy() if isinstance(c, torch.Tensor) else None for c in coeffs_list]
        reconstructed = pywt.waverec(numpy_coeffs, self.wavelet, axis=-1)
        return torch.from_numpy(reconstructed).float()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        ### 核心修复 ###
        修改 forward 签名以兼容标准训练框架，只使用第一个参数 x_enc。
        """
        # x_enc: [B, L_in, C]
        x = x_enc.permute(0, 2, 1) # -> [B, C, L_in]
        B, C, L_in = x.shape
        device = x.device

        # 模块一: 分解
        coeffs_x = self._decompose(x)
        
        # 模块三: 分层残差修正
        predicted_coeffs = [None] * (self.wavelet_levels + 1)
        
        # 1. 基准预测 (最粗尺度 L)
        ca_l_hidden = self.cA_predictors[0].input_proj(coeffs_x[0])
        h_context = ca_l_hidden.reshape(B, -1) # 全局上下文 [B, C * d_model]
        predicted_coeffs[0] = self.cA_predictors[0].output_proj(ca_l_hidden) # ĉA_L
        predicted_coeffs[1] = self.cD_predictors[0](coeffs_x[1]) # ĉD_L
        
        # 2. 迭代修正 (从 L-1 到 0)
        for i in range(1, self.wavelet_levels + 1):
            cA_recon = pywt.idwt(predicted_coeffs[i-1].detach().cpu(), 
                                 predicted_coeffs[i].detach().cpu(), 
                                 self.wavelet, axis=-1)
            cA_recon = torch.from_numpy(cA_recon).float().to(device)
            
            target_len = self.in_coeffs_lens[i]
            if cA_recon.shape[-1] > target_len: cA_recon = cA_recon[..., :target_len]
            elif cA_recon.shape[-1] < target_len: cA_recon = F.pad(cA_recon, (0, target_len - cA_recon.shape[-1]))

            ca_i_input = torch.cat([coeffs_x[i], cA_recon], dim=-1)
            predicted_coeffs[i] = self.cA_predictors[i](ca_i_input) # 修正ĉA_{L-i}
            
            if i < self.wavelet_levels:
                predicted_coeffs[i+1] = self.cD_predictors[i](coeffs_x[i+1]) # 预测ĉD_{L-i-1}

        # 模块四: 自适应集成合成器
        ensemble_weights = self.gating_network(h_context) # -> [B, L+1]
        
        final_prediction = torch.zeros(B, C, self.pred_len, device=device)
        
        # 对每个尺度重构一个完整的预测，然后加权
        for i in range(self.wavelet_levels + 1):
            temp_coeffs = list(predicted_coeffs)
            
            # 创建一个用于重构的系数列表副本
            recon_coeffs_list = [None] * (self.wavelet_levels + 1)
            recon_coeffs_list[0:i+1] = temp_coeffs[0:i+1]

            recon = self._reconstruct(recon_coeffs_list).to(device)
            
            if recon.shape[-1] > self.pred_len: recon = recon[..., :self.pred_len]
            elif recon.shape[-1] < self.pred_len: recon = F.pad(recon, (0, self.pred_len - recon.shape[-1]))
            
            weight = ensemble_weights[:, i].view(B, 1, 1)
            final_prediction += recon * weight

        return final_prediction.permute(0, 2, 1) # -> [B, L_out, C]