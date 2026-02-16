import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pywt

import time
import math
from numpy import dtype


class SharedBoundaries(nn.Module):
    def __init__(self, expert_num):
        super().__init__()
        self.expert_num = expert_num
        self.raw = nn.Parameter(torch.rand(expert_num - 1))  # learnable raw logits in (0,1)

    def forward(self):
        b = torch.sigmoid(self.raw)  # map to (0,1)
        b_sorted, _ = torch.sort(b)  # ensure order

        return b_sorted  # shape: (E - 1,)


class FreqMoE(nn.Module):

    def __init__(self, expert_num, seq_len, boundary_provider=None):
        super(FreqMoE, self).__init__()
        self.expert_num = expert_num
        self.seq_len = seq_len
        self.freq_len = seq_len // 2 + 1    # F

        self.boundary_provider = boundary_provider

        if boundary_provider is None:
            self.raw = nn.Parameter(torch.rand(expert_num - 1))  # fallback: internal version

        self.gating_network = nn.Sequential(nn.Linear(self.freq_len, self.freq_len),
                                            nn.ReLU(),
                                            nn.Linear(self.freq_len, self.expert_num))

    def _get_boundaries(self):
        if self.boundary_provider is None:
            b = torch.sigmoid(self.raw)
        else:
            b = self.boundary_provider()
        return torch.sort(b)[0]  # (E-1,)


    def forward(self, x):
        x_mean = torch.mean(x, dim=2, keepdim=True)     # x_mean->(B, C, 1)
        x = x - x_mean      # x->(B, C, T)
        x_var = torch.var(x, dim=2, keepdim=True) + 1e-5    # x_var->(B, C, 1)
        x = x / torch.sqrt(x_var)

        freq_x = torch.fft.rfft(x, dim=-1)
        total_freq_size = freq_x.size(-1)

        boundaries = self._get_boundaries()

        boundaries = torch.cat([                    # boundaries->(expert_num - 1 + 2)
            torch.tensor([0.0], device=boundaries.device),
            boundaries,
            torch.tensor([1.0], device=boundaries.device)
        ])

        indices = (boundaries * total_freq_size).long()     # indices->(expert_num -1 + 2)
        indices[-1] = total_freq_size

        components = []
        expert_outputs = []
        for i in range(self.expert_num):
            start_idx = indices[i].item()
            end_idx = indices[i + 1].item()

            freq_mask = torch.zeros_like(freq_x)    # freq_mask->(B, C, F)
            if end_idx > start_idx:
                freq_mask[:, :, start_idx:end_idx] = 1

            expert_component = freq_x * freq_mask
            components.append(expert_component.unsqueeze(-1))

            expert_time = torch.fft.irfft(expert_component, n=self.seq_len)  # (B, C, T)

            expert_outputs.append(expert_time.unsqueeze(1))  # (B, 1, C, T)

        components = torch.cat(components, dim=-1)  # (B, C, F, E)
        expert_outputs = torch.cat(expert_outputs, dim=1)  # (B, E, C, T)

        freq_magnitude = torch.abs(freq_x)
        gating_input = freq_magnitude.mean(dim=1)  # (B, F)
        gating_scores = nn.Softmax(dim=-1)(self.gating_network(gating_input))  # (B, E)

        gating_scores = gating_scores.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, E)
        combined_freq_output = torch.sum(components * gating_scores, dim=-1)  # (B, C, F)
        combined_output = torch.fft.irfft(combined_freq_output, n=self.seq_len)  # (B, C, T)
        residual = x - combined_output

        combined_output = combined_output * torch.sqrt(x_var) + x_mean

        return combined_output, boundaries, gating_scores.squeeze(1).squeeze(1), expert_outputs     # 最后一个 expert_outputs: (B, E, C, T)


class CWTMoE(nn.Module):
    def __init__(self, seq_len, input_channels, expert_num=3, num_scales=64, kernel_size=3, hidden_channels=32, boundary_provider=None, dropout_rate=0.1):
        super().__init__()
        self.expert_num = expert_num
        self.num_scales = num_scales

        self.boundary_provider = boundary_provider

        if boundary_provider is None:
            self.raw = nn.Parameter(torch.rand(expert_num - 1))

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
                # nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(hidden_channels, input_channels, kernel_size=kernel_size, padding=kernel_size//2)
            )
            for _ in range(expert_num)
        ])

        # Gating network
        self.gating_net = nn.Sequential(
            nn.Linear(self.num_scales * seq_len, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, expert_num)
        )

    def _get_boundaries(self):
        if self.boundary_provider is None:
            b = torch.sigmoid(self.raw)
        else:
            b = self.boundary_provider()
        return torch.sort(b)[0]

    def forward(self, power_spec):  # power_spec: (B, C, S, T)
        B, C, S, T = power_spec.shape

        boundaries = self._get_boundaries()

        boundaries = torch.cat([
            torch.tensor([0.0], device=power_spec.device),
            boundaries,
            torch.tensor([1.0], device=power_spec.device)
        ])  # (expert_num + 1,)
        indices = (boundaries * S).long()
        indices[-1] = S  # avoid overflow

        expert_outputs = []
        expert_reps = []
        for i in range(self.expert_num):
            start, end = indices[i].item(), indices[i + 1].item()
            if end <= start:
                out = torch.zeros_like(power_spec)  # fallback for degenerate case
            else:
                band = power_spec[:, :, start:end, :]  # (B, C, s_i, T)
                padded_band = nn.functional.pad(band, (0, 0, 0, S - band.shape[2]))  # pad back to (B, C, S, T)
                out = self.experts[i](padded_band)  # out shape: (B, C, S, T)
            expert_outputs.append(out.unsqueeze(-1))  # (B, C, S, T, 1)
            expert_reps.append(out.unsqueeze(1))  # (B, 1, C, S, T)

        # Combine expert outputs
        expert_stack = torch.cat(expert_outputs, dim=-1)  # (B, C, S, T, expert_num)
        expert_reps = torch.cat(expert_reps, dim=1)

        # Gating
        gating_input = power_spec.mean(dim=1)  # (B, S, T)
        gating_input_flat = gating_input.flatten(start_dim=1)  # (B, S*T)
        gating_scores = nn.Softmax(dim=-1)(self.gating_net(gating_input_flat))  # (B, expert_num)
        gating_scores = gating_scores.view(B, 1, 1, 1, self.expert_num)  # broadcast

        output = (expert_stack * gating_scores).sum(dim=-1)  # (B, C, S, T)

        return output, boundaries, gating_scores.squeeze(1).squeeze(1).squeeze(1), expert_reps


class ResolutionLinearAccumulateFusion(nn.Module):
    def __init__(self, num_resolutions, d_model):
        super().__init__()
        self.num_resolutions = num_resolutions

        self.linear_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_resolutions)
        ])

    def forward(self, features_list):
        assert len(features_list) == self.num_resolutions

        out = self.linear_layers[-1](features_list[-1].permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)
        # out = features_list[-1].permute(0, 2, 1)

        for i in reversed(range(self.num_resolutions - 1)):
            mapped = self.linear_layers[i](features_list[i].permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C)
            # mapped = features_list[i].permute(0, 2, 1)
            out = out + mapped

        return out.permute(0, 2, 1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class GatingUnit(nn.Module):
    def __init__(self, input_dim, init_gate_bias=2.94):
        super().__init__()
        self.linear = nn.Linear(input_dim * 2, input_dim)
        nn.init.constant_(self.linear.bias, init_gate_bias)

    def forward(self, x_pred, x_prev):
        combined = torch.cat([x_pred, x_prev], dim=-1)  # [B, T, 2C]
        gate = torch.sigmoid(self.linear(combined))     # [B, T, C]
        out = gate * x_pred + (1 - gate) * x_prev
        return out


def smooth_conv(seq, kernel_size, mode='replicate'):
    """
    Apply 1D smoothing convolution to input of shape (B, T, C)
    """
    if not isinstance(seq, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if seq.dim() != 3:
        raise ValueError("Input must have shape (B, T, C)")

    B, T, C = seq.shape

    # Reshape to (B * C, 1, T)
    seq_reshaped = seq.permute(0, 2, 1).reshape(B * C, 1, T)

    # Create kernel: shape (1, 1, kernel_size)
    kernel = torch.ones(1, 1, kernel_size, dtype=seq.dtype, device=seq.device) / kernel_size

    # Pad: symmetric padding
    pad_left = kernel_size // 2
    pad_right = kernel_size - 1 - pad_left
    seq_padded = F.pad(seq_reshaped, (pad_left, pad_right), mode=mode)  # seq_padded shape: (B * C, 1, T + kernel_size - 1)

    # Convolve
    out = F.conv1d(seq_padded, kernel)  # out shape: (B * C, 1, T)

    # Reshape back to (B, T, C)
    out = out.view(B, C, T).permute(0, 2, 1)    # out shape: (B, T, C)
    return out


class ExpertAlignmentLoss(nn.Module):
    def __init__(self, mode='cosine'):
        super().__init__()
        self.mode = mode

    def diversity_loss(self, expert_reps, mode='cosine'):
        """
        :param expert_reps: (B, E, C, T) or (B, E, C, S, T)
        :return: scalar diversity loss (the higher, the more diverse)
        """
        B, E = expert_reps.shape[:2]

        if expert_reps.ndim == 5:
            # Flatten (C, S, T) to vector
            expert_reps = expert_reps.flatten(start_dim=2)  # (B, E, C*S*T)
        else:
            expert_reps = expert_reps.flatten(start_dim=2)  # (B, E, C*T)

        # Normalize each expert vector for cosine
        if mode == 'cosine':
            expert_reps = nn.functional.normalize(expert_reps, dim=-1)  # (B, E, D)

        # Compute cosine similarity between all pairs
        loss_total = 0.0
        for i in range(E):
            for j in range(i + 1, E):
                sim = (expert_reps[:, i] * expert_reps[:, j]).sum(dim=-1)  # (B,)
                loss_total += sim.mean()  # similarity

        # Total number of pairs
        num_pairs = E * (E - 1) / 2
        return -loss_total / num_pairs

    def norm_std_diversity(self, expert_reps):
        if expert_reps.ndim == 5:
            expert_reps = expert_reps.flatten(start_dim=2)  # (B, E, D)
        else:
            expert_reps = expert_reps.flatten(start_dim=2)

        norms = expert_reps.norm(dim=2)  # (B, E)
        std = norms.std(dim=1).mean()  # scalar
        return std

    def forward(self, fft_expert_out, cwt_expert_out):
        # fft_expert_out: (B, E, C, T)
        # cwt_expert_out: (B, E, C, S, T)

        cwt_flat = cwt_expert_out.flatten(start_dim=3).mean(dim=3, keepdim=True)  # (B, E, C, 1)
        cwt_flat = cwt_flat.expand(-1, -1, -1, fft_expert_out.shape[-1])          # (B, E, C, T)

        if self.mode == 'cosine':
            fft_norm = nn.functional.normalize(fft_expert_out, dim=-1)
            cwt_norm = nn.functional.normalize(cwt_flat, dim=-1)
            loss = 1 - (fft_norm * cwt_norm).sum(dim=-1).mean()
        else:
            loss = (fft_expert_out - cwt_flat).pow(2).mean()

        # Diversity
        div_fft = self.norm_std_diversity(fft_expert_out)
        div_cwt = self.norm_std_diversity(cwt_expert_out)

        return loss, (div_fft + div_cwt)


class M2FMoE(nn.Module):
    def __init__(self, seq_len, enc_in, pre_len, dec_in=1, dropout_rate=0.3, num_scales=64,
                 expert_num=1, resolution_size=[12, 24], fs=1, args=None, use_revin=True, hidden_dim=64,
                 moe_hidden_dim=4, ts_ratio=0.1):

        super(M2FMoE, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pre_len
        self.channels = enc_in
        self.dec_channels = dec_in
        self.dropout_rate = dropout_rate

        self.look_back = int(ts_ratio * self.seq_len)

        self.use_revin = use_revin

        self.resolution = resolution_size
        self.fs = fs
        self.args = args
        self.num_scales = num_scales
        self.num_resolution = len(self.resolution) + 1

        self.expert_num = expert_num

        self.testmap = nn.Linear(enc_in, dec_in)

        self.shared_boundaries = SharedBoundaries(self.expert_num)

        self.fft_moe = FreqMoE(self.expert_num, self.look_back, boundary_provider=self.shared_boundaries)

        self.cwt_moe = CWTMoE(self.look_back, self.channels, self.expert_num, self.num_scales, boundary_provider=self.shared_boundaries, dropout_rate=dropout_rate)

        self.flatten_projector = nn.Sequential(
            nn.Flatten(start_dim=2),                         # (B, C, S*T)
            nn.Linear(self.num_scales * self.look_back, self.look_back),  # (B, C, T)
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv1d(2 * self.channels, self.channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.attn = nn.MultiheadAttention(hidden_dim * 1 + 2 * moe_hidden_dim, 2)

        self.linear_out = nn.Sequential(
            nn.Linear(hidden_dim * 1 + 2 * moe_hidden_dim, hidden_dim),
            nn.BatchNorm1d(self.pred_len),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, self.channels),
        )

        self.resolution_fusion = ResolutionLinearAccumulateFusion(self.num_resolution, self.pred_len)

        self.ebb = PositionalEmbedding(hidden_dim, self.pred_len)
        self.bn = nn.BatchNorm1d(self.pred_len)


        self.cwt_enc = nn.Linear(1, moe_hidden_dim)
        self.fft_enc = nn.Linear(1, moe_hidden_dim)

        self.res_linear = nn.Linear(self.seq_len, self.pred_len)

        self.previous_x_emb = nn.Linear(self.seq_len, self.pred_len)

        self.cwt_t_enc = nn.Linear(self.look_back, self.pred_len)
        self.fft_t_enc = nn.Linear(self.look_back, self.pred_len)

        self.output_gate = GatingUnit(input_dim=self.channels)
        self.alignment_loss_fn = ExpertAlignmentLoss(mode='cosine')

    def forward(self, x, total_pow_diff, total_freqs):

        diff_outputs = []
        inputs = x

        prvious_x = x[:, :, :]  # (B, look_back, C)

        x = x[:, -self.look_back:, :]
        total_pow_diff = total_pow_diff[..., -self.look_back:, :]
        time_ebb = self.ebb(inputs[:, -self.pred_len:, :])
        time_ebb = time_ebb.repeat(x.shape[0], 1, 1)

        cwt_w_list = []
        fft_w_list = []

        for i, k in enumerate([1] + self.resolution):

            if k == 1:
                smooth_seq = x
            else:
                smooth_seq = smooth_conv(x, k, 'replicate')

            diff_smooth_seq = torch.diff(smooth_seq, dim=1, prepend=smooth_seq[:, :1])

            smooth_seq_fft, fft_bound, fft_weight, fft_expert = self.fft_moe(diff_smooth_seq.permute(0, 2, 1))   # smooth_seq_fft (B, C, T), fft_bound (expert_num - 1 + 2,), fft_weight (B, expert_num)
            smooth_seq_fft = self.fft_enc(smooth_seq_fft.permute(0, 2, 1)).permute(0, 2, 1)
            smooth_seq_fft = self.fft_t_enc(smooth_seq_fft)

            pow_diff, freqs = total_pow_diff[..., i], total_freqs[..., i]
            diff_cwt, cwt_bound, cwt_weight, cwt_expert = self.cwt_moe(pow_diff)
            diff_cwt_map = self.flatten_projector(diff_cwt)
            diff_cwt_map = self.cwt_enc(diff_cwt_map.permute(0, 2, 1)).permute(0, 2, 1)
            diff_cwt_map = self.cwt_t_enc(diff_cwt_map)

            cwt_w_list.append(cwt_weight.unsqueeze(-1))
            fft_w_list.append(fft_weight.unsqueeze(-1))

            attn_input = torch.cat([diff_cwt_map, smooth_seq_fft, time_ebb.permute(0, 2, 1)], dim=1)
            attn_input = attn_input.permute(0, 2, 1)

            final_diff_output = self.linear_out(torch.relu(attn_input))  # (B, T, C)

            diff_outputs.append(final_diff_output)


        output = self.resolution_fusion(diff_outputs)

        prvious_x = self.previous_x_emb(prvious_x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        output = output.permute(0, 2, 1)
        output =  output + x[:, -1:, :]

        output = self.output_gate(output, prvious_x)

        align_loss, diversity_loss = self.alignment_loss_fn(fft_expert, cwt_expert)

        cwt_w_out = torch.cat(cwt_w_list, dim=-1)
        fft_w_out = torch.cat(fft_w_list, dim=-1)

        return output, align_loss, diversity_loss, cwt_w_out, fft_w_out