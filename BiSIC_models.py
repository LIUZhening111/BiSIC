import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import ste_round
import torch.nn.functional as F
from lib.entropy_models import Hyperprior3D_LRP
from compressai.ans import BufferedRansEncoder, RansDecoder
from lib.ckbd_tools import ckbd_anchor, ckbd_nonanchor, ckbd_merge, ckbd_split_3D, compress_anchor, compress_nonanchor, decompress_anchor, decompress_nonanchor
from lib.bidirectional_attention import MutualAttention_master


class MaskedConv3d(nn.Conv3d):

    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, _, h, w = self.mask.size()
        self.mask[:, :, :, h // 2, w // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, :, h // 2 + 1 :] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        self.weight.data *= self.mask
        return super().forward(x)



class BiSIC_master(nn.Module):
    def __init__(self, N = 192, M = 192, Myatten = MutualAttention_master):
        super().__init__()
        self.num_slices = 12
        self.chan_per_slice = N // self.num_slices
        self.featuremap_channels = N
        self.context_prediction_list = nn.ModuleList(
            MaskedConv3d(self.chan_per_slice, self.chan_per_slice * 2, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 1, 1)) 
            for i in range(self.num_slices))
        self.Conv3DEncoder_A = nn.Sequential(
            nn.Conv3d(3, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            # DGDN_FuseH(N),
        )
        self.En_Transfer_A = Myatten(N)
        self.Conv3DEncoder_B = nn.Sequential(
            nn.Conv3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
        )
        self.En_Transfer_B = Myatten(N)
        self.hyperprior = Hyperprior3D_LRP(in_planes=N, mid_planes=M, out_planes=N*2)
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.chan_SAM = Myatten(self.chan_per_slice * 8)
        self.Conv3DDecoder_A = nn.Sequential(
            nn.ConvTranspose3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            # DGDN_FuseH(N, inverse=True),
        )
        self.De_Transfer_A = Myatten(N)
        self.Conv3DDecoder_B = nn.Sequential(
            nn.ConvTranspose3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(N, 3, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
        )
        self.De_Transfer_B = Myatten(N)
        self.entropy_parameters_zero_means = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 2, 384, 1),
            nn.GELU(),
            nn.Conv2d(384, 192, 1),
            nn.GELU(),
            nn.Conv2d(192, 96, 1),
            nn.GELU(),
            nn.Conv2d(96, self.chan_per_slice, 1),
        )
        self.entropy_parameters_zero_scales = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 2, 384, 1),
            nn.GELU(),
            nn.Conv2d(384, 192, 1),
            nn.GELU(),
            nn.Conv2d(192, 96, 1),
            nn.GELU(),
            nn.Conv2d(96, self.chan_per_slice, 1),
        )
        self.entropy_parameters_list_means = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 10, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 96, 1),
                nn.GELU(),
                nn.Conv2d(96, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        self.entropy_parameters_list_scales = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 10, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 96, 1),
                nn.GELU(),
                nn.Conv2d(96, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        self.channel_context_prediction_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.chan_per_slice * i, 224, 1),
                nn.GELU(),
                nn.Conv2d(224, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 128, 1),
                nn.GELU(),
                nn.Conv2d(128, self.chan_per_slice * 8, 1),
            ) for i in range(1, self.num_slices)
        )
        self.LRP_Transform_zero = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 64, 1),
                nn.GELU(),
                nn.Conv2d(64, self.chan_per_slice, 1)
        )
        self.LRP_Transform_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * (8 + 1), 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 64, 1),
                nn.GELU(),
                nn.Conv2d(64, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        
    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1) 
        x_mix = torch.cat([x_left.unsqueeze(2), x_right.unsqueeze(2)], dim=2)
        y_mix = self.Conv3DEncoder_A(x_mix)
        # (B, 192, 2, 16, 16)
        y_l_tmp, y_r_tmp = self.En_Transfer_A(y_mix[:, :, 0, :, :], y_mix[:, :, 1, :, :])
        y_mix = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        y_mix = self.Conv3DEncoder_B(y_mix)
        y_l_tmp, y_r_tmp = self.En_Transfer_B(y_mix[:, :, 0, :, :], y_mix[:, :, 1, :, :])
        y_mix = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        
        # with LRP, we need to calculate means and scales in one and then split them into left and right
        hyper_params_means, hyper_params_scales, z_left_likelihoods, z_right_likelihoods, z_left_hat, z_right_hat = self.hyperprior(y_mix, out_z=True)
        hyper_left_params_means, hyper_right_params_means = torch.split(hyper_params_means, 1, dim=2)
        hyper_left_params_scales, hyper_right_params_scales = torch.split(hyper_params_scales, 1, dim=2)
        hyper_left_params_means, hyper_right_params_means = hyper_left_params_means.squeeze(2), hyper_right_params_means.squeeze(2)
        hyper_left_params_scales, hyper_right_params_scales = hyper_left_params_scales.squeeze(2), hyper_right_params_scales.squeeze(2)
        y_left, y_right = torch.split(y_mix, 1, dim=2)  # (24, 384, 1, 16, 16)
        y_left = y_left.squeeze(2)
        y_right = y_right.squeeze(2)
        y_raw_left_slices = y_left.chunk(self.num_slices, dim=1)
        y_raw_right_slices = y_right.chunk(self.num_slices, dim=1)
        y_left_noise = self.gaussian_conditional.quantize(y_left, "noise" if self.training else "dequantize")
        y_right_noise = self.gaussian_conditional.quantize(y_right, "noise" if self.training else "dequantize")
        y_left_noise_slices = y_left_noise.chunk(self.num_slices, dim=1)
        y_right_noise_slices = y_right_noise.chunk(self.num_slices, dim=1)
        y_left_hat_slices = []
        y_right_hat_slices = []
        y_left_likelihoods = []
        y_right_likelihoods = []
        
        for slice_idx, [y_left_slice, y_right_slice] in enumerate(zip(y_left_noise_slices, y_right_noise_slices)):
            y_raw_left_slice = y_raw_left_slices[slice_idx]
            y_raw_right_slice = y_raw_right_slices[slice_idx]
            if slice_idx == 0:
                # dependency on hyperprior and context
                ctx_inone = torch.cat([y_left_slice.unsqueeze(2), y_right_slice.unsqueeze(2)], dim=2)
                ctx_joint = self.context_prediction_list[slice_idx](ctx_inone)
                ctx_left = ctx_joint[:, :, 0, :, :]
                ctx_right = ctx_joint[:, :, 1, :, :]
                means_left = self.entropy_parameters_zero_means(torch.cat([hyper_left_params_means, ctx_left], dim=1))
                scales_left = self.entropy_parameters_zero_scales(torch.cat([hyper_left_params_scales, ctx_left], dim=1))
                means_right = self.entropy_parameters_zero_means(torch.cat([hyper_right_params_means, ctx_right], dim=1))
                scales_right = self.entropy_parameters_zero_scales(torch.cat([hyper_right_params_scales, ctx_right], dim=1))
                
                _, y_slice_likelihood_left = self.gaussian_conditional(y_raw_left_slice, scales_left, means=means_left)
                _, y_slice_likelihood_right = self.gaussian_conditional(y_raw_right_slice, scales_right, means=means_right)
                y_left_likelihoods.append(y_slice_likelihood_left)
                y_right_likelihoods.append(y_slice_likelihood_right)
                y_left_slice_STE = ste_round(y_raw_left_slice - means_left) + means_left
                y_right_slice_STE = ste_round(y_raw_right_slice - means_right) + means_right
                LRP_left = 0.5 * torch.tanh(self.LRP_Transform_zero(torch.cat([hyper_left_params_means, y_left_slice_STE], dim=1)))
                LRP_right = 0.5 * torch.tanh(self.LRP_Transform_zero(torch.cat([hyper_right_params_means, y_right_slice_STE], dim=1)))
                y_left_slice_STE = y_left_slice_STE + LRP_left
                y_right_slice_STE = y_right_slice_STE + LRP_right
                y_left_hat_slices.append(y_left_slice_STE)
                y_right_hat_slices.append(y_right_slice_STE)
            else:
                # dependency on hyperprior, context and channel
                channel_ctx_left = self.channel_context_prediction_list[slice_idx-1](torch.cat(y_left_hat_slices, dim=1))
                channel_ctx_right = self.channel_context_prediction_list[slice_idx-1](torch.cat(y_right_hat_slices, dim=1))
                channel_ctx_left, channel_ctx_right = self.chan_SAM(channel_ctx_left, channel_ctx_right)
                ctx_inone = torch.cat([y_left_slice.unsqueeze(2), y_right_slice.unsqueeze(2)], dim=2)
                ctx_joint = self.context_prediction_list[slice_idx](ctx_inone)
                ctx_left = ctx_joint[:, :, 0, :, :]
                ctx_right = ctx_joint[:, :, 1, :, :]
                means_left = self.entropy_parameters_list_means[slice_idx-1](torch.cat([hyper_left_params_means, ctx_left, channel_ctx_left], dim=1))
                scales_left = self.entropy_parameters_list_scales[slice_idx-1](torch.cat([hyper_left_params_scales, ctx_left, channel_ctx_left], dim=1))
                means_right = self.entropy_parameters_list_means[slice_idx-1](torch.cat([hyper_right_params_means, ctx_right, channel_ctx_right], dim=1))
                scales_right = self.entropy_parameters_list_scales[slice_idx-1](torch.cat([hyper_right_params_scales, ctx_right, channel_ctx_right], dim=1))

                _, y_slice_likelihood_left = self.gaussian_conditional(y_raw_left_slice, scales_left, means=means_left)
                _, y_slice_likelihood_right = self.gaussian_conditional(y_raw_right_slice, scales_right, means=means_right)
                y_left_likelihoods.append(y_slice_likelihood_left)
                y_right_likelihoods.append(y_slice_likelihood_right)
                y_left_slice_STE = ste_round(y_raw_left_slice - means_left) + means_left
                y_right_slice_STE = ste_round(y_raw_right_slice - means_right) + means_right
                LRP_left = 0.5 * torch.tanh(self.LRP_Transform_list[slice_idx-1](torch.cat([hyper_left_params_means, channel_ctx_left, y_left_slice_STE], dim=1)))
                LRP_right = 0.5 * torch.tanh(self.LRP_Transform_list[slice_idx-1](torch.cat([hyper_right_params_means, channel_ctx_right, y_right_slice_STE], dim=1)))
                y_left_slice_STE = y_left_slice_STE + LRP_left
                y_right_slice_STE = y_right_slice_STE + LRP_right
                y_left_hat_slices.append(y_left_slice_STE)
                y_right_hat_slices.append(y_right_slice_STE)
        
        y_left_hat_deco = torch.cat(y_left_hat_slices, dim=1)
        y_right_hat_deco = torch.cat(y_right_hat_slices, dim=1)
        y_likelihoods_left_inone = torch.cat(y_left_likelihoods, dim=1)
        y_likelihoods_right_inone = torch.cat(y_right_likelihoods, dim=1)
        
        y_l_tmp, y_r_tmp = self.De_Transfer_A(y_left_hat_deco, y_right_hat_deco)
        y_mix_deco = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        y_mix_deco = self.Conv3DDecoder_A(y_mix_deco)
        y_l_tmp, y_r_tmp = self.De_Transfer_B(y_mix_deco[:, :, 0, :, :], y_mix_deco[:, :, 1, :, :])
        y_mix_deco = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        x_mix_deco = self.Conv3DDecoder_B(y_mix_deco)
        
        
        x_left_hat, x_right_hat = torch.split(x_mix_deco, 1, dim=2)
        x_left_hat = x_left_hat.squeeze(2)
        x_right_hat = x_right_hat.squeeze(2)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [{"y": y_likelihoods_left_inone, "z": z_left_likelihoods}, {"y":y_likelihoods_right_inone, "z":z_right_likelihoods}],
            "feature": [y_left_hat_deco, y_right_hat_deco, z_left_hat, z_right_hat]
        }
        
        
    
    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated



class BiSIC_Fast(nn.Module):
    def __init__(self, N = 192, M = 192, Myatten = MutualAttention_master):
        super().__init__()
        self.num_slices = 6
        self.chan_per_slice = N // self.num_slices
        self.featuremap_channels = N
        self.context_prediction_CKBD_list = nn.ModuleList(
            nn.Conv3d(self.chan_per_slice, self.chan_per_slice * 2, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 1, 1)) 
            for i in range(self.num_slices))
        self.Conv3DEncoder_A = nn.Sequential(
            nn.Conv3d(3, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
        )
        self.En_Transfer_A = Myatten(N)
        self.Conv3DEncoder_B = nn.Sequential(
            nn.Conv3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GELU(),
            nn.Conv3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
        )
        self.En_Transfer_B = Myatten(N)
        self.hyperprior = Hyperprior3D_LRP(in_planes=N, mid_planes=M, out_planes=N*2)
        self.gaussian_conditional = GaussianConditional(None)
        self.M = M
        self.chan_SAM = Myatten(self.chan_per_slice * 8)
        self.Conv3DDecoder_A = nn.Sequential(
            nn.ConvTranspose3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
        )
        self.De_Transfer_A = Myatten(N)
        self.Conv3DDecoder_B = nn.Sequential(
            nn.ConvTranspose3d(N, N, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(N, 3, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
        )
        self.De_Transfer_B = Myatten(N)
        self.entropy_params_anchor_zero_means = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2, 384, 1),
            nn.GELU(),
            nn.Conv2d(384, 192, 1),
            nn.GELU(),
            nn.Conv2d(192, 96, 1),
            nn.GELU(),
            nn.Conv2d(96, self.chan_per_slice, 1),
        )
        self.entropy_params_anchor_zero_scales = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2, 384, 1),
            nn.GELU(),
            nn.Conv2d(384, 192, 1),
            nn.GELU(),
            nn.Conv2d(192, 96, 1),
            nn.GELU(),
            nn.Conv2d(96, self.chan_per_slice, 1),
        )
        self.entropy_params_nonanchor_zero_means = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 2, 384, 1),
            nn.GELU(),
            nn.Conv2d(384, 192, 1),
            nn.GELU(),
            nn.Conv2d(192, 96, 1),
            nn.GELU(),
            nn.Conv2d(96, self.chan_per_slice, 1),
        )
        self.entropy_params_nonanchor_zero_scales = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 2, 384, 1),
            nn.GELU(),
            nn.Conv2d(384, 192, 1),
            nn.GELU(),
            nn.Conv2d(192, 96, 1),
            nn.GELU(),
            nn.Conv2d(96, self.chan_per_slice, 1),
        )
        self.entropy_params_anchor_list_means = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 8, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 96, 1),
                nn.GELU(),
                nn.Conv2d(96, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        self.entropy_params_anchor_list_scales = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 8, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 96, 1),
                nn.GELU(),
                nn.Conv2d(96, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        self.entropy_params_nonanchor_list_means = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 10, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 96, 1),
                nn.GELU(),
                nn.Conv2d(96, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        self.entropy_params_nonanchor_list_scales = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * 10, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 96, 1),
                nn.GELU(),
                nn.Conv2d(96, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        self.channel_context_prediction_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.chan_per_slice * i, 224, 1),
                nn.GELU(),
                nn.Conv2d(224, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 128, 1),
                nn.GELU(),
                nn.Conv2d(128, self.chan_per_slice * 8, 1),
            ) for i in range(1, self.num_slices)
        )
        self.LRP_Transform_zero = nn.Sequential(
            nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice, 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 64, 1),
                nn.GELU(),
                nn.Conv2d(64, self.chan_per_slice, 1)
        )
        self.LRP_Transform_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.featuremap_channels * 2 + self.chan_per_slice * (8 + 1), 384, 1),
                nn.GELU(),
                nn.Conv2d(384, 192, 1),
                nn.GELU(),
                nn.Conv2d(192, 64, 1),
                nn.GELU(),
                nn.Conv2d(64, self.chan_per_slice, 1),
            ) for i in range(1, self.num_slices)
        )
        
    def forward(self, x):
        x_left, x_right = x[0], x[1] #x.chunk(2, 1) 
        x_mix = torch.cat([x_left.unsqueeze(2), x_right.unsqueeze(2)], dim=2)
        y_mix = self.Conv3DEncoder_A(x_mix)
        # (B, 192, 2, 16, 16)
        y_l_tmp, y_r_tmp = self.En_Transfer_A(y_mix[:, :, 0, :, :], y_mix[:, :, 1, :, :])
        y_mix = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        y_mix = self.Conv3DEncoder_B(y_mix)
        y_l_tmp, y_r_tmp = self.En_Transfer_B(y_mix[:, :, 0, :, :], y_mix[:, :, 1, :, :])
        y_mix = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        
        hyper_params_means, hyper_params_scales, z_left_likelihoods, z_right_likelihoods, z_left_hat, z_right_hat = self.hyperprior(y_mix, out_z=True)
        hyper_left_params_means, hyper_right_params_means = torch.split(hyper_params_means, 1, dim=2)
        hyper_left_params_scales, hyper_right_params_scales = torch.split(hyper_params_scales, 1, dim=2)
        hyper_left_params_means, hyper_right_params_means = hyper_left_params_means.squeeze(2), hyper_right_params_means.squeeze(2)
        hyper_left_params_scales, hyper_right_params_scales = hyper_left_params_scales.squeeze(2), hyper_right_params_scales.squeeze(2)
        y_left, y_right = torch.split(y_mix, 1, dim=2)  # (24, 384, 1, 16, 16)
        y_left = y_left.squeeze(2)
        y_right = y_right.squeeze(2)
        y_raw_left_slices = y_left.chunk(self.num_slices, dim=1)
        y_raw_right_slices = y_right.chunk(self.num_slices, dim=1)
        y_left_noise = self.gaussian_conditional.quantize(y_left, "noise" if self.training else "dequantize")
        y_right_noise = self.gaussian_conditional.quantize(y_right, "noise" if self.training else "dequantize")
        y_left_noise_slices = y_left_noise.chunk(self.num_slices, dim=1)
        y_right_noise_slices = y_right_noise.chunk(self.num_slices, dim=1)
        y_left_hat_slices = []
        y_right_hat_slices = []
        y_left_likelihoods = []
        y_right_likelihoods = []
        
        for slice_idx, [y_left_slice, y_right_slice] in enumerate(zip(y_left_noise_slices, y_right_noise_slices)):
            anchor_3D, nonanchor_3D = ckbd_split_3D(torch.cat([y_left_slice.unsqueeze(2), y_right_slice.unsqueeze(2)], dim=2))
            y_raw_left_slice = y_raw_left_slices[slice_idx]
            y_raw_right_slice = y_raw_right_slices[slice_idx]
            if slice_idx == 0:
                # anchor part: dependency on hyperprior
                means_anchor_left = self.entropy_params_anchor_zero_means(hyper_left_params_means)
                scales_anchor_left = self.entropy_params_anchor_zero_scales(hyper_left_params_scales)
                means_anchor_right = self.entropy_params_anchor_zero_means(hyper_right_params_means)
                scales_anchor_right = self.entropy_params_anchor_zero_scales(hyper_right_params_scales)
                means_anchor_left, scales_anchor_left = ckbd_anchor(means_anchor_left), ckbd_anchor(scales_anchor_left)
                means_anchor_right, scales_anchor_right = ckbd_anchor(means_anchor_right), ckbd_anchor(scales_anchor_right)
                # non-anchor part: dependency on hyperprior and anchor
                local_ctx_CKBD = self.context_prediction_CKBD_list[slice_idx](anchor_3D)
                ctx_left = local_ctx_CKBD[:, :, 0, :, :]
                ctx_right = local_ctx_CKBD[:, :, 1, :, :]
                means_nonanchor_left = self.entropy_params_nonanchor_zero_means(torch.cat([hyper_left_params_means, ctx_left], dim=1))
                scales_nonanchor_left = self.entropy_params_nonanchor_zero_scales(torch.cat([hyper_left_params_scales, ctx_left], dim=1))
                means_nonanchor_right = self.entropy_params_nonanchor_zero_means(torch.cat([hyper_right_params_means, ctx_right], dim=1))
                scales_nonanchor_right = self.entropy_params_nonanchor_zero_scales(torch.cat([hyper_right_params_scales, ctx_right], dim=1))
                means_nonanchor_left, scales_nonanchor_left = ckbd_nonanchor(means_nonanchor_left), ckbd_nonanchor(scales_nonanchor_left)
                means_nonanchor_right, scales_nonanchor_right = ckbd_nonanchor(means_nonanchor_right), ckbd_nonanchor(scales_nonanchor_right)
                # merge and return
                means_left = ckbd_merge(means_anchor_left, means_nonanchor_left)
                scales_left = ckbd_merge(scales_anchor_left, scales_nonanchor_left)
                means_right = ckbd_merge(means_anchor_right, means_nonanchor_right)
                scales_right = ckbd_merge(scales_anchor_right, scales_nonanchor_right)
                
                _, y_slice_likelihood_left = self.gaussian_conditional(y_raw_left_slice, scales_left, means=means_left)
                _, y_slice_likelihood_right = self.gaussian_conditional(y_raw_right_slice, scales_right, means=means_right)
                y_left_likelihoods.append(y_slice_likelihood_left)
                y_right_likelihoods.append(y_slice_likelihood_right)
                y_left_slice_STE = ste_round(y_raw_left_slice - means_left) + means_left
                y_right_slice_STE = ste_round(y_raw_right_slice - means_right) + means_right
                # y_left_hat_slices.append(y_left_slice)
                # y_right_hat_slices.append(y_right_slice)
                LRP_left = 0.5 * torch.tanh(self.LRP_Transform_zero(torch.cat([hyper_left_params_means, y_left_slice_STE], dim=1)))
                LRP_right = 0.5 * torch.tanh(self.LRP_Transform_zero(torch.cat([hyper_right_params_means, y_right_slice_STE], dim=1)))
                y_left_slice_STE = y_left_slice_STE + LRP_left
                y_right_slice_STE = y_right_slice_STE + LRP_right
                y_left_hat_slices.append(y_left_slice_STE)
                y_right_hat_slices.append(y_right_slice_STE)
            else:
                # anchor part: dependency on hyperprior and channel context
                channel_ctx_left = self.channel_context_prediction_list[slice_idx-1](torch.cat(y_left_hat_slices, dim=1))
                channel_ctx_right = self.channel_context_prediction_list[slice_idx-1](torch.cat(y_right_hat_slices, dim=1))
                channel_ctx_left, channel_ctx_right = self.chan_SAM(channel_ctx_left, channel_ctx_right)
                means_anchor_left = self.entropy_params_anchor_list_means[slice_idx-1](torch.cat([hyper_left_params_means, channel_ctx_left], dim=1))
                scales_anchor_left = self.entropy_params_anchor_list_scales[slice_idx-1](torch.cat([hyper_left_params_scales, channel_ctx_left], dim=1))
                means_anchor_right = self.entropy_params_anchor_list_means[slice_idx-1](torch.cat([hyper_right_params_means, channel_ctx_right], dim=1))
                scales_anchor_right = self.entropy_params_anchor_list_scales[slice_idx-1](torch.cat([hyper_right_params_scales, channel_ctx_right], dim=1))
                means_anchor_left, scales_anchor_left = ckbd_anchor(means_anchor_left), ckbd_anchor(scales_anchor_left)
                means_anchor_right, scales_anchor_right = ckbd_anchor(means_anchor_right), ckbd_anchor(scales_anchor_right)
                # non-anchor part: dependency on hyperprior, channel context and anchor
                local_ctx_CKBD = self.context_prediction_CKBD_list[slice_idx](anchor_3D)
                ctx_left = local_ctx_CKBD[:, :, 0, :, :]
                ctx_right = local_ctx_CKBD[:, :, 1, :, :]
                means_nonanchor_left = self.entropy_params_nonanchor_list_means[slice_idx-1](torch.cat([hyper_left_params_means, ctx_left, channel_ctx_left], dim=1))
                scales_nonanchor_left = self.entropy_params_nonanchor_list_scales[slice_idx-1](torch.cat([hyper_left_params_scales, ctx_left, channel_ctx_left], dim=1))
                means_nonanchor_right = self.entropy_params_nonanchor_list_means[slice_idx-1](torch.cat([hyper_right_params_means, ctx_right, channel_ctx_right], dim=1))
                scales_nonanchor_right = self.entropy_params_nonanchor_list_scales[slice_idx-1](torch.cat([hyper_right_params_scales, ctx_right, channel_ctx_right], dim=1))
                means_nonanchor_left, scales_nonanchor_left = ckbd_nonanchor(means_nonanchor_left), ckbd_nonanchor(scales_nonanchor_left)
                means_nonanchor_right, scales_nonanchor_right = ckbd_nonanchor(means_nonanchor_right), ckbd_nonanchor(scales_nonanchor_right)
                # merge and return
                means_left = ckbd_merge(means_anchor_left, means_nonanchor_left)
                scales_left = ckbd_merge(scales_anchor_left, scales_nonanchor_left)
                means_right = ckbd_merge(means_anchor_right, means_nonanchor_right)
                scales_right = ckbd_merge(scales_anchor_right, scales_nonanchor_right)

                _, y_slice_likelihood_left = self.gaussian_conditional(y_raw_left_slice, scales_left, means=means_left)
                _, y_slice_likelihood_right = self.gaussian_conditional(y_raw_right_slice, scales_right, means=means_right)
                y_left_likelihoods.append(y_slice_likelihood_left)
                y_right_likelihoods.append(y_slice_likelihood_right)
                y_left_slice_STE = ste_round(y_raw_left_slice - means_left) + means_left
                y_right_slice_STE = ste_round(y_raw_right_slice - means_right) + means_right
                # y_left_hat_slices.append(y_left_slice)
                # y_right_hat_slices.append(y_right_slice)
                LRP_left = 0.5 * torch.tanh(self.LRP_Transform_list[slice_idx-1](torch.cat([hyper_left_params_means, channel_ctx_left, y_left_slice_STE], dim=1)))
                LRP_right = 0.5 * torch.tanh(self.LRP_Transform_list[slice_idx-1](torch.cat([hyper_right_params_means, channel_ctx_right, y_right_slice_STE], dim=1)))
                y_left_slice_STE = y_left_slice_STE + LRP_left
                y_right_slice_STE = y_right_slice_STE + LRP_right
                y_left_hat_slices.append(y_left_slice_STE)
                y_right_hat_slices.append(y_right_slice_STE)
        
        y_left_hat_deco = torch.cat(y_left_hat_slices, dim=1)
        y_right_hat_deco = torch.cat(y_right_hat_slices, dim=1)
        y_likelihoods_left_inone = torch.cat(y_left_likelihoods, dim=1)
        y_likelihoods_right_inone = torch.cat(y_right_likelihoods, dim=1)
        
        # y_mix_deco = torch.cat([y_left_hat_deco.unsqueeze(2), y_right_hat_deco.unsqueeze(2)], dim=2)
        y_l_tmp, y_r_tmp = self.De_Transfer_A(y_left_hat_deco, y_right_hat_deco)
        y_mix_deco = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        y_mix_deco = self.Conv3DDecoder_A(y_mix_deco)
        y_l_tmp, y_r_tmp = self.De_Transfer_B(y_mix_deco[:, :, 0, :, :], y_mix_deco[:, :, 1, :, :])
        y_mix_deco = torch.cat([y_l_tmp.unsqueeze(2), y_r_tmp.unsqueeze(2)], dim=2)
        x_mix_deco = self.Conv3DDecoder_B(y_mix_deco)
        
        
        x_left_hat, x_right_hat = torch.split(x_mix_deco, 1, dim=2)
        x_left_hat = x_left_hat.squeeze(2)
        x_right_hat = x_right_hat.squeeze(2)
        
        return {
            "x_hat": [x_left_hat, x_right_hat],
            "likelihoods": [{"y": y_likelihoods_left_inone, "z": z_left_likelihoods}, {"y":y_likelihoods_right_inone, "z":z_right_likelihoods}],
            "feature": [y_left_hat_deco, y_right_hat_deco, z_left_hat, z_right_hat]
        }
        
    
    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.hyperprior.entropy_bottleneck,
            "hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.hyperprior.entropy_bottleneck.update(force=force)
        return updated

