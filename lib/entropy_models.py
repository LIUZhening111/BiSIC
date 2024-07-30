import torch
import math
from torch import nn
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import ste_round
import torch.nn.functional as F


class Hyperprior3D_LRP(CompressionModel):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 384):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        # super().__init__()
        self.hyper_encoder = nn.Sequential(
            nn.Conv3d(in_planes, mid_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_planes, mid_planes, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_planes, mid_planes, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
        )
        self.hyper_decoder_means = nn.Sequential(
            nn.ConvTranspose3d(mid_planes, mid_planes, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(in_planes, mid_planes, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.hyper_decoder_scales = nn.Sequential(
            nn.ConvTranspose3d(mid_planes, mid_planes, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(in_planes, mid_planes, kernel_size=(3, 5, 5), stride=(1, 2, 2), output_padding=(0, 1, 1), padding=(1, 2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )

    def forward(self, y, out_z=False):
        z_mix = self.hyper_encoder(y)
        z_left, z_right = torch.split(z_mix, 1, dim=2)  # (B, 192, 1, 4, 4)
        z_left, z_right = z_left.squeeze(2), z_right.squeeze(2)  # (B, 192, 4, 4)
        # left
        _, z_left_likelihoods = self.entropy_bottleneck(z_left)
        z_left_offset = self.entropy_bottleneck._get_medians()
        z_left_hat = ste_round(z_left - z_left_offset) + z_left_offset
        # right
        _, z_right_likelihoods = self.entropy_bottleneck(z_right)
        z_right_offset = self.entropy_bottleneck._get_medians()
        z_right_hat = ste_round(z_right - z_right_offset) + z_right_offset
        z_mix_hat = torch.cat([z_left_hat.unsqueeze(2), z_right_hat.unsqueeze(2)], dim=2)
        params_means = self.hyper_decoder_means(z_mix_hat)
        params_scales = self.hyper_decoder_scales(z_mix_hat)
        if out_z:
            return params_means, params_scales, z_left_likelihoods, z_right_likelihoods, z_left_hat, z_right_hat
        else:
            return params_means, params_scales, z_left_likelihoods, z_right_likelihoods

    def compress(self, y):
        z_mix = self.hyper_encoder(y)
        z_left, z_right = torch.split(z_mix, 1, dim=2)  # (B, 192, 1, 4, 4)
        z_left, z_right = z_left.squeeze(2), z_right.squeeze(2)  # (B, 192, 4, 4)
        z_strings_left = self.entropy_bottleneck.compress(z_left)
        z_left_hat = self.entropy_bottleneck.decompress(z_strings_left, z_left.size()[-2:])
        z_strings_right = self.entropy_bottleneck.compress(z_right)
        z_right_hat = self.entropy_bottleneck.decompress(z_strings_right, z_right.size()[-2:])
        z_mix_hat = torch.cat([z_left_hat.unsqueeze(2), z_right_hat.unsqueeze(2)], dim=2)
        params_means = self.hyper_decoder_means(z_mix_hat)
        params_scales = self.hyper_decoder_scales(z_mix_hat)
        return params_means, params_scales, z_mix_hat, z_strings_left, z_strings_right #{"strings": z_string, "shape": z.size()[-2:]}

    def decompress(self, string_left, string_right, shape):
        z_left_hat = self.entropy_bottleneck.decompress(string_left, shape)
        z_right_hat = self.entropy_bottleneck.decompress(string_right, shape)
        z_mix_hat = torch.cat([z_left_hat.unsqueeze(2), z_right_hat.unsqueeze(2)], dim=2)
        params_means = self.hyper_decoder_means(z_mix_hat)
        params_scales = self.hyper_decoder_scales(z_mix_hat)
        return params_means, params_scales, z_mix_hat
