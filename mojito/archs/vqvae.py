# Partially from https://github.com/Mael-zys/T2M-GPT

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from typing import Union

from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset

class VQVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        code_num: int = 512,
        code_dim: int = 512,
        output_emb_width: int = 512,
        down_t: int = 3,
        stride_t: int = 2,
        width: int = 512,
        depth: int = 3,
        dilation_growth_rate: int = 3,
        activation: str = "relu",
        **kwargs):
        super(VQVae, self).__init__()

        self.code_dim = code_dim
        self.use_zipf = kwargs["use_zipf"]

        self.encoder = Encoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=None
        )
        self.decoder = Decoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=None
        )
        self.quantizer = QuantizeEMAReset(
            code_num,
            code_dim,
            mu=0.99,
            use_zipf=self.use_zipf
        )

    def preprocess(self, x):
        # (bs, T, D) -> (bs, D, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, D, T) ->  (bs, T, D)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: torch.Tensor):
        # quantized encoder
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, commit_loss, perplexity, additional_loss, freq = self.quantizer(x_encoder)
            
        # decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, x_quantized.permute(0, 2, 1), commit_loss, perplexity, additional_loss, freq

    def encode(self, features: torch.Tensor) -> Union[torch.Tensor, Distribution]:
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)

        return code_idx, None

    def decode(self, z: torch.Tensor):
        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out

    def quantize_only(self, features: torch.Tensor):
        # quantized encoder
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        
        x_quantized, _, _, _, freq = self.quantizer(x_encoder)
        
        return x_quantized.permute(0, 2, 1), freq

class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm),
                nn.Upsample(scale_factor=stride_t, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
