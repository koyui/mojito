import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mojito.losses.utils import zipf_loss

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, mu, **kwargs):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu
        self.use_zipf = kwargs["use_zipf"]

        self.reset_codebook()
        if self.use_zipf:
            self.make_zipf()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).to(device))

    def make_zipf(self):
        ranks = torch.arange(1, self.nb_code + 1)
        weights = torch.reciprocal(ranks + 2.7)
        weights /= weights.sum()
        self.register_buffer("zipf_dist", weights, persistent=False)

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        k_w = self.codebook.t()

        distance = torch.sum(x**2, dim=-1, keepdim=True) - 2*torch.matmul(x, k_w) + torch.sum(k_w**2, dim=0,keepdim=True)  # (N * L, b) 
        _, code_idx = torch.min(distance, dim=-1)
        
        # calculate code frequency
        normed_freq = None
        if self.use_zipf:
            freq =  F.gumbel_softmax(torch.neg(distance), tau=0.1, hard=False)   # (N * L, nb_code)
            normed_freq = torch.sum(freq, dim=0) / distance.shape[0]  # (nb_code, ) and normalized
            
        return code_idx, normed_freq

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx, freq = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # passthrough
        x_d = x + (x_d - x).detach()

        # postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        additional_loss = dict()
        
        if self.use_zipf:
            additional_loss['zipf_loss'] = zipf_loss(freq, self.zipf_dist)
        
        return x_d, commit_loss, perplexity, additional_loss, freq