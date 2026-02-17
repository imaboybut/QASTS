import torch
import numpy as np

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

def clip(x, eps):
    x_clip = torch.clamp(x, min=eps)
    return x - x.detach() + x_clip.detach()

class LsqQuantizerWeightWithSampling(torch.nn.Module):
    def __init__(self, bit, all_positive=False, per_channel=True, learnable=True, 
                 sampling_rate=1.0, **kwargs):
        """
        LSQ Weight Quantizer with Rate Sampling
        
        Args:
            bit: quantization bit width
            all_positive: whether to use unsigned quantization
            per_channel: whether to use per-channel quantization
            learnable: whether scale factor is learnable
            sampling_rate: ratio of weights to quantize (0.0 to 1.0)
        """
        super(LsqQuantizerWeightWithSampling, self).__init__()
        
        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        
        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        self.sampling_rate = sampling_rate
        self.register_parameter('s', None)
        self.initialized_alpha = False
    
    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            if len(x.shape) == 4:
                self.s = torch.nn.Parameter(x.detach().abs().mean(dim=list(range(1, len(x.shape))), keepdim=True) * 2 / self.thd_pos)
            else:
                self.s = torch.nn.Parameter(x.detach().abs().mean(dim=1, keepdim=True) * 2 / self.thd_pos)
        else:
            self.s = torch.nn.Parameter(x.detach().abs().mean() * 2 / self.thd_pos)
        self.initialized_alpha = True
    
    def forward(self, x):
        if not self.initialized_alpha:
            self.init_from(x)
        
        eps = 1e-7
        if self.learnable:
            s = clip(self.s, eps)
        else:
            s = self.s
        
        # Apply rate sampling
        if self.training and self.sampling_rate < 1.0:
            # Create binary mask for sampling
            mask = torch.rand_like(x) < self.sampling_rate
            
            # Quantize only sampled weights
            s_grad_scale = ((self.thd_pos * x.numel()) / (self.thd_pos - self.thd_neg)) ** 0.5
            s_scale = grad_scale(s, s_grad_scale)
            
            # Apply quantization to sampled weights
            x_quantized = torch.zeros_like(x)
            if mask.any():
                x_sampled = x * mask
                x_sampled_q = torch.clamp(x_sampled / s_scale, self.thd_neg, self.thd_pos)
                x_sampled_q = round_pass(x_sampled_q)
                x_sampled_q = x_sampled_q * s_scale
                x_quantized = x_sampled_q * mask + x * (~mask)
            else:
                x_quantized = x  # No quantization if nothing sampled
            
            return x_quantized
        else:
            # Full quantization during inference or when sampling_rate=1.0
            s_grad_scale = ((self.thd_pos * x.numel()) / (self.thd_pos - self.thd_neg)) ** 0.5
            s_scale = grad_scale(s, s_grad_scale)
            
            x = torch.clamp(x / s_scale, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
            
            return x

class LsqQuantizerActWithSampling(torch.nn.Module):
    def __init__(self, bit, all_positive=False, per_channel=False, learnable=True,
                 sampling_rate=1.0, **kwargs):
        """
        LSQ Activation Quantizer with Rate Sampling
        
        Args:
            bit: quantization bit width
            all_positive: whether to use unsigned quantization
            per_channel: whether to use per-channel quantization
            learnable: whether scale factor is learnable
            sampling_rate: ratio of activations to quantize (0.0 to 1.0)
        """
        super(LsqQuantizerActWithSampling, self).__init__()
        
        if all_positive:
            if bit == 1:
                self.thd_neg = 0
                self.thd_pos = 1
            else:
                # unsigned activation is quantized to [0, 2^b-1]
                self.thd_neg = 0
                self.thd_pos = 2 ** bit - 1
        else:
            if bit == 1:
                self.thd_neg = -1
                self.thd_pos = 1
            else:
                # signed activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        
        self.bit = bit
        self.per_channel = per_channel
        self.all_positive = all_positive
        self.learnable = learnable
        self.sampling_rate = sampling_rate
        self.s = torch.nn.Parameter(torch.tensor([1.0]))
    
    def forward(self, x):
        eps = 1e-7
        if self.learnable:
            s = clip(self.s, eps)
        else:
            s = self.s
        
        # Apply rate sampling
        if self.training and self.sampling_rate < 1.0:
            # Create binary mask for sampling
            mask = torch.rand_like(x) < self.sampling_rate
            
            # Quantize only sampled activations
            s_grad_scale = ((self.thd_pos * x.numel()) / (self.thd_pos - self.thd_neg)) ** 0.5
            s_scale = grad_scale(s, s_grad_scale)
            
            # Apply quantization to sampled activations
            x_quantized = torch.zeros_like(x)
            if mask.any():
                x_sampled = x * mask
                x_sampled_q = torch.clamp(x_sampled / s_scale, self.thd_neg, self.thd_pos)
                x_sampled_q = round_pass(x_sampled_q)
                x_sampled_q = x_sampled_q * s_scale
                x_quantized = x_sampled_q * mask + x * (~mask)
            else:
                x_quantized = x  # No quantization if nothing sampled
            
            return x_quantized
        else:
            # Full quantization during inference or when sampling_rate=1.0
            s_grad_scale = ((self.thd_pos * x.numel()) / (self.thd_pos - self.thd_neg)) ** 0.5
            s_scale = grad_scale(s, s_grad_scale)
            
            x = torch.clamp(x / s_scale, self.thd_neg, self.thd_pos)
            x = round_pass(x)
            x = x * s_scale
            
            return x