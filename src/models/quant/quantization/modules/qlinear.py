import imp
from numpy import clip
import torch
import torch.nn as nn

from .qbias import LearnableBias, LearnableBias4img
from ..quantizer.statsq import StatsQuantizer
from ...deit_vision_transformer import Mlp
try:
    from timm.models.layers import to_2tuple
except ImportError:
    # Fallback if timm is not available
    def to_2tuple(x):
        if isinstance(x, (list, tuple)):
            return x
        return (x, x)
from ..quantizer.lsq import LsqQuantizer, LsqQuantizerWeight, LsqQuantizer4img, LsqQuantizer4Conv2d, LsqQuantizer4head_input

class LSQ_input(nn.Module):
    def __init__(self, bit=2,all_positive=False, learnable = True, learanbaleBiasdim = 192, device=None):
        super().__init__()
        self.input_bits = bit
        self.all_positive = all_positive
        self.learnable = learnable
        self.input_quant_fn = LsqQuantizer(bit=bit,all_positive=all_positive, learnable = learnable)
        self.move_b4 = LearnableBias(learanbaleBiasdim)
        self.move_aft = LearnableBias(learanbaleBiasdim)
        if device:
            self.to(device)

    def forward(self, input):
        
        input = self.move_b4(input)
        input = self.input_quant_fn(input)
        input = self.move_aft(input)
        return input
    
class QLinear(nn.Linear):

    def __init__(self, *kargs, m: torch.nn.Linear, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        super(QLinear, self).__init__(m.in_features, m.out_features,bias=True)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.aq_learnable = aq_learnable
        self.wq_learnable = wq_learnable
        self.symmetric = symmetric
        self.weight_channelwise = weight_channelwise # not gonna used atm
        self.input_channelwise = input_channelwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.input_quant_fn = LsqQuantizer(bit=input_bits,all_positive=(symmetric==False), learnable =  aq_learnable).to(m.weight.device)
        self.pretrained_initialized = pretrained_initialized
        if pretrained_initialized != False:
            # CRITICAL FIX: Copy data directly
            self.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                self.bias.data.copy_(m.bias.detach())
        if weight_quant_method == 'statsq':
            self.statsq_fn = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable).to(m.weight.device)
            # StatsQ initializes itself on first forward pass
        else:
            raise ValueError("Unknown quant_method")        

        self.move_b4 = LearnableBias(self.weight.shape[1], device=m.weight.device)
        self.move_aft = LearnableBias(self.weight.shape[1], device=m.weight.device)

    def forward(self, input):

        # quantize weight
        if self.weight_quant_method == 'statsq':
            weight = self.statsq_fn(self.weight)
        else:
            raise ValueError("Unknown quant_method")    
        # quantize input
        # TESTING: Disable LearnableBias to see if it's causing issues
        # input = self.move_b4(input)
        input = self.input_quant_fn(input)
        # input = self.move_aft(input)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self):
        return (
            f"act_bit={self.input_bits}, "
            f"weight_bit={self.weight_bits}, "
            f"act_all_positive={not self.symmetric}, "
            f"wq_learnable={self.wq_learnable}, "
            f"aq_learnable={self.aq_learnable}, "
            f"weight_channelwise ={self.weight_channelwise}, "
            f"input_channelwise ={self.input_channelwise}, "
            f"weight_quant_method={self.weight_quant_method}, "
            f"activation_quant_method={self.input_quant_method}, "
            f"pretrained_initialized = {self.pretrained_initialized}"
        )

class QMLP(Mlp):
    def __init__(self, *kargs, m: Mlp, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq", act_layer=nn.GELU,
                    pretrained_initialized = False,
                    **kwargs):
            # Handle different drop attribute names
            if hasattr(m, 'drop'):
                drop_prob = m.drop.p
            elif hasattr(m, 'drop1'):
                drop_prob = m.drop1.p
            else:
                drop_prob = 0.0
                
            super().__init__(
                in_features = m.fc1.in_features,
                hidden_features = m.fc1.out_features,
                out_features = m.fc2.out_features,
                drop = drop_prob
            )
                
            out_features = m.fc2.out_features or m.fc1.in_features
            hidden_features = m.fc1.out_features or m.fc1.in_features
            drop_probs = to_2tuple(self.drop)

            self.fc1 = QLinear(m = m.fc1,weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=True,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)

            self.act_layer = act_layer
            
            if act_layer != 'rprelu':
                if act_layer != 'None':
                    self.act = act_layer()
                else:
                    self.act = nn.Identity()
                

            self.drop1 = nn.Dropout(drop_probs[0])
            self.fc2 = QLinear(m = m.fc2,weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=False,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)
            self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        
        x = self.fc1(x)
        if self.act_layer != 'rprelu':
            x = self.act(x)
        else:
            x = self.move1(x)
            x = self.act(x)
            x = self.move2(x)

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LSQ_QConv2d(nn.Conv2d):
    def __init__(self, *kargs, m: torch.nn.Conv2d, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="lsq", input_quant_method="lsq",
                 pretrained_initialized = False,input_fdim=128, input_tdim=1024,
                 **kwargs):
        super(LSQ_QConv2d, self).__init__(in_channels = m.in_channels, out_channels = m.out_channels, kernel_size = m.kernel_size
                                          , stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups, bias=True)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.aq_learnable = aq_learnable
        self.wq_learnable = wq_learnable
        self.symmetric = symmetric
        self.weight_channelwise = weight_channelwise # not gonna used atm
        self.input_channelwise = input_channelwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.input_quant_fn = LsqQuantizer4img(bit=input_bits,all_positive=(symmetric==False), learnable =  aq_learnable).to(m.weight.device)
        self.pretrained_initialized = pretrained_initialized
        if pretrained_initialized != False:
            # CRITICAL FIX: Copy data directly
            self.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                self.bias.data.copy_(m.bias.detach())
    
        self.lsqw_fn = LsqQuantizer4Conv2d(bit=self.weight_bits, learnable=aq_learnable).to(m.weight.device)
        # Initialize quantizer scale from pretrained weights
        # WARNING: Do NOT call init_from here - it modifies the weights!
        # The scale will be initialized on first forward pass

        self.move_b4 = LearnableBias4img(input_fdim * input_tdim, device=m.weight.device)
        self.move_aft = LearnableBias4img(input_fdim * input_tdim, device=m.weight.device)

    def forward(self, input):
        # quantize weight
        weight = self.lsqw_fn(self.weight)
  
        # quantize input
        # TESTING: Disable LearnableBias to see if it's causing issues
        # input = self.move_b4(input)
        input = self.input_quant_fn(input)
        # input = self.move_aft(input)
        out = nn.functional.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return out

    def extra_repr(self):
        return (
            f"act_bit={self.input_bits}, "
            f"weight_bit={self.weight_bits}, "
            f"act_all_positive={not self.symmetric}, "
            f"wq_learnable={self.wq_learnable}, "
            f"aq_learnable={self.aq_learnable}, "
            f"weight_channelwise ={self.weight_channelwise}, "
            f"input_channelwise ={self.input_channelwise}, "
            f"weight_quant_method={self.weight_quant_method}, "
            f"activation_quant_method={self.input_quant_method}, "
            f"pretrained_initialized = {self.pretrained_initialized}"
        )

class LSQ_QLinear4head(nn.Linear):

    def __init__(self, *kargs, m: torch.nn.Linear, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        super(LSQ_QLinear4head, self).__init__(m.in_features, m.out_features,bias=True)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.aq_learnable = aq_learnable
        self.wq_learnable = wq_learnable
        self.symmetric = symmetric
        self.weight_channelwise = weight_channelwise # not gonna used atm
        self.input_channelwise = input_channelwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.input_quant_fn = LsqQuantizer4head_input(bit=input_bits,all_positive=(symmetric==False), learnable =  aq_learnable).to(m.weight.device)
        self.pretrained_initialized = pretrained_initialized
        if pretrained_initialized != False:
            # CRITICAL FIX: Copy data directly
            self.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                self.bias.data.copy_(m.bias.detach())
        if weight_quant_method == 'lsq':
            self.lsqw_fn = LsqQuantizerWeight(bit=self.weight_bits, per_channel=weight_channelwise ,learnable=wq_learnable).to(m.weight.device)
            # Initialize quantizer scale from pretrained weights
            # WARNING: Do NOT call init_from here - it modifies the weights!
            # The scale will be initialized on first forward pass
        elif weight_quant_method == 'statsq':
            self.statsq_fn = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable).to(m.weight.device)
            # StatsQ initializes itself on first forward pass
        else:
            raise ValueError(f"Unknown quant_method: {weight_quant_method}")        

        self.move_b4 = LearnableBias(self.weight.shape[1], device=m.weight.device)
        self.move_aft = LearnableBias(self.weight.shape[1], device=m.weight.device)

    def forward(self, input):

        # quantize weight
        if self.weight_quant_method == 'lsq':
            weight = self.lsqw_fn(self.weight)
        elif self.weight_quant_method == 'statsq':
            weight = self.statsq_fn(self.weight)
        else:
            raise ValueError(f"Unknown quant_method: {self.weight_quant_method}")    
        # quantize input
        # TESTING: Disable LearnableBias to see if it's causing issues
        # input = self.move_b4(input)
        input = self.input_quant_fn(input)
        # input = self.move_aft(input)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self):
        return (
            f"act_bit={self.input_bits}, "
            f"weight_bit={self.weight_bits}, "
            f"act_all_positive={not self.symmetric}, "
            f"wq_learnable={self.wq_learnable}, "
            f"aq_learnable={self.aq_learnable}, "
            f"weight_channelwise ={self.weight_channelwise}, "
            f"input_channelwise ={self.input_channelwise}, "
            f"weight_quant_method={self.weight_quant_method}, "
            f"activation_quant_method={self.input_quant_method}, "
            f"pretrained_initialized = {self.pretrained_initialized}"
        )

class LSQ_w_and_act_QLinear(nn.Linear):

    def __init__(self, *kargs, m: torch.nn.Linear, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                 symmetric=True, weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq",
                 pretrained_initialized = False,
                 **kwargs):
        super(LSQ_w_and_act_QLinear, self).__init__(m.in_features, m.out_features,bias=True)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.aq_learnable = aq_learnable
        self.wq_learnable = wq_learnable
        self.symmetric = symmetric
        self.weight_channelwise = weight_channelwise # not gonna used atm
        self.input_channelwise = input_channelwise
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.input_quant_fn = LsqQuantizer(bit=input_bits,all_positive=(symmetric==False), learnable =  aq_learnable).to(m.weight.device)
        self.pretrained_initialized = pretrained_initialized
        if pretrained_initialized != False:
            # CRITICAL FIX: Copy data directly instead of creating new parameters
            self.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                self.bias.data.copy_(m.bias.detach())
        if weight_quant_method == 'lsq':
            self.lsqw_fn = LsqQuantizerWeight(bit=self.weight_bits, per_channel=weight_channelwise ,learnable=wq_learnable).to(m.weight.device)
            # Initialize quantizer scale from pretrained weights
            # WARNING: Do NOT call init_from here - it modifies the weights!
            # The scale will be initialized on first forward pass
        elif weight_quant_method == 'statsq':
            self.statsq_fn = StatsQuantizer(num_bits=self.weight_bits, clip_learnable=wq_learnable).to(m.weight.device)
            # StatsQ initializes itself on first forward pass
        else:
            raise ValueError(f"Unknown quant_method: {weight_quant_method}")        

        self.move_b4 = LearnableBias(self.weight.shape[1], device=m.weight.device)
        self.move_aft = LearnableBias(self.weight.shape[1], device=m.weight.device)

    def forward(self, input):

        # quantize weight
        if self.weight_quant_method == 'lsq':
            weight = self.lsqw_fn(self.weight)
        elif self.weight_quant_method == 'statsq':
            weight = self.statsq_fn(self.weight)
        else:
            raise ValueError(f"Unknown quant_method: {self.weight_quant_method}")    

        # quantize input
        # TESTING: Disable LearnableBias to see if it's causing issues
        # input = self.move_b4(input)
        input = self.input_quant_fn(input)
        # input = self.move_aft(input)
        out = nn.functional.linear(input, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def extra_repr(self):
        return (
            f"act_bit={self.input_bits}, "
            f"weight_bit={self.weight_bits}, "
            f"act_all_positive={not self.symmetric}, "
            f"wq_learnable={self.wq_learnable}, "
            f"aq_learnable={self.aq_learnable}, "
            f"weight_channelwise ={self.weight_channelwise}, "
            f"input_channelwise ={self.input_channelwise}, "
            f"weight_quant_method={self.weight_quant_method}, "
            f"activation_quant_method={self.input_quant_method}, "
            f"pretrained_initialized = {self.pretrained_initialized}"
        )

class LSQ_w_and_act_QMLP(Mlp):
    def __init__(self, *kargs, m: Mlp, weight_bits=8, input_bits=8, aq_learnable=True, wq_learnable = True,
                    weight_channelwise=True, input_channelwise=True, weight_quant_method="statsq", input_quant_method="lsq", act_layer=nn.GELU,
                    pretrained_initialized = False,
                    **kwargs):

            # Handle different drop attribute names
            if hasattr(m, 'drop'):
                drop_prob = m.drop.p
            elif hasattr(m, 'drop1'):
                drop_prob = m.drop1.p
            else:
                drop_prob = 0.0
                
            super().__init__(
                in_features = m.fc1.in_features,
                hidden_features = m.fc1.out_features,
                out_features = m.fc2.out_features,
                drop = drop_prob
            )
                
            out_features = m.fc2.out_features or m.fc1.in_features
            hidden_features = m.fc1.out_features or m.fc1.in_features
            drop_probs = to_2tuple(self.drop)

            self.fc1 = LSQ_w_and_act_QLinear(m = m.fc1,weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=True,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)

            self.act_layer = act_layer
            
            if act_layer != 'rprelu':
                if act_layer != 'None':
                    self.act = act_layer()
                else:
                    self.act = nn.Identity()
                

            self.drop1 = nn.Dropout(drop_probs[0])
            self.fc2 = LSQ_w_and_act_QLinear(m = m.fc2,weight_bits=weight_bits,input_bits=input_bits,
            aq_learnable=aq_learnable,wq_learnable=wq_learnable,symmetric=False,weight_channelwise=weight_channelwise,input_channelwise=input_channelwise,
            weight_quant_method=weight_quant_method,input_quant_method=input_quant_method, pretrained_initialized = pretrained_initialized)
            self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        
        x = self.fc1(x)
        if self.act_layer != 'rprelu':
            x = self.act(x)
        else:
            x = self.move1(x)
            x = self.act(x)
            x = self.move2(x)

        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x





