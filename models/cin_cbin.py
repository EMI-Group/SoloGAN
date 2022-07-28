import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F


class _CIN_CBINorm(_BatchNorm):  # num_con means conditional code length, while num_features means number of input channel
    def __init__(self, num_features, num_d=2, num_s=8, eps=1e-5, momentum=0.1, affine=False):
        super(_CIN_CBINorm, self).__init__(num_features, eps, momentum, affine)
        
        self.scale = nn.Embedding(num_d, num_features)
        self.shift = nn.Embedding(num_d, num_features)
        
        self.weight = None
        self.bias = None
        
        self.ConBias = nn.Sequential(nn.Linear(num_s, num_features), nn.Tanh())    ## each layer has a separate condition embedding
        
                # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input, domain_info, style_info):
        b, c = input.size(0), input.size(1)    # batch and channel 
        
        weight = self.scale(domain_info).view(b, c, 1, 1)
        bias = self.shift(domain_info).view(b, c, 1, 1)
        
        # Repeat stored stats and affine transform params
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        
        tarBias = self.ConBias(style_info).view(b, c, 1, 1)    # The condition is linearly projected to each layer's gain and biases  
        
        input_reshaped = input.contiguous().view(1, b * c, *input.size()[2:])
        
        out = F.batch_norm(input_reshaped, running_mean, running_var, None, None, True, self.momentum, self.eps)
        
        # Reshape back
        self.running_mean.copy_(running_mean.view(b, c).mean(0, keepdim=False))
        self.running_var.copy_(running_var.view(b, c).mean(0, keepdim=False))

        return (out.view(b, c, *input.size()[2:]) + tarBias) * weight + bias

    def eval(self):
        return self


class CIN_CBINorm2d(_CIN_CBINorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))
        super(CIN_CBINorm2d, self)._check_input_dim(input)