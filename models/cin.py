import torch 
from torch import nn 
import torch.nn.functional as F
from torch.autograd import Variable 

class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_d=2, eps=1e-5, momentum=0.1, affine=False):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.scale = nn.Embedding(num_d, num_features)
        self.shift = nn.Embedding(num_d, num_features)
        
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, y):
        
        self.weight = self.scale(y)
        self.bias = self.shift(y)
        
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling Conditional Instance Normalization!"
        
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance normalization
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'