import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import math

import functools
from .cbin import CBINorm2d
from .cin_cbin import CIN_CBINorm2d
from .cin import ConditionalInstanceNorm2d 
from .spectral import SpectralNorm


def get_norm_layer(layer_type='in', num_d=2, num_s=8):
    if 'in' in layer_type:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        if layer_type == 'cin':
            c_norm_layer = functools.partial(ConditionalInstanceNorm2d, num_d=num_d, affine=False)
        elif layer_type == 'cin_cbin':
            c_norm_layer = functools.partial(CIN_CBINorm2d, affine=False, num_d=num_d, num_s=num_s)
        else:
            c_norm_layer = functools.partial(CBINorm2d, affine=True, num_con=num_d+num_s)
            
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer


def get_act_layer(layer_type='relu'):  # get activation layer
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer


def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)

    return init_fun


class SNConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, pad_type='reflect', bias=True, norm_layer=None, act_layer=None):
        super(SNConv2d, self).__init__()
        
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)

        self.conv = SpectralNorm(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if act_layer is not None:
            self.activation = act_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))


class TrConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 bias=True, norm_layer=None, nl_layer=None):
        super(TrConv2dBlock, self).__init__()
        self.trConv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                         stride=stride, padding=padding, bias=bias)
        if norm_layer is not None:
            self.norm = norm_layer(out_planes)
        else:
            self.norm = lambda x: x

        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x

    def forward(self, x):
        return self.activation(self.norm(self.trConv(x)))


        # Augmenting Conv2d with Spectral Normalization
class Upsampling2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, type='Trp', norm_layer=None, nl_layer=None):
        super(Upsampling2dBlock, self).__init__()
        if type == 'Trp':
            self.upsample = TrConv2dBlock(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False, norm_layer=norm_layer, nl_layer=nl_layer)
        elif type == 'Ner':
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                          SNConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1,pad_type='reflect', bias=False, norm_layer=norm_layer, act_layer=nl_layer))
        else:
            raise ('None Upsampling type {}'.format(type))

    def forward(self, x):
        return self.upsample(x)


                    ########### Generator ############
class CResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, h_dim, c_norm_layer=None, act_layer=None):
        super(CResidualBlock, self).__init__()
        d = 1
        self.n1 = c_norm_layer(h_dim)
        self.a1 = act_layer()
        self.c1 = SNConv2d(h_dim, h_dim, kernel_size=3, stride=1, dilation=d, padding=d, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim)
        self.a2 = act_layer()
        self.c2 = SNConv2d(h_dim, h_dim, kernel_size=3, stride=1, dilation=d, padding=d, pad_type='reflect', bias=False)

    def forward(self, input):
        x, s = input[0], input[1]
        y = self.a1(self.n1(x, s))
        y = self.c1(y)
        y = self.a2(self.n2(y, s))
        y = self.c2(y)
        return [y+x, s]


class Generator(nn.Module):
    def __init__(self, ngf=64, ns=8, up_type='Trp', nd=2):
        super(Generator, self).__init__()
        
        norm_layer, c_norm_layer = get_norm_layer(layer_type='cbin', num_d=nd, num_s=ns)
        
        act_layer = get_act_layer(layer_type='relu')
        d = 1
        self.c = SNConv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, dilation=d, padding=d, pad_type='reflect', bias=False)

        self.b1 = CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, act_layer=act_layer)
        self.b2 = CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, act_layer=act_layer)
        self.b3 = CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, act_layer=act_layer)
        self.b4 = CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, act_layer=act_layer)
        self.b5 = CResidualBlock(ngf * 4, c_norm_layer=c_norm_layer, act_layer=act_layer)
        self.n3 = c_norm_layer(ngf * 4)
        self.a3 = act_layer()
        
        self.u1 = Upsampling2dBlock(ngf * 4, ngf * 2, type=up_type)
        self.u_n1 = c_norm_layer(ngf * 2)
        self.act1 = act_layer()
        self.u2 = Upsampling2dBlock(ngf * 2, ngf, type=up_type)
        self.u_n2 = c_norm_layer(ngf)
        self.act2 = act_layer()

        self.block = SNConv2d(ngf, 3, kernel_size=7, stride=1, padding=3, pad_type='reflect', bias=False, act_layer=nn.Tanh)

    def forward(self, x, s):
        x = self.c(x)

        x = self.b1([x, s])[0]
        x = self.b2([x, s])[0]
        x = self.b3([x, s])[0]
        x = self.b4([x, s])[0]
        x = self.b5([x, s])[0]
        
        x = self.a3(self.n3(x, s))
        
        x = self.act1(self.u_n1(self.u1(x), s))
        x = self.act2(self.u_n2(self.u2(x), s))
        x = self.block(x)
        return x


                    ########### Discirminator ############
class D_Cls_Net(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, num_class=2, num_scale=0):
        super(D_Cls_Net, self).__init__()
        norm_layer = None
        act_layer = get_act_layer('lrelu')    # Conv2dBlock, SNConv2d
        block = [SNConv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=False, norm_layer=norm_layer, act_layer=act_layer)]
        dim_in = ndf
        for n in range(1, block_num):
            dim_out = min(dim_in * 2, ndf * 8)
            block += [SNConv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1, bias=False, norm_layer=norm_layer, act_layer=act_layer)]
            dim_in = dim_out
        self.main = nn.Sequential(*block)

                # Projection Discriminator
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dis = SpectralNorm(nn.Linear(dim_in, 1, bias=True))
        self.embed = None 
        self.embed = nn.Embedding(num_class, dim_in)
                # Auxiliary Classifier
        kernel = 256 // (2 ** (1 + block_num + num_scale))
        self.cls = None
        self.cls = nn.Sequential(SNConv2d(dim_in, dim_in * 2, kernel_size=4, stride=2, padding=1, bias=False, norm_layer=norm_layer, act_layer=act_layer), SNConv2d(dim_in * 2, num_class, kernel_size=kernel, stride=1, padding=0, bias=True))

    def forward(self, x, c):
        x = self.main(x)
        y = self.global_pool(x).squeeze()
        pred = self.dis(y)
        if self.embed is not None:
            pred += torch.sum(y * self.embed(c), dim=1, keepdim=True)
            # print("  ### Using Projection Discriminator ###  ")
        if self.cls is not None:
            cls = self.cls(x)
        else:
            cls = None
        return pred, cls


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32, block_num=3, norm_type='in', nd=2):
        super(Discriminator, self).__init__()
        self.model_1 = D_Cls_Net(input_nc=input_nc, ndf=ndf, block_num=block_num, num_class=nd)

    def forward(self, x, c):
        pre1, cls1 = self.model_1(x, c)
        return [pre1], [cls1]


                    ######### Style Encoder ##########
def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [SpectralNorm(nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True))]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [SNConv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False, norm_layer=None, act_layer=None)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, out_dim, norm_layer=None, act_layer=None):
        super(BasicBlock, self).__init__()
        self.norm1 = norm_layer(input_dim)
        self.act1 = act_layer()
        self.conv1 = SNConv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False, norm_layer=None, act_layer=None)
        self.norm2 = norm_layer(input_dim)
        self.act2 = act_layer()
        self.cmp = convMeanpool(input_dim, out_dim)
        self.shortcut = meanpoolConv(input_dim, out_dim)

    def forward(self, input):
        x, c = input
        out = self.act1(self.norm1(x, c))
        out = self.conv1(out)
        out = self.act2(self.norm2(out, c))
        out = self.cmp(out)
        out += self.shortcut(x)
        return [out, c]


class Style(nn.Module):
    def __init__(self, output_nc=8, nef=64, nd=2, adding_CIN=True):
        super(Style, self).__init__()
        _, norm_layer = get_norm_layer(layer_type='cbin', num_d=nd, num_s=0)
        max_ndf = 4
        act_layer = get_act_layer(layer_type='relu')
        self.entry = SNConv2d(3, nef, kernel_size=4, stride=2, padding=1, bias=True)
        
        conv_layers = []
        for n in range(1, 4):
            input_ndf = nef * min(max_ndf, n)  # 2**(n-1)
            output_ndf = nef * min(max_ndf, n + 1)  # 2**n
            conv_layers += [BasicBlock(input_ndf, output_ndf, norm_layer, act_layer)]

        self.middle = nn.Sequential(*conv_layers)
        self.norm = norm_layer(output_ndf)
        self.exit = nn.Sequential(*[act_layer(), nn.AdaptiveAvgPool2d(1)])

        self.fc = nn.Sequential(*[SpectralNorm(nn.Linear(output_ndf, output_nc))])

    def forward(self, x, c):
        x = self.entry(x)
        x = self.middle([x, c])[0]
        x = self.norm(x, c)
        x_conv = self.exit(x)
        b = x_conv.size(0)
        x_conv = x_conv.view(b, -1)
        style = self.fc(x_conv)
        return style


                ######## Content encoder #########
class ResBlock(nn.Module):
    def __init__(self, h_dim, norm_layer=None, act_layer=None):
        super(ResBlock, self).__init__()
        d = 1
        self.c1 = SNConv2d(h_dim, h_dim, kernel_size=3, stride=1, dilation=d, padding=d, pad_type='reflect', bias=False)
        self.n1 = norm_layer(h_dim)
        self.a1 = act_layer()

        self.c2 = SNConv2d(h_dim, h_dim, kernel_size=3, stride=1, dilation=d, padding=d, pad_type='reflect', bias=False)
        self.n2 = norm_layer(h_dim)
        self.a2 = act_layer()

    def forward(self, x):
        y = self.c1(x)
        y = self.a1(self.n1(y))
        y = self.c2(y)
        y = self.a2(self.n2(y))
        return x + y


class Content(nn.Module):
    def __init__(self, dim, nd=2):
        super(Content, self).__init__()
        norm_layer, _ = get_norm_layer(layer_type='in')
        act_layer = get_act_layer(layer_type='relu')
        pad_type = 'reflect'

        self.c1 = SNConv2d(3, dim, kernel_size=7, stride=1, padding=3, pad_type=pad_type)
        self.n1 = norm_layer(dim)
        self.a1 = act_layer()
       
        # downsampling blocks
        self.c2 = SNConv2d(dim, 2 * dim, kernel_size=4, stride=2, padding=1, pad_type=pad_type)
        self.n2 = norm_layer(dim * 2)
        self.a2 = act_layer()
        dim *= 2
        self.c3 = SNConv2d(dim, 2 * dim, kernel_size=4, stride=2, padding=1, pad_type=pad_type)
        self.n3 = norm_layer(dim * 2)
        self.a3 = act_layer()
        dim *= 2

        # residual blocks
        self.res1 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)
        self.res2 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)
        self.res3 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)
        self.res4 = ResBlock(dim, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        x = self.a1(self.n1(self.c1(x)))
        
        x = self.c2(x)
        x = self.a2(self.n2(x))
        x = self.c3(x)
        x = self.a3(self.n3(x))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x
