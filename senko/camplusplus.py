# https://github.com/modelscope/3D-Speaker/tree/main/speakerlab/models/campplus

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as cp

class CAMPPlus(nn.Module):
   def __init__(self, feat_dim=80, embedding_size=512, growth_rate=32, bn_size=4, init_channels=128, memory_efficient=True):
       super(CAMPPlus, self).__init__()
       self.head = FCM(feat_dim=feat_dim)
       channels = self.head.out_channels

       self.xvector = nn.Sequential(OrderedDict([
           ('tdnn', TDNNLayer(channels, init_channels, 5, stride=2, dilation=1, padding=-1))
       ]))
       
       channels = init_channels
       blocks_config = [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
       
       for i, (num_layers, kernel_size, dilation) in enumerate(blocks_config):
           block = CAMDenseTDNNBlock(num_layers=num_layers, in_channels=channels, out_channels=growth_rate, bn_channels=bn_size * growth_rate, kernel_size=kernel_size, dilation=dilation, bias=False, memory_efficient=memory_efficient)
           self.xvector.add_module(f'block{i+1}', block)
           channels = channels + num_layers * growth_rate
           
           self.xvector.add_module(f'transit{i+1}', TransitLayer(channels, channels // 2, False))
           channels //= 2

       out_nonlinear = nn.Sequential()
       out_nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
       out_nonlinear.add_module('relu', nn.ReLU(inplace=True))
       self.xvector.add_module('out_nonlinear', out_nonlinear)
       
       self.xvector.add_module('stats', StatsPool())
       self.xvector.add_module('dense', DenseLayer(channels * 2, embedding_size, False))

       for m in self.modules():
           if isinstance(m, (nn.Conv1d, nn.Linear)):
               nn.init.kaiming_normal_(m.weight.data)
               if m.bias is not None: nn.init.zeros_(m.bias)

   def forward(self, x):
       x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
       x = self.head(x)
       x = self.xvector(x)
       return x

class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FCM(nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = nn.Sequential(
            block(self.in_planes, m_channels, stride=2),
            block(m_channels, m_channels, stride=1)
        )
        self.in_planes = m_channels * block.expansion

        self.layer2 = nn.Sequential(
            block(self.in_planes, m_channels, stride=2),
            block(m_channels, m_channels, stride=1)
        )

        self.conv2 = nn.Conv2d(m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1]*shape[2], shape[3])
        return out


class StatsPool(nn.Module):
    def forward(self, x, eps=1e-2):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=True) if x.size(-1) > 1 else torch.zeros_like(mean) + eps
        return torch.cat([mean, std], dim=-1)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
            
        self.linear = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        self.nonlinear = nn.Sequential()
        self.nonlinear.add_module('batchnorm', nn.BatchNorm1d(out_channels))
        self.nonlinear.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.nonlinear(self.linear(x))


class CAMLayer(nn.Module):
    def __init__(self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(bn_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1) 
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        return y * self.sigmoid(self.linear2(self.relu(self.linear1(context))))

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True) if stype == 'avg' else F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        return seg.unsqueeze(-1).expand(*seg.shape, seg_len).reshape(*seg.shape[:-1], -1)[..., :x.shape[-1]]


class CAMDenseTDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(kernel_size)
        self.memory_efficient = memory_efficient
        
        self.nonlinear1 = nn.Sequential()
        self.nonlinear1.add_module('batchnorm', nn.BatchNorm1d(in_channels))
        self.nonlinear1.add_module('relu', nn.ReLU(inplace=True))
        
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        
        self.nonlinear2 = nn.Sequential()
        self.nonlinear2.add_module('batchnorm', nn.BatchNorm1d(bn_channels))
        self.nonlinear2.add_module('relu', nn.ReLU(inplace=True))
        
        padding = (kernel_size - 1) // 2 * dilation
        self.cam_layer = CAMLayer(bn_channels, out_channels, kernel_size, stride, padding, dilation, bias)

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(lambda x: self.linear1(self.nonlinear1(x)), x)
        else:
            x = self.linear1(self.nonlinear1(x))
        return self.cam_layer(self.nonlinear2(x))


class CAMDenseTDNNBlock(nn.ModuleList):
   def __init__(self, num_layers, in_channels, out_channels, bn_channels, kernel_size, stride=1, dilation=1, bias=False, memory_efficient=False):
       super(CAMDenseTDNNBlock, self).__init__()
       for i in range(num_layers):
           layer = CAMDenseTDNNLayer(in_channels + i * out_channels, out_channels, bn_channels, kernel_size, stride, dilation, bias, memory_efficient)
           self.add_module(f'tdnnd{i+1}', layer)

   def forward(self, x):
       for layer in self:
           x = torch.cat([x, layer(x)], dim=1)
       return x


class TransitLayer(nn.Module):
   def __init__(self, in_channels, out_channels, bias=True):
       super(TransitLayer, self).__init__()
       self.nonlinear = nn.Sequential()
       self.nonlinear.add_module('batchnorm', nn.BatchNorm1d(in_channels))
       self.nonlinear.add_module('relu', nn.ReLU(inplace=True))
       self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

   def forward(self, x):
       return self.linear(self.nonlinear(x))


class DenseLayer(nn.Module):
   def __init__(self, in_channels, out_channels, bias=False):
       super(DenseLayer, self).__init__()
       self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
       self.nonlinear = nn.Sequential()
       self.nonlinear.add_module('batchnorm', nn.BatchNorm1d(out_channels, affine=False))
       self.nonlinear.add_module('relu', nn.ReLU(inplace=True))

   def forward(self, x):
       return self.nonlinear(self.linear(x if len(x.shape) == 3 else x.unsqueeze(-1)).squeeze(-1) if len(x.shape) == 2 else self.linear(x))