# MIT License
# 
# Copyright (c) 2018 Junyu Gao
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""PA_CAE passenger-counting network."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

def initialize_weights(models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, std=0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Module):
        for mini_m in m.children():
            real_init_weights(mini_m)
    else:
        print(m)

class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttWithSpatial(nn.Module):

    def __init__(self, inp, oup, reduction=32):
        super(CoordAttWithSpatial, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_s = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_s = torch.cat([avg_out, max_out], dim=1)
        a_s = self.conv_s(x_s).sigmoid()
        out = identity * a_w * a_h * a_s
        return out

class convDU(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(1, 9)):
        super(convDU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)), nn.ReLU(inplace=True))

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        for i in range(h):
            i_fea = fea.select(2, i).unsqueeze(2)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)
        for i in range(h):
            pos = h - i - 1
            if pos == h - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
        fea = torch.cat(fea_stack, 2)
        return fea

class convLR(nn.Module):

    def __init__(self, in_out_channels=2048, kernel_size=(9, 1)):
        super(convLR, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_out_channels, in_out_channels, kernel_size, stride=1, padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)), nn.ReLU(inplace=True))

    def forward(self, fea):
        n, c, h, w = fea.size()
        fea_stack = []
        for i in range(w):
            i_fea = fea.select(3, i).unsqueeze(3)
            if i == 0:
                fea_stack.append(i_fea)
                continue
            fea_stack.append(self.conv(fea_stack[i - 1]) + i_fea)
        for i in range(w):
            pos = w - i - 1
            if pos == w - 1:
                continue
            fea_stack[pos] = self.conv(fea_stack[pos + 1]) + fea_stack[pos]
        fea = torch.cat(fea_stack, 3)
        return fea

class PA_CAE(nn.Module):

    def __init__(self, pretrained=True):
        super(PA_CAE, self).__init__()
        self.seen = 0
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = []
        self.coordatt = CoordAttWithSpatial(inp=64, oup=64)
        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.convDU = convDU(in_out_channels=64, kernel_size=(1, 9))
        self.convLR = convLR(in_out_channels=64, kernel_size=(9, 1))
        self.output_layer = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.ReLU())
        initialize_weights(self.modules())
        res = models.resnet101(pretrained=pretrained)
        self.frontend = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2)
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

    def forward(self, x):
        x = self.frontend(x)
        x = self.own_reslayer_3(x)
        x = self.backend(x)
        x1 = x
        x2 = x
        x2 = self.coordatt(x2)
        x1 = self.convDU(x1)
        x1 = self.convLR(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.output_layer(x)
        x = F.upsample(x, scale_factor=8)
        return x

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

