'''
/* ===========================================================================
** Copyright (C) 2019 Infineon Technologies AG. All rights reserved.
** ===========================================================================
**
** ===========================================================================
** Infineon Technologies AG (INFINEON) is supplying this file for use
** exclusively with Infineon's sensor products. This file can be freely
** distributed within development tools and software supporting such 
** products.
** 
** THIS SOFTWARE IS PROVIDED "AS IS".  NO WARRANTIES, WHETHER EXPRESS, IMPLIED
** OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
** MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE.
** INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR DIRECT, INDIRECT, 
** INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON 
** WHATSOEVER.
** ===========================================================================
*/
'''
"""
This is the mobilenetv2 implementation for 3D CNN architecture
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ptflops import get_model_complexity_info
import time
#from gpu_mem_track import MemTracker
import gc

class conv_bn(nn.Module):
    def __init__(self, inp, oup, des = 'first'):
        super(conv_bn, self).__init__()
        if des == 'first':
            self.convbn = nn.Sequential(
                nn.Conv3d(inp, oup, kernel_size=3, stride= (1,2,2), padding=(1,1,1), bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU6(inplace=True)
            )
        else:
            self.convbn = nn.Sequential(
                nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU6(inplace=True)
            )
    def forward(self, input):
        return self.convbn(input)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, hiden, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = hiden
        self.cc = 0
        self.ccc = 0
        #hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace = True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace = True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if  self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        del x
        gc.collect()
        torch.cuda.empty_cache()
        return out

class InvertedResidual_normal(nn.Module):
    def __init__(self, inp, oup, stride, hiden, expand_ratio):
        super(InvertedResidual_normal, self).__init__()
        self.stride = stride
       # hidden_dim = hiden
        self.cc = 0
        self.ccc = 0
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace = True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace = True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if  self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, sample_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32     #32
        last_channel = 1280     #1280
        interverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1,1,1)],
            [2,  24, 2, (1,2,2)],
            [2,  32, 3, (2,2,2)],
            [2,  64, 4, (2,2,2)],
            [2,  96, 3, (1,1,1)],
            [2, 160, 3, (2,2,2)],
            [2, 320, 1, (1,1,1)],
        ]
       # hiden = [6, 19, 28, 28, 38, 38, 38, 76, 76, 76, 76,115,115,115,191,191,191]
        hiden = [32,96,144,144,192,192,192,384,384,384,384,576,576,576,960,960,960]
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(2, input_channel, des= 'first')]   #ddd
        cnt = 0
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1,1,1)
                self.features.append(block(input_channel, output_channel, stride, hiden[cnt],expand_ratio=t))
                input_channel = output_channel
                cnt += 1
        # building last several layers
        self.features.append(conv_bn(input_channel, self.last_channel, des='last'))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
            #parameters.append({'params': v, 'requires_grad': False})

    return parameters

    
def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MobileNetV2(**kwargs)
    return model


if __name__ == "__main__":
    """Testing
    """
        input1 =Variable(torch.Tensor(1,2,8,112,112)).cuda()
        model = get_model(num_classes=15, sample_size=112, width_mult=1)

        model = model.cuda()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (total/1e6))
    #FLOPs,params = get_model_complexity_info(model, (2,16, 112,112),as_strings = True, print_per_layer_stat = True)
 #   print('Flops:{}'.format(FLOPs))
 #   print('params:'+ params)
   # for k, v in model.state_dict().items() :
          #          print(k)

