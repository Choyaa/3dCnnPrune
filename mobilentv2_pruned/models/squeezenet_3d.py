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
This is the squeezenet implementation for 3D CNN architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)
        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)

        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        out = self.relu(out)

        return out


class SqueezeNet(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,
    	         version=1.1,
    	         num_classes=400):
        super(SqueezeNet, self).__init__()
        if not version == 1.1:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "Only 1.1 expected".format(version=version))
        self.num_classes = num_classes

        last_duration = int(math.ceil(sample_duration / 8))
        last_size = int(math.ceil(sample_size / 32))

        cfg = [64, 16,64,64,16,64,64,32,128,128,32,128,128,48,192,192,48,192,192,64,256,256,64,256,256]
      #  cfg = [22,11,28,27,12,3,49,29,2,27,27,3,47,43,3,11,44,3,12,46,3,13,58,125,255]
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv3d(2, cfg[0], kernel_size=3, stride=(1,2,2), padding=(1,1,1)),
                nn.BatchNorm3d(cfg[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1),
                Fire(cfg[0], cfg[1], cfg[2], cfg[3]),
                Fire(cfg[2]+cfg[3], cfg[4], cfg[5], cfg[6], use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(cfg[5]+cfg[6], cfg[7], cfg[8], cfg[9]),
                Fire(cfg[8]+cfg[9], cfg[10], cfg[11], cfg[12], use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(cfg[11]+cfg[12],cfg[13], cfg[14], cfg[15]),
                Fire(cfg[14]+cfg[15], cfg[16], cfg[17], cfg[18], use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(cfg[17]+cfg[18], cfg[19], cfg[20], cfg[21]),
                Fire(cfg[20]+cfg[21], cfg[22], cfg[23], cfg[24], use_bypass=True),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv3d(cfg[23]+cfg[24], self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.8),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0),-1)


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = SqueezeNet(**kwargs)
    return model




if __name__ == '__main__':
    model = SqueezeNet(version=1.1, sample_size = 112, sample_duration = 32, num_classes=27)
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = torch.randn(8, 2, 32, 112, 112).cuda()
    output = model(input_var)
    total = sum([param.nelement() for param in model.parameters()])
    print("%.2fM" %(total/1e6))
    print(output.shape)
