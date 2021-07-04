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

import torch
from torch import nn
from models import squeezenet_3d, mobilenetv2_3d
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import numpy as np
from gpu_mem_track import MemTracker
from opts import parse_opts

def generate_model(opt,cfg = None):
    assert opt.model in ['squeezenet', 'mobilenetv2']
    if opt.model == 'squeezenet':
        model = squeezenet_3d.get_model(
                version=opt.version,
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                cfg = cfg)

    else:
        model = mobilenetv2_3d.get_model(
            num_classes=opt.n_classes,
            sample_size=opt.sample_size,
            width_mult=opt.width_mult)

    if opt.modality == 'IR' or opt.modality == 'D':
        dim_new = 1
    elif opt.modality == 'IRD':
        dim_new = 2
    else:
        dim_new = 3
    ## change the first conv layer in the model
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d), list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (dim_new,) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
    new_conv = nn.Conv3d(dim_new, conv_layer.out_channels,
                         conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                         bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data  # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

    # replace the first convolutional layer
    setattr(container, layer_name, new_conv)
    print('Convert the first layer to %d channels.'%dim_new)

    if not opt.no_cuda:
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
       # print(device)
        model = model.cuda()
        # model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            pretrain = torch.load(opt.pretrain_path, map_location = 'cuda:0')
            assert opt.arch == pretrain['arch']

            if opt.same_modality_finetune:
                model.load_state_dict(pretrain['state_dict'])
                print('loaded pretrained model {}'.format(opt.pretrain_path))
            else:
                pretrained_state_dict = pretrain['state_dict']
                pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'module.features.0' not in k}
                model_dict = model.state_dict()
                model_dict.update(pretrained_state_dict)
                model.load_state_dict(model_dict)
                print('loaded pretrained model {}'.format(opt.pretrain_path))

            ##  The last layer needs to be changed
            if opt.model == 'mobilenetv2':
                l = list(model.module.classifier.modules())[-1]  # this is the last layer in classifier
                model.module.classifier = nn.Sequential(nn.Dropout(0.2),
                                                        nn.Linear(l.in_features, opt.n_finetune_classes))
                model.module.classifier = model.classifier.cuda()
                parameters = model.parameters()

            else:
                conv_l = list(model.classifier.modules())[
                    2]  # this is the last conv layer in the classifier that should be modified
                avg_pool = list(model.classifier.modules())[
                    -1]  # this is the last average pooling layer in the classifier
                model.classifier = nn.Sequential(nn.Dropout(p=0.8),
                                                        nn.Conv3d(conv_l.in_channels, opt.n_finetune_classes,
                                                                  kernel_size=conv_l.kernel_size),
                                                        nn.ReLU(inplace=True),
                                                        avg_pool
                                                        )
                model.classifier = model.classifier.cuda()
                parameters = model.parameters()


    else:
        if opt.pretrain_path:
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            if opt.same_modality_finetune:
                model.load_state_dict(pretrain['state_dict'])
                print('loaded pretrained model {}'.format(opt.pretrain_path))
            else:
                pretrained_state_dict = pretrain['state_dict']
                pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'module.features.0' not in k}
                model_dict = model.state_dict()
                model_dict.update(pretrained_state_dict)
                model.load_state_dict(model_dict)
                print('loaded pretrained model {}'.format(opt.pretrain_path))

            ##  The last layer needs to be changed
            if opt.model == 'mobilenetv2':
                l = list(model.module.classifier.modules())[-1]  # this is the last layer in classifier
                model.module.classifier = nn.Sequential(nn.Dropout(0.2),
                                                        nn.Linear(l.in_features, opt.n_finetune_classes))
                model.module.classifier = model.module.classifier.cuda()
                parameters = model.parameters()

            else:
                conv_l = list(model.module.classifier.modules())[
                    2]  # this is the last conv layer in the classifier that should be modified
                avg_pool = list(model.module.classifier.modules())[
                    -1]  # this is the last average pooling layer in the classifier
                model.module.classifier = nn.Sequential(nn.Dropout(p=0.8),
                                                        nn.Conv3d(conv_l.in_channels, opt.n_finetune_classes,
                                                                  kernel_size=conv_l.kernel_size),
                                                        nn.ReLU(inplace=True),
                                                        avg_pool
                                                        )
                model.module.classifier = model.module.classifier.cuda()




    return model, model.parameters()
def visualization_weight(model, title):
    aw = np.zeros([0, ])
    for name, parameters in model.named_parameters():
        w = np.reshape(parameters.detach().cpu().numpy(), [-1])
        aw = np.concatenate([aw, w], axis = 0)
    zero = 0
    for i in range(len(aw)):
        if aw[i] == 0:
            zero += 1
    print(zero, len(aw))
    
    f, b, _ = plt.hist(aw, bins = 100, range = [-0.3, 0.3], color = "b")
    print('频数', tuple(f))
    print('取值范围', tuple(b))
    plt.title("weights visulization after Prune")
    plt.yticks(range(0,600000, 100000))
    plt.ylim(0, 600000)
    plt.xlabel("",fontdict = {'size':10})
    plt.ylabel("nums",fontdict = {'size':10})
    plt.savefig("%s.png"%title)
    plt.clf()

if __name__ == "__main__":
    opt = parse_opts()
    opt.arch = '{}'.format(opt.model)
    inputs = torch.randn(1, 2, 8, 112, 112).cuda()

    model, parameters = generate_model(opt)
    model.cuda()
    model.eval()
    gpu_tracker = MemTracker()
    gpu_tracker.track()
    model(inputs)
    gpu_tracker.track()
    
    
    
    #parameters number
    total = sum(p.numel() for p in model.parameters())
    print("number of parameter: %.2fM" %(total/1e6))
    
    #weight_visualization
#    visualization_weight(model, "before sparsity")

    #weight shape
    for name, param in model.named_parameters():
        print(name, '\t\t', param.shape)

    #flops and params


