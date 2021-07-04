import os
import sys
import json
import shutil
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test

import data_loader
from model import olde_key_to_new

from torch2trt import *

from torchvision.models import *

from collections import OrderedDict
import time
from utils import AverageMeter
from collections import deque
import math
from model import olde_key_to_new

opt = parse_opts()

opt.arch = '{}'.format(opt.model)
opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
opt.std = get_std(opt.norm_value)

opt.store_name = '_'.join([opt.dataset, opt.model,
                            opt.modality, str(opt.sample_duration)])


#model, parameters = generate_model(opt)

#print(model)
#========================注意：这是test，训练的resume改
''' 
new_state = OrderedDict()
if opt.resume_path:
    model = torch.load(opt.resume_path)
for k, v in model.state_dict().items():
    name =k[7:]
    new_state[name]= v
    
state = {
        'arch':opt.arch,
        'state_dict':new_state
        }
print(state['arch'])
torch.save(state,'%s/%s_pruned40.pth'%(opt.result_path, opt.store_name))



pretrain = torch.load(opt.pretrain_path)
assert opt.arch == pretrain['arch']
'''
#model.cuda()

model,parameters = generate_model(opt)





   


criterion, norm_method= data_loader.get_criterion_and_norm(opt)
 
if opt.nesterov:
        dampening = 0
else:
    dampening = opt.dampening

optimizer = optim.SGD(
    model.parameters(),
    lr=opt.learning_rate,
    momentum=opt.momentum,
    dampening=dampening,
    weight_decay=opt.weight_decay,
    nesterov=opt.nesterov)
scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt.lr_patience)

test_loader = data_loader.get_testinfo(opt, norm_method)

model.eval()


n_stride = 1



batch_time = AverageMeter()
data_time = AverageMeter()
end_time = time.time()

with torch.no_grad():
    for i, (inputs, targets, input_length) in enumerate(test_loader):
        step = math.ceil((input_length-opt.sample_duration) // n_stride)
        for j in range(0,200):
            #min_j = min(j*n_stride+opt.sample_duration, input_length)
           # M = list(range(0, 8))
            M = list(range(j*n_stride, j*n_stride+opt.sample_duration))
            if len(M) < 8:
                break
            else:
                input_single = inputs[:, :, M[0]:(M[0] +opt.sample_duration),:,:] 
                output= model(input_single)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        print('time:{batch_time.avg:.5f}'.format(batch_time = batch_time))
#                torch.cuda.empty_cache()

'''




device = torch.device("cuda")
model_org =mobilenet_v2(pretrained = True)
model_org .to(device)
model_org.eval()


test1 = torch.rand(1, 3, 224 , 224).to(device)
model_trt = torch2trt(model_org, [test1], fp16_mode = True, max_workspace_size = 100)
model_trt.eval()


test4 = model_trt(test1)
batch_time.update(time.time() - end_time)
end_time = time.time()
print('time:{batch_time.avg:.5f}'.format(batch_time = batch_time))



test2 = model_org(test1.cuda())
batch_time.update(time.time() - end_time)
end_time = time.time()
print('time:{batch_time.avg:.5f}'.format(batch_time = batch_time))


for i in range(50):
    test3 = model_org(test1.cuda())
    batch_time.update(time.time() - end_time)
    end_time = time.time()
    print('time:{batch_time.avg:.5f}'.format(batch_time = batch_time))
print('next')
for i in range(50):
    test4 = model_trt(test1.cuda())
    batch_time.update(time.time() - end_time)
    end_time = time.time()
    print('time:{batch_time.avg:.5f}'.format(batch_time = batch_time))


M = list(range(0, 8))

input_var = 
output = model(input_var)
batch_time.update(time.time() - end_time)
end_time = time.time()
print('time:{batch_time.avg:.5f}'.format(batch_time = batch_time))

#for i, (inputs, targets, input_length) in enumerate(test_loader):
'''
