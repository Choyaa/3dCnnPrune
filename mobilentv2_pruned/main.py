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
注意：这是test，训练的resume改 if opt.resume:
SGD里改回parameters，现在是model.parameters()


'''
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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate


best_prec1 = 0

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    #opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)

    opt.store_name = '_'.join([opt.dataset, opt.model,
                               opt.modality, str(opt.sample_duration)])

    print(opt)

    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)   #将python对象编码成Json字符串

    torch.manual_seed(opt.manual_seed)  #生成固定随机数

    #model, parameters = generate_model(opt)

    #print(model)
    #========================注意：这是test，训练的resume改
    if opt.resume_path:
        model = torch.load(opt.resume_path)
        print(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

#get criterion, norm_method
    criterion, norm_method= data_loader.get_criterion_and_norm(opt)

### when test mode, train and val are turned off. Because test mode uses different dataset
    if opt.test:
        opt.no_train = True
        opt.no_val = True
        if not opt.resume_path:
            print('Please give a path to a trained model for testing.')
            sys.exit()
#get training dataset
    if not opt.no_train:
        train_loader, train_logger, train_batch_logger = data_loader.get_traininfo(opt, norm_method)
#get dampening  , optimizer , scheduler
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
#get validation dataset
    if not opt.no_val:
        validation_data, val_loader, val_logger = data_loader.get_valinfo(opt, norm_method)


    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        adjust_learning_rate(optimizer, i, opt.lr_steps)
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
            state = {
               'epoch': i,
               'arch': opt.arch,
               'state_dict': model.state_dict(),
               'optimizer': optimizer.state_dict(),
               'best_prec1': 0
               }
            save_checkpoint(state, False)
            torch.cuda.empty_cache()
        if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best)

        #if not opt.no_train and not opt.no_val:
        #    scheduler.step(validation_loss)

    if opt.test:
        test_loader = data_loader.get_testinfo(opt, norm_method)
        test.test(test_loader, model, opt)

