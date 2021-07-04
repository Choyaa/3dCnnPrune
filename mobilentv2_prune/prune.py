import os
import torch
from torchvision import datasets, transforms
from model import generate_model
from opts import parse_opts
import json
from os.path import join
from dataset import get_training_set, get_validation_set, get_test_set
import data_loader
from mean import get_mean, get_std
from torch.optim import lr_scheduler
from torch import optim
from Slimming import SlimmingPrune
from train import train_epoch
from validation import val_epoch
import test
import numpy as np
import shutil

def save_checkpoint(model, is_best, filename='checkpoint.pth.tar'):
    torch.save(model, '%s/%s_checkpoint.pth' % (opt_prune.result_path, opt_prune.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt_prune.result_path, opt_prune.store_name),'%s/%s_best.pth' % (opt_prune.result_path, opt_prune.store_name))

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt_prune.learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate

opt_prune = parse_opts()
best_prec1  = 0
#===========initialize
opt_prune.scales = [opt_prune.initial_scale]
for i in range(1, opt_prune.n_scales):
    opt_prune.scales.append(opt_prune.scales[-1] * opt_prune.scale_step)
# opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
opt_prune.arch = '{}'.format(opt_prune.model)
opt_prune.mean = get_mean(opt_prune.norm_value, dataset=opt_prune.mean_dataset)
opt_prune.std = get_std(opt_prune.norm_value)

opt_prune.store_name = '_'.join([opt_prune.dataset, opt_prune.model,
                                 opt_prune.modality, str(opt_prune.sample_duration)])

torch.manual_seed(opt_prune.manual_seed)  # 生成固定随机数



#==========generate model
new_model, newparameters = generate_model(opt_prune)
model, parameters = generate_model(opt_prune)   #if opt_prune.pretrain_path , 预装模型初始化和加载

criterion, norm_method= data_loader.get_criterion_and_norm(opt_prune)
#===========train dataset
if not opt_prune.no_train:
    train_loader, train_logger, train_batch_logger = data_loader.get_traininfo(opt_prune, norm_method)


#==========validation dataset
if not opt_prune.no_val:
    validation_data, val_loader, val_logger = data_loader.get_valinfo(opt_prune, norm_method)


#++++++++++++++++开始！+++++++
#赋值给slimming 中 BasePruner(通过子类SlimmingPruner)
pruner = SlimmingPrune(model= model, newmodel= new_model, args=opt_prune)
pruner.prune()

new_model = pruner.return_model()

#++++++++++++结束++++++++++
total = sum(p.numel() for p in new_model.parameters())
print("Total params: %.2fM" % (total/1e6))
#print(new_model)
#visualization_weight(new_model,"after_prune")
#=============

if opt_prune.nesterov:
    dampening = 0
else:
    dampening = opt_prune.dampening
#====optimizer , scheduler
optimizer = optim.SGD(
        new_model.parameters(),
        lr=opt_prune.learning_rate,
        momentum=opt_prune.momentum,
        dampening=dampening,
        weight_decay=opt_prune.weight_decay,
        nesterov=opt_prune.nesterov)
scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt_prune.lr_patience)



print('run')
for i in range(opt_prune.begin_epoch, opt_prune.n_epochs + 1):
    adjust_learning_rate(optimizer, i, opt_prune.lr_steps)
    if not opt_prune.no_train:
        train_epoch(i, train_loader, new_model, criterion, optimizer, opt_prune,
                    train_logger, train_batch_logger)

        save_checkpoint(new_model, False)
        torch.cuda.empty_cache()
    if not opt_prune.no_val:
        validation_loss, prec1 = val_epoch(i, val_loader, new_model, criterion, opt_prune,
                                    val_logger)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint(new_model, is_best)



