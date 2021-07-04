import os
from opts import parse_opts
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from utils import Logger
import torch.nn as nn

from dataset import get_training_set, get_validation_set, get_test_set



def get_criterion_and_norm(opt):
    if opt.dataset == 'SHGD' and (opt.n_finetune_classes == 13 or opt.n_classes == 13):
        # give the "No gesture/Hand up/Hand down less weight than the other classes. No:4420 Hand up:2280 Hand Down:2190 Others:228
        class_weights = [1, 1, 1, 1 / 10, 1 / 10, 1 / 20, 1, 1, 1, 1, 1, 1, 1]
        if not opt.no_cuda:
            class_weights = torch.Tensor(class_weights).cuda()
        else:
            class_weights = torch.Tensor(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if False and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    return criterion, norm_method



def get_traininfo(opt, norm_method ):
    assert opt.train_crop in ['random', 'corner', 'center']
    if opt.train_crop == 'random':
        crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'corner':
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
    elif opt.train_crop == 'center':
        crop_method = MultiScaleCornerCrop(
            opt.scales, opt.sample_size, crop_positions=['c'])
    spatial_transform = Compose([
        RandomRotate(),
        RandomResize(),
        crop_method,
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(opt.sample_duration)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform,
                                     temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True)
    train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
    return train_loader, train_logger, train_batch_logger

def get_valinfo(opt, norm_method):
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalCenterCrop(opt.sample_duration)

    target_transform = ClassLabel()
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    val_logger = Logger(
        os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])
    return validation_data, val_loader, val_logger

def get_testinfo(opt, norm_method):
    spatial_transform = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    # temporal_transform = LoopPadding(opt.sample_duration)
    target_transform = ClassLabel()

    test_data = get_test_set(opt, spatial_transform, target_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,  # batchsize must be 1
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    return  test_loader
