'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : pose_manager.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/11/4 21:25

# @Desciption: 
'''

import os

import torch as tc
from torch.utils.data import DataLoader
import dataset as dataloader_nocs

import net_respo.net_sarnet as sarnet

import lib.utils_files as file_utils
import torch
import numpy as np
from torch.optim import lr_scheduler
from lib.scheduler import GradualWarmupScheduler


def setup_env(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('Using GPU {}.'.format(args.gpu_id))


def setup_network(args, train_mode=True):
    encoder_decoder = sarnet.EncoderDecoder(args.num_cate)
    encoder_decoder.apply(init_weights)
    net = sarnet.SARNet(encoder_decoder)


    net = net.cuda()
    net.set_mode(train=train_mode)

    net_parameters = net.encoder_decoder.parameters()
    optimizer = tc.optim.Adam(net_parameters, lr=args.lr)

    scheduler_steplr = lr_scheduler.StepLR(optimizer, 4, 0.75)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=scheduler_steplr)

    if args.net_recover:
        # resume net, optimizer and scheduler_warmup
        start_epoch = recover_network(args, net, optimizer, scheduler_warmup)

        # resemble these two schedulers
        scheduler_steplr.optimizer = optimizer
        scheduler_warmup.optimizer = optimizer
        scheduler_warmup.after_scheduler = scheduler_steplr
    else:
        save_data = (net, optimizer, scheduler_warmup, 0)
        save_network(save_data, args.net_dump_folder)
        start_epoch = 0

    # parallel
    if args.is_parallel:
        print('=== parallel mode ===')
        print(args.device_id)
        net = torch.nn.DataParallel(net, device_ids=args.device_id)
        optimizer = torch.nn.DataParallel(optimizer, device_ids=args.device_id)
        scheduler_steplr = torch.nn.DataParallel(scheduler_steplr, device_ids=args.device_id)
        scheduler_warmup = torch.nn.DataParallel(scheduler_warmup, device_ids=args.device_id)


    return net, optimizer, scheduler_steplr, scheduler_warmup, start_epoch



def load_dataset(args):
    print('load {} dataset from {}'.format(args.dataset, args.json_file_path))
    dataset = dataloader_nocs.NOCS_DataSet(args)


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True #deterministic用来固定内部随机性
        torch.backends.cudnn.benchmark = True #benchmark用在输入尺寸一致，可以加速训练

    # 设置随机数种子
    setup_seed(0)

    dataloader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)

    return dataloader


def save_network(data, folder):
    model, optimizer, scheduler, epoch_num = data
    file_utils.create_folder(folder)

    file_name = os.path.join(folder, '{}.pth'.format(str(epoch_num).zfill(6)))
    print('save network at {}'.format(file_name))
    if hasattr(model, 'module'):
        torch.save({
           'net': model.module.state_dict(),
           'optimizer': optimizer.module.state_dict(),
           'epoch': epoch_num,
           'scheduler': scheduler.module.state_dict()
        }, file_name)
    else:
        torch.save({
           'net': model.state_dict(),
           'optimizer': optimizer.state_dict(),
           'epoch': epoch_num,
           'scheduler': scheduler.state_dict()
        }, file_name)
        # print(optimizer.param_groups[0]['lr'])

def recover_network(args, network, optimizer, scheduler):
    print("Recovering network training from checkpoint: %s (%d epochs)" %
          (args.net_recover_folder, args.net_recover_epoch))

    weight_file = os.path.join(args.net_recover_folder, '{}.pth'.format(str(args.net_recover_epoch).zfill(6)))
    load_parameters(network, optimizer, scheduler, weight_file)
    print("Network recovered correctly")
    start_epoch = args.net_recover_epoch + 1

    return start_epoch

def load_parameters(network, optimizer, scheduler, weight_file):
    if hasattr(network, 'module'):
        network.module.load_state_dict(tc.load(weight_file)['net'])
        optimizer.module.load_state_dict(tc.load(weight_file)['optimizer'])
        scheduler.module.load_state_dict(tc.load(weight_file)['scheduler'])
    else:
        network.load_state_dict(tc.load(weight_file)['net'])
        optimizer.load_state_dict(tc.load(weight_file)['optimizer'])
        scheduler.load_state_dict(tc.load(weight_file)['scheduler'])
        # print(optimizer.param_groups[0]['lr'])

    print("Successfully loaded network parameters from file: {}".format(weight_file))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
