'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : train_sarnet.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/4/21 20:05

# @Desciption: 
'''

import numpy as np
import pose_manager
import lib.utils_files as file_utils

import os
from net_respo.net_sarnet import Loss_Func
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from config.config_sarnet import args

def optimizer_zero_grad(optimizer):
    optimizer.module.zero_grad() if args.is_parallel else optimizer.zero_grad()

def optimizer_step(optimizer):
    optimizer.module.step() if args.is_parallel else optimizer.step()

def scheduler_step(scheduler_warmup, epoch):
    scheduler_warmup.module.step(epoch) if args.is_parallel else scheduler_warmup.step(epoch)

def set_get_lr(scheduler_warmup, optimizer):
    current_set_lr = scheduler_warmup.module.get_lr()[0] if args.is_parallel else scheduler_warmup.get_lr()[0]
    current_get_lr = optimizer.module.param_groups[0]['lr'] if args.is_parallel else optimizer.param_groups[0]['lr']
    return current_set_lr, current_get_lr

def train_model(args):
    # setup environment
    pose_manager.setup_env(args)
    log_string ='epoch:{:0>3d}, deform_loss = {:.6f}, sym_loss = {:.6f}, center_loss = {:.6f}, size_loss = {:.6f}'

    net, optimizer, _, scheduler_warmup, start_epoch = pose_manager.setup_network(args, train_mode=True)

    # start_epoch = 0
    # if args.net_recover:
    #     start_epoch = pose_manager.recover_network(args, net, optimizer, scheduler_warmup)
    # else:
    #     save_data = (net, optimizer, scheduler_warmup, 0)
    #     pose_manager.save_network(save_data, args.net_dump_folder)

    # optimizer = tc.optim.Adam(net.encoder_decoder.parameters(), lr=0.0001*args.batch_size/8.0)
    # scheduler_steplr = lr_scheduler.StepLR(optimizer, 4, 0.75)

    loss_function = Loss_Func()
    dataloader = pose_manager.load_dataset(args)

    optimizer_zero_grad(optimizer)
    optimizer_step(optimizer)
    print("Starting training (%d samples, %d epochs)" % (len(dataloader.dataset), args.num_epochs))

    progress=Progress(TextColumn("[progress.description]{task.description}"),
                      SpinnerColumn(),
                      BarColumn(),
                      TextColumn("[progress.filesize]{task.completed:>06d}/{task.total:0>6d}"),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn())
    epoch_tqdm = progress.add_task(description="[purple]Epoch progress", total=args.num_epochs, completed=start_epoch)
    batch_tqdm = progress.add_task(description="[purple]Batch progress", total=len(dataloader))
    progress.start() ## 开启

    for epoch in range(start_epoch,  args.num_epochs):
        total_deform_loss = []
        total_center_loss = []
        total_size_loss = []
        total_sym_loss = []

        scheduler_step(scheduler_warmup, epoch+1) # from 1 to N

        current_set_lr, current_get_lr = set_get_lr(scheduler_warmup, optimizer)
        # current_lr = scheduler_steplr.module.get_last_lr()[0] if args.is_parallel else scheduler_steplr.get_last_lr()[0]
        print('#########################################')
        print('epoch: {:0>3d}, set lr: {:.6f}, get lr: {:.6f}'.format(epoch, current_set_lr, current_get_lr)) #torch 1.4

        # for idx, data in enumerate(tqdm(dataloader, ncols=80)):
        for idx, data in enumerate(dataloader):
            progress.advance(batch_tqdm, advance=1)
            obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, cate_id = [d.cuda() for d in data]

            optimizer_zero_grad(optimizer)

            preds = net(obsv_pcd, temp_pcd, cate_id, mode='train', target_SC=target_SC)

            inputs = (obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, cate_id)
            loss, deform_loss, sym_loss, center_loss, size_loss = loss_function(preds, inputs)
            loss.backward()
            total_deform_loss.append(deform_loss.item())
            total_sym_loss.append(sym_loss.item())
            total_center_loss.append(center_loss.item())
            total_size_loss.append(size_loss.item())

            optimizer_step(optimizer)

            print_frequency = len(dataloader) // args.print_num
            if idx % print_frequency == 0:
                loss_log = log_string.format(epoch,
                                             np.mean(total_deform_loss),
                                             np.mean(total_sym_loss),
                                             np.mean(total_center_loss),
                                             np.mean(total_size_loss))
                print(loss_log)

        progress.advance(epoch_tqdm, advance=1)
        progress.reset(batch_tqdm)

        # save checkpoints
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_data = (net, optimizer, scheduler_warmup, epoch)
            pose_manager.save_network(save_data, args.net_dump_folder)

        # output and save log
        print('epoch {} is finished'.format(epoch))
        loss_log = log_string.format(epoch,
                                     np.mean(total_deform_loss),
                                     np.mean(total_sym_loss),
                                     np.mean(total_center_loss),
                                     np.mean(total_size_loss))

        file_utils.log_data_to_file(loss_log, os.path.join(args.net_dump_folder, 'loss.txt'))

        # scheduler_warmup.module.step() if args.is_parallel else scheduler_warmup.step()

        print(loss_log)
        print('#########################################')

    net.eval()

    print("Finished training")

    # Saving final network to file
    print("Saving final network parameters")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_model(args)
