'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : config_sarnet.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/4/21 21:24

# @Desciption: 
'''

import os
import lib.utils_files as file_utils
import pickle

#batch_size[32*4] warmup 2
#batch_size[32*8] warmup 4
#batch_size[32*16]
class Config_SARNet(object):
    # assign GPU
    gpu_id = '0'
    # gpu_id = '1,2,3,4'

    device_id = []
    cuda = []
    for i in range(len(gpu_id.split(','))):
        device_id.append(i) #[0,1,2,3,4,5]
        cuda.append(str(i)) #['0', '1', '2', '3', '4', '5']
    cuda = ','.join(cuda) #'0,1,2,3,4,5'

    is_parallel = True if len(device_id)>1 else False

    # assign net parameters
    mode = 'train'
    num_epochs = 200
    # batch_size = 32 #* 8#len(device_id)
    batch_size = 32 * len(device_id)
    num_workers = 8 #* 2
    checkpoint_interval = 2
    print_num = 5
    num_cate = 6
    warmup_epochs = 2 if is_parallel else 1# set 1 for disable warmup
    lr = 0.0001*batch_size/8.0

    # assign data folder
    data_root = './data'
    dataset = 'NOCS'
    json_name = 'camera_train.json'
    json_file_path = os.path.join(data_root, dataset, json_name)

    categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    nFPS = 36
    temp_folder = os.path.join(data_root, dataset, 'template_FPS')


    net_dump_root = "./checkpoints/"
    net_dump_name = "sarnet_GPUs" if is_parallel else "sarnet_GPU"
    net_dump_folder = os.path.join(net_dump_root, dataset, net_dump_name)

    # net_recover = False
    net_recover_epoch = 0
    net_recover_folder = os.path.join(net_dump_folder)
    net_recover = True if net_recover_epoch>0 else False


args = Config_SARNet()

# create net dump folder
file_utils.create_folder(args.net_dump_folder)

# save args.txt
f = os.path.join(args.net_dump_folder, 'arg.txt')
with open(f, 'w') as file:
    for arg in (vars(Config_SARNet)):
        if '__' in arg:
            continue
        attr = getattr(args, arg)
        file.write('{} = {}\n'.format(arg, attr))

# save args.pkl
f = os.path.join(args.net_dump_folder, 'arg.pkl')
with open(f, 'wb') as file:
    str = pickle.dumps(args)
    file.write(str)

# # read args.pkl
# args=Config_SARNet()
# with open(f,'rb') as file:
#     args  = pickle.loads(file.read())