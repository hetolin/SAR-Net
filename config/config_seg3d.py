'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : config_seg3d.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/5/9 16:11

# @Desciption: 
'''

class Config_Seg3d(object):
    #visible gpu id
    gpu_id = '1,2,3,4,5,6'

    device_id = []
    cuda = []
    for i in range(len(gpu_id.split(','))):
        device_id.append(i) #[0,1,2,3,4,5]
        cuda.append(str(i)) #['0', '1', '2', '3', '4', '5']

    cuda = ','.join(cuda) #'0,1,2,3,4,5'


    is_parallel = True

    mode = 'train'
    epoch = 50
    lr = 1e-4*6*2
    bs = 4*6*2
    num_workers = 8

    dataset = '../data'
    load = None
    load_model = '../net_models/1.pth'
    save = '../net_models_20210530'

    # point = 4096
    support = 1
    neighbor = 50
    class_num = 2*6
    cat_num = 6

    normal = False
    shift = None
    scale = None
    rotate = None
    axis = 0
    random = False

    interval = 1000
    record= 'record.log'
    output = '../output'

    train_json_file = '../data/camera_train.json'
    validate_json_file = '../data/camera_validate.json'

    PART_NUM = {
        "bottle": 2,
        "bowl": 2,
        "camera": 2,
        "can": 2,
        "laptop": 2,
        "mug": 2,
    }

    TOTAL_PARTS_NUM = sum(PART_NUM.values())

    point_num_seg = 2048
    point_num_pose = 1024
    point_num_minimum = 100
    pcd_filter_network_isNeed = True
    pcd_filter_isNeed = True
    pcd_isSave = True
    is_symmetry = True
    pcd_filter_pass_through_isNeed = False

    output_realworld = '../output_realworld'


args = Config_Seg3d()