'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : generate_json.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/4/25 19:40

# @Desciption: 
'''

import os
import json
import random

def read_json_file(file_path):
    with open(file_path, 'r') as stream:
        return json.load(stream)

if __name__ == '__main__':
    data_path = './data/NOCS/camera_train/'
    # data_path = '/var/lib/docker/robotics_group/linhaitao/lht/tmp_data/camera_train'

    category_list = os.listdir(data_path)
    train_list = {}

    categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

    data = []
    for category in category_list:
        instance_list = os.listdir(os.path.join(data_path, category))
        print(category, len(instance_list))

        for instance in instance_list:
            pointcloud_absolute_path = os.path.abspath(data_path)
            # instance_absolute_path = os.path.join(pointcloud_absolute_path, category, instance, 'pointclouds')
            instance_absolute_path = os.path.join(pointcloud_absolute_path, category, instance)
            for mfile in os.listdir(instance_absolute_path):
                # if "obsv.obj" in mfile:
                #     obsv_pcd = os.path.join(instance_absolute_path, mfile)
                #     target_SA = obsv_pcd.replace('obsv.obj', 'SA.obj')
                #     target_SC = obsv_pcd.replace('obsv.obj', 'SC.obj')
                #     rot = obsv_pcd.replace('obsv.obj', 'rot.txt')
                #     target_sOC = obsv_pcd.replace('obsv.obj', 'sOC.txt')
                #     target_OS = obsv_pcd.replace('obsv.obj', 'OS.txt')
                #     cate_id = [idx for idx in range(len(categories)) if categories[idx] == category]
                #     data.append((obsv_pcd, target_SA, target_SC, target_sOC, target_OS, rot, cate_id[0]))

                if "x.obj" in mfile:
                    x = os.path.join(instance_absolute_path, mfile)
                    z = x.replace('x.obj', 'z_36.obj')
                    sym = x.replace('x.obj', 's.obj')
                    rot = x.replace('x.obj', 'rot.txt')
                    center = x.replace('x.obj', 'center.txt')
                    size = x.replace('x.obj', 'size.txt')
                    cate_id = [idx for idx in range(len(categories)) if categories[idx] == category]
                    data.append((x, z, sym, center, size, rot, cate_id[0]))

    random.seed(0)
    random.shuffle(data)
    train_list = data

    with open('./data/NOCS/camera_train.json', "w") as outfile:
        json.dump(train_list, outfile)
    print(len(train_list))

    # test json
    model_names = read_json_file('./data/NOCS/camera_train.json')

    print(model_names[0][0])
    print(model_names[0][1])
    print(model_names[0][2])
    print(model_names[0][3])
    print(model_names[0][4])
    print(model_names[0][5])
    print(model_names[0][6])
