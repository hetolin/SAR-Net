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
    data_path = './data/NOCS/camera_train_processed/'

    category_list = os.listdir(data_path)
    train_list = {}

    categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

    data = []
    instance_num=0
    for category in category_list:
        instance_list = os.listdir(os.path.join(data_path, category))
        print(category, len(instance_list))

        for instance in instance_list:
            pointcloud_absolute_path = os.path.abspath(data_path)
            # instance_absolute_path = os.path.join(pointcloud_absolute_path, category, instance, 'pointclouds')
            instance_absolute_path = os.path.join(pointcloud_absolute_path, category, instance)
            for mfile in os.listdir(instance_absolute_path):
                if "obsv.obj" in mfile:
                    obsv_pcd = os.path.join(instance_absolute_path, mfile)
                    target_SA = obsv_pcd.replace('obsv.obj', 'SA.obj')
                    target_SC = obsv_pcd.replace('obsv.obj', 'SC.obj')
                    rot = obsv_pcd.replace('obsv.obj', 'rot.txt')
                    target_sOC = obsv_pcd.replace('obsv.obj', 'sOC.txt')
                    target_OS = obsv_pcd.replace('obsv.obj', 'OS.txt')
                    cate_id = [idx for idx in range(len(categories)) if categories[idx] == category]
                    data.append((obsv_pcd, target_SA, target_SC, target_sOC, target_OS, rot, cate_id[0]))

                    instance_num += 1

    print('total instance number={}'.format(instance_num)) #623143
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
