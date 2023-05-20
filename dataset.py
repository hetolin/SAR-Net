'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : dataset.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/11/4 21:45

# @Desciption: 
'''


# import math
# import os
# import numpy as np
# import torch as tc
# from torch.lib.data import Dataset
# import pymesh
# from math import sin, cos
# import json
# from copy import deepcopy
#
# #mc_dropout_network_v23
# def pc_normalize(pcd):
#     """ pc: NxC, return NxC """
#     pc = deepcopy(pcd)
#
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
#     pc = pc / scale
#
#     return pc, centroid, scale

# class NOCS_DataSet(Dataset):
#     def __init__(self, json_file_path):
#         self.file_path = json_file_path
#
#         with open(json_file_path, "r") as stream:
#             self.data = json.load(stream)
#
#         self.length = len(self.data)
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
#
#         input_data = self.data[idx][0]
#         output_data = self.data[idx][1]
#         sym_data = self.data[idx][2]
#         rot_data = self.data[idx][3]
#         center_data = self.data[idx][4]
#         size_data = self.data[idx][5]
#         cate_id = self.data[idx][6]
#         # print(self.data[idx])
#
#         self.category = categories[cate_id]
#         prior_model_path = './data/FPS_sample/' + self.category + '_fps_36.obj'
#         prior_model = pymesh.load_mesh(prior_model_path).vertices
#         ''''''
#         prior_model, _, _ = pc_normalize(prior_model)
#
#         self.prior_model = prior_model
#
#         _sym = deepcopy(pymesh.load_mesh(sym_data).vertices)
#
#         x = tc.from_numpy(pymesh.load_mesh(input_data).vertices).contiguous()
#         z = tc.from_numpy(pymesh.load_mesh(output_data).vertices).contiguous()
#         sym_part = tc.from_numpy(_sym).contiguous()
#         rot = tc.from_numpy(np.loadtxt(rot_data)).contiguous()
#
#         np.random.seed(idx)
#         x_sample = np.random.choice(x.shape[0], size=1024, replace=False)
#         x = x[x_sample]
#         sym_part = sym_part[x_sample]
#
#         '''add noise'''
#         # np.random.seed(idx)
#         # dist_noise = np.random.normal(loc=0, scale=[0, 0, 0.02], size=(1024,3))
#         # x += dist_noise
#         # sym_part+=dist_noise
#
#         center = tc.from_numpy(np.loadtxt(center_data)[:-1]).reshape(3, 1).contiguous()  # (
#
#         prior = tc.from_numpy(self.prior_model).contiguous()
#
#
#         '''
#         x = tc.from_numpy(pcl.load(input_data).to_array()).contiguous()
#         y = tc.from_numpy(pcl.load(output_data).to_array()).contiguous()
#         '''
#
#         size = tc.from_numpy(np.loadtxt(size_data)).unsqueeze(1) #(3,1)
#         # x_sample = np.random.choice(x.shape[0], size=2048, replace=False)
#         # x = x[x_sample]
#         cate_id = tc.from_numpy(np.array(cate_id))
#
#         # (3, N)
#         x = x.float().transpose(1, 0).contiguous()
#         z = z.float().transpose(1, 0).contiguous()
#         sym_part = sym_part.float().transpose(1, 0).contiguous()
#
#
#         rot = rot.float()
#         center = center.float()
#
#         # noise_tensor = noise_tensor.float()
#         prior = prior.float().transpose(1, 0).contiguous()
#         size = size.float()
#
#         if self.category in ['bowl', 'can', 'bottle']:
#             theta = 2 * np.pi / 12
#             rot_y_matrix = np.array([[cos(theta), 0, sin(theta)],
#                                      [0, 1, 0],
#                                      [-sin(theta), 0, cos(theta)]
#                                      ])
#             rot_y_matrix = tc.from_numpy(rot_y_matrix).float()
#
#             # x,y (3, 36)
#             # rot (3, 3)
#             # self_coords (3, 36)
#             self_coords = tc.matmul(rot.transpose(1, 0).contiguous(), z)
#
#             # self coords
#             # self_rot_n (3, 36)
#             self_rot_1 = tc.matmul(rot_y_matrix, self_coords)
#             self_rot_2 = tc.matmul(rot_y_matrix, self_rot_1)
#             self_rot_3 = tc.matmul(rot_y_matrix, self_rot_2)
#             self_rot_4 = tc.matmul(rot_y_matrix, self_rot_3)
#             self_rot_5 = tc.matmul(rot_y_matrix, self_rot_4)
#             self_rot_6 = tc.matmul(rot_y_matrix, self_rot_5)
#             self_rot_7 = tc.matmul(rot_y_matrix, self_rot_6)
#             self_rot_8 = tc.matmul(rot_y_matrix, self_rot_7)
#             self_rot_9 = tc.matmul(rot_y_matrix, self_rot_8)
#             self_rot_10 = tc.matmul(rot_y_matrix, self_rot_9)
#             self_rot_11 = tc.matmul(rot_y_matrix, self_rot_10)
#
#             # world coords
#             # world_rot_n (3, 36)
#             world_rot_1 = tc.matmul(rot, self_rot_1)
#             world_rot_2 = tc.matmul(rot, self_rot_2)
#             world_rot_3 = tc.matmul(rot, self_rot_3)
#             world_rot_4 = tc.matmul(rot, self_rot_4)
#             world_rot_5 = tc.matmul(rot, self_rot_5)
#             world_rot_6 = tc.matmul(rot, self_rot_6)
#             world_rot_7 = tc.matmul(rot, self_rot_7)
#             world_rot_8 = tc.matmul(rot, self_rot_8)
#             world_rot_9 = tc.matmul(rot, self_rot_9)
#             world_rot_10 = tc.matmul(rot, self_rot_10)
#             world_rot_11 = tc.matmul(rot, self_rot_11)
#
#             # check
#             # save_to_obj_pts(target[0].cpu().detach().numpy().transpose(), './src.obj')
#             # save_to_obj_pts(target[0].cpu().detach().numpy().transpose(), './sym_gt.obj')
#             # save_to_obj_pts(world_rot_1[0].cpu().detach().numpy().transpose(), './src2.obj')
#             # save_to_obj_pts(output[0].cpu().detach().numpy().transpose(), './sym_out.obj')
#
#             # target_gather (12, 3, 36)
#             z_target_gather = tc.stack([z, world_rot_1, world_rot_2, world_rot_3, world_rot_4, world_rot_5,
#                                         world_rot_6, world_rot_7, world_rot_8, world_rot_9, world_rot_10, world_rot_11],
#                                        dim=0)
#             z = z_target_gather
#
#
#         else:
#             z = z.unsqueeze(0).repeat((12,1,1))
#
#         return x, prior, z, sym_part, center, size, cate_id
import os
import numpy as np
import torch as tc

from math import sin, cos
import json
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from lib.utils_pose import save_to_obj_pts, load_obj
from config.config_sarnet import args

#mc_dropout_network_v23
def pc_normalize(pcd):
    """ pc: NxC, return NxC """
    pc = deepcopy(pcd)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / scale

    return pc, centroid, scale

# category_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
# # category_list = ['bathtub', 'bed', 'bookshelf', 'chair', 'sofa', 'table']
class NOCS_DataSet(Dataset):
    def __init__(self, args):
        self.json_file_path = args.json_file_path
        self.dataset = args.dataset
        self.nFPS = args.nFPS
        with open(self.json_file_path, "r") as stream:
            self.data = json.load(stream)

        self.length = len(self.data)

        self.categories = args.categories
        self.temp_folder = args.temp_folder


    def __len__(self):
        return self.length
        # return 1200

    def __getitem__(self, idx):
        obsv_pcd_path = self.data[idx][0]
        target_SA_path = self.data[idx][1] # Shape Alignment
        target_SC_path = self.data[idx][2] # Symmetry Correspondence
        target_sOC_path = self.data[idx][3] # scale_factor(not used yet), Object Center
        target_OS_path = self.data[idx][4] # Object Size
        rot_path = self.data[idx][5]
        cate_id = self.data[idx][6]


        self.category = self.categories[cate_id]
        temp_pcd_path = os.path.join(self.temp_folder, f'{self.category}_fps_{self.nFPS}_normalized.obj')
        # temp_pcd_path = os.path.join('../data', self.dataset, 'template_FPS/{}_fps_{}_normalized.obj'.format(self.category, self.nFPS))
        temp_pcd, _ = load_obj(temp_pcd_path)

        obsv_pcd, _ = load_obj(obsv_pcd_path)
        target_SC, _ = load_obj(target_SC_path)
        target_OC = np.loadtxt(target_sOC_path)[:-1].reshape(3, 1)  # (3,1)
        target_OS = np.loadtxt(target_OS_path).reshape(3, 1)  # (3,1)
        rot = np.loadtxt(rot_path)
        cate_id = np.array(cate_id)
        target_SA = np.dot(temp_pcd, rot.T)

        # add in-plane rotation
        # np.random.seed(idx)
        # in_plane = np.random.uniform(-60,60,1)
        # in_plane = np.radians(in_plane)
        # rotz = np.array([[cos(in_plane), -sin(in_plane), 0], [sin(in_plane), cos(in_plane), 0], [0, 0, 1]])

        # obsv_pcd = np.dot(obsv_pcd, rotz.T) #(N,3)
        # target_SA = np.dot(target_SA, rotz.T) #(N,3)
        # target_SC = np.dot(target_SC, rotz.T) #(N,3)
        # rot = np.dot(rotz, rot)
        # targe_OC = np.dot(rotz, target_OC) #(N,3)

        # to tensor
        data = (obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, rot, cate_id)
        obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, rot, cate_id = [tc.from_numpy(d) for d in data]

        # sample
        np.random.seed(idx)
        sample = np.random.choice(obsv_pcd.shape[0], size=1024, replace=False)
        obsv_pcd = obsv_pcd[sample]
        target_SC = target_SC[sample]

        # adjust shape and type
        # (3, N)
        obsv_pcd = obsv_pcd.float().transpose(1, 0).contiguous()
        temp_pcd = temp_pcd.float().transpose(1, 0).contiguous()
        target_SA = target_SA.float().transpose(1, 0).contiguous()
        target_SC = target_SC.float().transpose(1, 0).contiguous()
        target_OC = target_OC.float()
        target_OS = target_OS.float()
        rot = rot.float()


        if self.category in ['bowl', 'can', 'bottle']:
            theta = 2 * np.pi / 12
            rot_y_matrix = np.array([[cos(theta), 0, sin(theta)],
                                     [0, 1, 0],
                                     [-sin(theta), 0, cos(theta)]
                                     ])
            rot_y_matrix = tc.from_numpy(rot_y_matrix).float()

            # target_SA (3, 36)
            # rot (3, 3)
            _SA_obj = tc.matmul(rot.transpose(1, 0).contiguous(), target_SA)

            # in object coordinate
            GT_NUM = 12
            SA_obj_list = []
            SA_obj_list.append(_SA_obj)
            for i in range(1, GT_NUM):
                _SA_obj = tc.matmul(rot_y_matrix, SA_obj_list[i-1])
                SA_obj_list.append(_SA_obj)


            # in camera coordinate
            SA_cam_list = []
            SA_cam_list.append(target_SA)
            for i in range(1, GT_NUM):
                _SA_cam = tc.matmul(rot, SA_obj_list[i])
                SA_cam_list.append(_SA_cam)

            target_SA = tc.stack(SA_cam_list, dim=0)

        else:
            target_SA = target_SA.unsqueeze(0).repeat((12,1,1))

        return obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, cate_id

def test_dataset():
    nocs = NOCS_DataSet(args)
    dataloader = DataLoader(nocs, batch_size= 4, shuffle= True)
    categories = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

    for i, (obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, cate_id) in enumerate(dataloader):
        print(target_SA.shape)
        save_to_obj_pts(obsv_pcd[0].numpy().transpose(), './debug/{}_obsv.obj'.format(cate_id[0]))
        save_to_obj_pts(target_SA[0][0].numpy().transpose(), './debug/{}_SA.obj'.format(cate_id[0]))
        save_to_obj_pts(target_SC[0].numpy().transpose(), './debug/{}_SC.obj'.format(cate_id[0]))

        if i == 1:
            break

if __name__ == '__main__':
    test_dataset()