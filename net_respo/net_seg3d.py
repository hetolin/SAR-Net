'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : net_seg3d.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/5/8 11:17

# @Desciption: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from net_respo import gcn3d

class GCN3D(nn.Module):
    def __init__(self, class_num, cat_num, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num

        '''cat_num is not class_num, class_num = cat_num * part_num'''

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)

        dim_fuse = sum([128, 128, 256, 256, 512, 512, cat_num])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                onehot: "tensor (bs, cat_num)"):
        """
        Return: (bs, vertice_num, class_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)
        fm_1 = F.relu(self.conv_1(neighbor_index, vertices, fm_0), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)

        fm_2 = F.relu(self.conv_2(neighbor_index, v_pool_1, fm_pool_1), inplace= True)
        fm_3 = F.relu(self.conv_3(neighbor_index, v_pool_1, fm_2), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global, onehot], dim= 2)

        # print(fm_0.shape, fm_1.shape, fm_2.shape, fm_3.shape, fm_4.shape, f_global.shape)
        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch)
        return pred

def test():
    model = GCN3D(class_num= 12, cat_num=6, support_num= 1, neighbor_num= 50)

if __name__ == "__main__":
    test()