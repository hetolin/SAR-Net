'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : net_sarnet.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2022/4/21 20:21

# @Desciption: 
'''
import os
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from lib.utils_files import create_folder
from lib.utils_pose import save_to_obj_pts

def pc_centralize(pcd):
    '''
    :param pcd: (B,3,N)
    :return: (B,3,N)
    '''
    pc = pcd.clone()
    centroid = tc.mean(pc, dim=2).unsqueeze(2) #(B,3,1)
    pc = pc - centroid

    return pc, centroid

class PointEncoder(nn.Module):
    def __init__(self):
        super(PointEncoder, self).__init__()
        self.e_conv1 = nn.Conv1d(3, 64, 1)  # (B,3,N)
        self.e_conv2 = nn.Conv1d(64, 64, 1)
        self.e_conv3 = nn.Conv1d(64, 128, 1)
        self.e_conv4 = nn.Conv1d(128, 256, 1)
        self.e_conv5 = nn.Conv1d(256, 512, 1)

        self.bn1 = nn.InstanceNorm1d(64)
        self.bn2 = nn.InstanceNorm1d(64)
        self.bn3 = nn.InstanceNorm1d(128)
        self.bn4 = nn.InstanceNorm1d(256)
        self.bn5 = nn.InstanceNorm1d(512)

    def forward(self, x):
        '''
        :param x: (B,3,N)
        :return:
        '''
        x = F.relu(self.bn1(self.e_conv1(x)))
        x = F.relu(self.bn2(self.e_conv2(x)))
        x = F.relu(self.bn3(self.e_conv3(x)))  # (B,128,N)
        maxpool_128, _ = tc.max(x, 2)  # (B,128)
        x = F.relu(self.bn4(self.e_conv4(x)))
        maxpool_256, _ = tc.max(x, 2)
        x = F.relu(self.bn5(self.e_conv5(x)))
        maxpool_512, _ = tc.max(x, 2)

        Feature = [maxpool_128, maxpool_256, maxpool_512]
        inst_global = tc.cat(Feature, 1).unsqueeze(2)  # (B,896,1)

        return inst_global



class EncoderDecoder(nn.Module):
    def __init__(self, num_cate=6):
        super(EncoderDecoder, self).__init__()
        self.num_cate = num_cate
        self.posefeat = PointEncoder()
        self.transfeat = PointEncoder()

        self.category_local = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.category_global = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.deformation = nn.Sequential(
            nn.Conv1d((3 + 64 + 1024 + 896), 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3*self.num_cate, 1),
        )

        self.vote = nn.Sequential(
            nn.Conv1d((3 + 1024 + 896), 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3*self.num_cate, 1),
            # nn.Tanh()
        )

        self.reg_size = nn.Sequential(
            nn.Linear(896, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3*self.num_cate),
        )

        self.sym = nn.Sequential(
            nn.Conv1d((3 + 1024 + 896), 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3*self.num_cate, 1),
        )

    def forward(self, obsv_pcd, temp_pcd, cate_id, mode, target_SC = None):
        '''
        :param x: (B,3,N)
        :param prior: (B,3,N_k)
        :param cate_id: (B)
        :param mode: 'train' or 'test'
        :param sym_part_gt: (B,3,N)
        :return:
        '''

        # temp_pcd = deepcopy(prior)
        # obsv_pcd = deepcopy(x)


        bs, dim, temp_npts = temp_pcd.shape
        _, _, obsv_npts = obsv_pcd.shape

        inst_global = self.posefeat(obsv_pcd) # (B,896,1)

        temp_local = self.category_local(temp_pcd)  # (B,64,N)
        temp_global = self.category_global(temp_local)  # (B,1024,1)

        deform_feat = tc.cat((temp_pcd,
                              temp_local,
                              temp_global.repeat(1, 1, temp_npts),
                              inst_global.repeat(1, 1, temp_npts)),
                             dim=1)  # bs x (3+64+1024+896)) x 36

        sym_feat = tc.cat((obsv_pcd,
                           temp_global.repeat(1, 1, obsv_npts),
                           inst_global.repeat(1, 1, obsv_npts)),
                          dim=1)  # bs x (3+1024+896)) x 2048

        index = cate_id.squeeze() + tc.arange(bs, dtype=tc.long).cuda() * self.num_cate

        # Shape Alignment
        pred_deltaSA = self.deformation(deform_feat)
        pred_deltaSA = pred_deltaSA.view(-1, 3, temp_npts).contiguous()  # bs, nc*3, cate_npts -> bs*nc, 3, cate_npts
        pred_deltaSA = tc.index_select(pred_deltaSA, 0, index).contiguous()  # bs x 3 x cate_npts


        # Symmetric Correspondence
        pred_SC = self.sym(sym_feat)
        pred_SC = pred_SC.view(-1, 3, obsv_npts).contiguous() # bs, nc*3, cate_npts -> bs*nc, 3, cate_npts
        pred_SC = tc.index_select(pred_SC, 0, index).contiguous()  # bs x 3 x inst_npts

        if mode == 'train':
            if target_SC is not None:
                coarse_pcd = tc.cat((obsv_pcd, target_SC), dim=2)
            else:
                coarse_pcd = tc.cat((obsv_pcd, pred_SC), dim=2)
        else:
            coarse_pcd = tc.cat((obsv_pcd, pred_SC), dim=2)

        coarse_pcd, centroid = pc_centralize(coarse_pcd)
        _, _, coarse_npts = coarse_pcd.shape

        coarse_global = self.transfeat(coarse_pcd)
        trans_feat = tc.cat((coarse_pcd,
                             temp_global.repeat(1, 1, coarse_npts),
                             coarse_global.repeat(1, 1, coarse_npts)),
                             dim=1)  # bs x (3+3+1024+896)) x 2048*2

        # Object Center
        pred_deltaOC = self.vote(trans_feat)
        pred_deltaOC = pred_deltaOC.view(-1, 3, coarse_npts).contiguous()  # bs, nc*3, inst_npts -> bs*nc, 3, inst_npts
        pred_deltaOC = tc.index_select(pred_deltaOC, 0, index).contiguous()  # bs x 3 x inst_npts


        # Object Size
        pred_OS = self.reg_size(coarse_global.squeeze(2)) # (B,896)
        pred_OS = pred_OS.view(-1, 3, 1).contiguous() #bs*nc, 3, 1
        pred_OS = tc.index_select(pred_OS, 0, index).contiguous()  # bs x 3 x 1

        return pred_deltaSA, pred_SC, pred_deltaOC, pred_OS, centroid

class SARNet(nn.Module):
    def __init__(self, encoder_decoder):
        super(SARNet, self).__init__()
        self.encoder_decoder = encoder_decoder

    def forward(self, obsv_pcd, temp_pcd, cate_id, mode='train', target_SC=None):
        if mode=='train':
            _preds = self.encoder_decoder.forward(obsv_pcd, temp_pcd, cate_id, mode, target_SC)
            (pred_deltaSA, pred_SC, pred_deltaOC, pred_OS, centroid) = _preds
            pred_SA = pred_deltaSA + temp_pcd

            if target_SC is not None:
                pred_OC = pred_deltaOC + tc.cat((obsv_pcd, target_SC), dim=2) # center of obsv_pcd/G'
            else:
                pred_OC = pred_deltaOC + tc.cat((obsv_pcd, pred_SC), dim=2)

            preds = [pred_SA, pred_SC, pred_OC, pred_OS, centroid]
            return preds

        elif mode=='test':
            _preds  = self.encoder_decoder.forward(obsv_pcd, temp_pcd, cate_id, mode, target_SC=None)
            (pred_deltaSA, pred_SC, pred_deltaOC, pred_OS, centroid) = _preds
            pred_SA = pred_deltaSA + temp_pcd
            # pred_OC = pred_deltaOC + (tc.cat((obsv_pcd, pred_SC), dim=2) - centroid) # center of G
            pred_OC = pred_deltaOC + tc.cat((obsv_pcd, pred_SC), dim=2) # center of obsv_pcd

            preds = (pred_SA, pred_SC, pred_OC, pred_OS, centroid)
            preds = [pred.squeeze().cpu().detach().numpy() for pred in preds]
            tc.cuda.empty_cache()

            return preds

    def set_mode(self, train):
        if train:
            self.train()
        else:
            self.eval()

class Loss_Func(nn.Module):
    def __init__(self):
        super(Loss_Func, self).__init__()

    def deformation_loss(self, pred, targets):
        '''
        :param pred: (B, 3, N_k)
        :param targets: (B, 12, 3, N_k)
        :return:
        '''
        # (12, batch_size, 3, 36) - (batch_size, 3, 36)
        loss = tc.abs(targets.permute(1,0,2,3).contiguous() - pred)
        # (12, batch_size)
        loss = tc.mean(loss, dim=(2, 3))
        # (batch_size)
        loss, index = tc.min(loss, dim=0)

        mean_loss = tc.mean(loss)

        min_gt = []
        for i in range(pred.shape[0]):
            min_gt.append(targets.permute(1,0,2,3).contiguous()[index[i], i, :, :])  # (3,36)

        min_gt = tc.stack(min_gt, dim=0)  # (B, 3,36)

        return mean_loss, min_gt

    def symmetric_loss(self, pred, target):
        '''
        :param pred: (B,3,2N)
        :param target: (B,3,2N)
        :return:
        '''
        loss = tc.abs(target - pred)
        mean_loss = tc.mean(loss)

        return mean_loss

    def center_loss(self, pred, target):
        '''
        :param pred: (B,3,2N)
        :param target: (B,3,1)
        :return:
        '''
        diff = pred - target
        abs_diff = tc.abs(diff)
        mu_x = tc.mean(abs_diff[:, 0, :])  # (1)
        mu_y = tc.mean(abs_diff[:, 1, :])
        mu_z = tc.mean(abs_diff[:, 2, :])
        #
        mu = (mu_x + mu_y + mu_z)
        mean_loss = mu  # + epsilon

        return mean_loss

    def size_loss(self, pred, target):
        '''
        :param pred: (B,3,1)
        :param target: (B,3,1)
        :return:
        '''
        diff = (pred - target) #(B,3,1)
        abs_diff = abs(diff)
        xyz_loss = tc.mean(abs_diff, dim=(0,2))
        mean_loss = tc.sum(xyz_loss)

        return mean_loss

    def forward(self, preds, inputs):
        pred_SA, pred_SC, pred_OC, pred_OS, centroid = preds
        obsv_pcd, temp_pcd, target_SA, target_SC, target_OC, target_OS, cate_id = inputs

        deform_loss, mini_gt = self.deformation_loss(pred_SA, target_SA)
        sym_loss = self.symmetric_loss(pred_SC, target_SC)
        center_loss = self.center_loss(pred_OC, target_OC)
        size_loss = self.size_loss(pred_OS, target_OS)

        loss = deform_loss + sym_loss + center_loss + size_loss


        '''for debug'''
        # debug_folder = './debug'
        # create_folder(debug_folder)
        # save_to_obj_pts(mini_gt[0].cpu().detach().numpy().transpose(), os.path.join(debug_folder, f'{cate_id[0]}_SA_gt.obj'))
        # save_to_obj_pts(obsv_pcd[0].cpu().detach().numpy().transpose(), os.path.join(debug_folder, f'{cate_id[0]}_inst.obj'))
        # save_to_obj_pts((pred_SA[0]).cpu().detach().numpy().transpose(), os.path.join(debug_folder, f'{cate_id[0]}_SA_pred.obj'))
        # save_to_obj_pts(target_SC[0].cpu().detach().numpy().transpose(), os.path.join(debug_folder, f'{cate_id[0]}_SC_gt.obj'))
        # save_to_obj_pts(pred_SC[0].cpu().detach().numpy().transpose(), os.path.join(debug_folder, f'{cate_id[0]}_SC_pred.obj'))

        return loss, deform_loss, sym_loss, center_loss, size_loss
