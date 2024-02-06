#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project   : SAR-Net
# @File      : demo.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2024/2/6 15:53

# @Desciption:


import open3d as o3d
import os
import argparse
import cv2
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
from lib.utils_files import create_folder

# SARNet
import net_respo.net_sarnet as sarnet
import lib.umeyama as umeyama
from lib.utils_pose import load_obj, load_depth, save_to_obj_pts, pc_normalize, draw_detections

# seg 3D
from config.config_seg3d import args
from net_respo.net_seg3d import GCN3D
from lib.utils_seg import get_valid_labels, get_mask, get_catgory_onehot
import configargparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = configargparse.ArgumentParser()
# parser = argparse.ArgumentParser()
parser.add_argument('--config', is_config_file=True,help='config file path')
parser.add_argument('--data_type', type=str, default='real_test', help='cam_val, real_test')
parser.add_argument('--data_folder', type=str, default='../data/NOCS', help='data directory')
parser.add_argument('--results_folder', type=str, default='../results/NOCS', help='root path for saving results')
parser.add_argument('--temp_folder', type=str, default='../data/NOCS/template_FPS', help='root path for saving template')
parser.add_argument('--backproj_npts', type=int, default=2048, help='number of foreground points')
parser.add_argument('--gpu', type=str, default='6', help='GPU to use')
parser.add_argument('--detect_type', type=str, default='mask', help='[mask, bbx]]')
parser.add_argument('--detect_network', type=str, default='mrcnn', help='[mrcnn, yolo, ...]]')
parser.add_argument('--GCN3D_isNeed', type=str2bool, default=True, help='add 3d point segmentation 3DGCN')
parser.add_argument('--pcd_isSave', type=str2bool, default=False, help='save immediate point cloud')
parser.add_argument('--output_pcd_folder', type=str, default='', help='path of immediate point cloud')
parser.add_argument('--SARNet_model_path', type=str, default='', help='model path of SARNet')
parser.add_argument('--GCN3D_model_path', type=str, default='', help='model path of 3DGCN')

opt = parser.parse_args()

cam_fx, cam_fy, cam_cx, cam_cy = 606.9066, 606.2557, 319.9248, 240.7983
intrinsics = np.array([[cam_fx, 0,      cam_cx],
                       [0,      cam_fy, cam_cy],
                       [0,      0,      1     ]])
xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0
categories_seg = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

print('============================')
print('detect_network = [{}]'.format(opt.detect_network))
print('detect_type = [{}]'.format(opt.detect_type))
print('3DGCN segmentation = [{}]'.format(opt.GCN3D_isNeed))
print('============================')

def depth2pcd(depth, choose, crop_reigon=None):
    if crop_reigon is not None:
        rmin, rmax, cmin, cmax = crop_reigon
    else:
        rmin, rmax, cmin, cmax = (0, depth.shape[0], 0, depth.shape[1])

    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis] #(opt.n_pts,1)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]

    # /1000 unit：mm->m
    pt2 = depth_masked / norm_scale
    pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
    points = np.concatenate((pt0, -pt1, -pt2), axis=1)#(N,3)
    return points

def adjust_npts(choose, target_npts):
    if len(choose) > target_npts:
        c_mask = np.zeros(len(choose), dtype=int) # len(choose)
        c_mask[:target_npts] = 1 # len(choose)
        np.random.shuffle(c_mask) #len(choose)
        choose = choose[c_mask.nonzero()] # opt.n_pts 4096
    else:
        # <npts 填充
        choose = np.pad(choose, (0, target_npts-len(choose)), 'wrap')

    return choose

def estimate():
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    categories = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

    # load DONet
    # model_path = './models_pose/net_epoch_9_sizeloss1.model'#'../networks/sarnet_camera275K/00069.model'
    model_path = opt.SARNet_model_path #'../networks/sarnet_camera275K/00069.model'
    encoder_decoder = sarnet.EncoderDecoder()
    net_pose = sarnet.SARNet(encoder_decoder)
    net_pose.eval()
    net_pose.cuda()
    if 'net' in torch.load(model_path).keys():
        net_pose.load_state_dict(torch.load(model_path)['net'])
    else:
        net_pose.load_state_dict(torch.load(model_path))

    print('Load_model in {}.'.format(model_path))

    # load Seg3D
    model_seg_path = opt.GCN3D_model_path#'./models_seg/20210530_5.pth'
    net_Seg = GCN3D(class_num= args.class_num, cat_num=args.cat_num, support_num= args.support, neighbor_num= args.neighbor)
    net_Seg.eval()
    net_Seg = torch.nn.DataParallel(net_Seg)
    net_Seg.load_state_dict(torch.load(model_seg_path)['net'])
    net_Seg = net_Seg.module.cuda()
    print('Load_model in {}.'.format(model_seg_path))

    '''
    ###############################
    # Step 0 Define your image path
    ###############################
    '''
    raw_rgb = cv2.imread('./examples/0000_image.png')[:, :, :3] #BGR
    raw_depth = load_depth('./examples/0000')
    pred_mask = cv2.imread('./examples/0000_mask.png')[..., 0]
    pred_mask = np.array(pred_mask > 0)

    cate_id = int(categories.index('mug')) - 1  # 1~6 -> 0~5

    f_sRT = np.zeros((1, 4, 4), dtype=float)
    f_size = np.zeros((1, 3), dtype=float)
    f_class_id = np.array([[cate_id+1]]) #0:BG, 1-6:object
    output_dir = './examples/'

    '''
    ###############################
    # Step 1 Pre-processing
    ###############################
    '''
    # object by object test

    # rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['pred_bboxes'])
    mask = np.logical_and(pred_mask, raw_depth > 0)
    choose = mask.flatten().nonzero()[0]

    if len(choose) < 32:
        print('{} less than 32'.format(len(choose)))
        return

    choose = adjust_npts(choose, opt.backproj_npts)
    backproj_pcd = depth2pcd(depth=raw_depth, choose=choose) #(N,3)

    backproj_pcd, centroid_seg, s_factor_seg = pc_normalize(backproj_pcd)

    sample = np.random.choice(backproj_pcd.shape[0], size=opt.backproj_npts, replace=False)
    backproj_pcd = backproj_pcd[sample]

    '''Seg3D predict'''
    if opt.GCN3D_isNeed:
        # config
        backproj_npts, _ =  backproj_pcd.shape
        category = categories_seg[cate_id]
        onehot = get_catgory_onehot(category=category, categories=categories_seg).unsqueeze(0).cuda()
        mask_pts = get_mask(category=category, points_num=backproj_npts).unsqueeze(0).cuda()
        backproj_pcd_tensor = torch.from_numpy(backproj_pcd).unsqueeze(0).cuda().float() #(B,npts,3)

        # pred
        with torch.no_grad():
            pred_seg = net_Seg(backproj_pcd_tensor, onehot) #(B, num_pts, num_class)
        pred_seg[mask_pts == 0] = pred_seg.min()
        pred_seg_label = torch.max(pred_seg, 2)[1]  #(B, num_pts)
        obsv_pcd_tensor = backproj_pcd_tensor[pred_seg_label == get_valid_labels(category=category)[1]]
        obsv_pcd = obsv_pcd_tensor.cpu().detach().numpy()
    else:
        obsv_pcd = backproj_pcd.copy()

    if obsv_pcd.shape[0]<32:
        return

    '''3D Filter'''
    if opt.data_type == 'real_test':
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obsv_pcd)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1)
        inlier_cloud = pcd.select_down_sample(ind)
        obsv_pcd = np.asarray(inlier_cloud.points)

    if opt.pcd_isSave:
        create_folder(opt.output_pcd_folder)
        save_to_obj_pts(backproj_pcd, os.path.join(opt.output_pcd_folder, 'src_in.obj'))
        save_to_obj_pts(obsv_pcd, os.path.join(opt.output_pcd_folder, 'seg.obj'))

    '''
    ###############################
    # Step 2 SARNet inference
    ###############################
    '''
    obsv_pcd, centroid, s_factor = pc_normalize(obsv_pcd)  # (N,3)
    obsv_pcd_tensor = torch.from_numpy(obsv_pcd).unsqueeze(0).transpose(2, 1).contiguous() # (3,N)
    obsv_pcd_tensor = obsv_pcd_tensor.cuda().float()


    template_path = os.path.join(opt.temp_folder, '{}_fps_36_normalized.obj'.format(categories[cate_id + 1]))
    # template_path = os.path.join(ROOT_PATH, 'FPS_sample', '{}_fps_128_normalized.obj'.format(categories[cate_id + 1]))
    temp_pcd, _ = load_obj(template_path)
    temp_pcd, _, _ = pc_normalize(temp_pcd) #(N,3)
    temp_pcd_tensor = torch.from_numpy(temp_pcd).unsqueeze(0).transpose(2, 1).contiguous() #(B,3,N_k)
    temp_pcd_tensor = temp_pcd_tensor.cuda().float()

    # pred
    cate_id = np.array(cate_id)
    cate_id_tensor = torch.from_numpy(cate_id).cuda()
    with torch.no_grad():
        preds = net_pose(obsv_pcd_tensor, temp_pcd_tensor, cate_id_tensor, mode='test')
    pred_SA, pred_SC, pred_OC, pred_OS, new_centroid = preds

    if opt.pcd_isSave:
        create_folder(opt.output_pcd_folder)
        save_to_obj_pts(obsv_pcd, os.path.join(opt.output_pcd_folder, 'in.obj'))
        save_to_obj_pts(pred_SA.transpose(), os.path.join(opt.output_pcd_folder, 'SA.obj'))
        save_to_obj_pts(pred_SC.transpose(), os.path.join(opt.output_pcd_folder, 'SC.obj'))

    '''
    ###############################
    # Step 3 Post-processing
    ###############################
    '''
    _, _, _, pred_sRT = umeyama.estimateSimilarityTransform(temp_pcd, pred_SA.transpose(), False)  # (N,3)
    if pred_sRT is None:
        pred_sRT = np.identity(4, dtype=float)

    pred_sRT[1, :3] = -pred_sRT[1, :3]
    pred_sRT[2, :3] = -pred_sRT[2, :3]

    '''recovered pose(RT) by Umeyama composes of size factor(s)'''
    s1 = np.cbrt(np.linalg.det(pred_sRT[:3, :3]))
    pred_sRT[:3, :3] = pred_sRT[:3, :3] / s1

    cluster_center = np.mean(pred_OC, axis=1)

    # pred_sRT[0, 3] =  (centroid_seg[0] + (centroid[0] + (cluster_center[0] + new_centroid[0]) * s_factor) * s_factor_seg)
    # pred_sRT[1, 3] = -(centroid_seg[1] + (centroid[1] + (cluster_center[1] + new_centroid[1]) * s_factor) * s_factor_seg)
    # pred_sRT[2, 3] = -(centroid_seg[2] + (centroid[2] + (cluster_center[2] + new_centroid[2]) * s_factor) * s_factor_seg)
    pred_sRT[0, 3] =  (centroid_seg[0] + (centroid[0] + (cluster_center[0] ) * s_factor) * s_factor_seg)
    pred_sRT[1, 3] = -(centroid_seg[1] + (centroid[1] + (cluster_center[1] ) * s_factor) * s_factor_seg)
    pred_sRT[2, 3] = -(centroid_seg[2] + (centroid[2] + (cluster_center[2] ) * s_factor) * s_factor_seg)


    f_size[0] = pred_OS * s_factor * s_factor_seg
    f_sRT[0] = pred_sRT

    draw_detections(raw_rgb, output_dir, 'd435', '0000', intrinsics, f_sRT, f_size, f_class_id,
                    [], [], [], [], [], [], draw_gt=False, draw_nocs=False)


if __name__ == "__main__":
    print('Estimating ...')
    estimate()




