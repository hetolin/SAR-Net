#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project   : SAR-Net
# @File      : shape_data.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2023/6/17 12:51

# @Desciption: borrowed from https://github.com/mentian/object-deformnet/blob/master/preprocess/shape_data.py.

import os
import glob
import numpy as np
import _pickle as cPickle
from lib.utils_pose import sample_points_from_mesh
from tqdm import tqdm

def save_nocs_model_to_file(obj_model_dir):
    """ Sampling points from mesh model and normalize to NOCS.
        Models are centered at origin, i.e. NOCS-0.5

    """
    mug_meta = {}

    # used for re-align mug category
    special_cases = {'3a7439cfaa9af51faf1af397e14a566d': np.array([0.115, 0.0, 0.0]),
                     '5b0c679eb8a2156c4314179664d18101': np.array([0.083, 0.0, -0.044]),
                     '649a51c711dc7f3b32e150233fdd42e9': np.array([0.0, 0.0, -0.017]),
                     'bf2b5e941b43d030138af902bc222a59': np.array([0.0534, 0.0, 0.0]),
                     'ca198dc3f7dc0cacec6338171298c66b': np.array([0.120, 0.0, 0.0]),
                     'f42a9784d165ad2f5e723252788c3d6e': np.array([0.117, 0.0, -0.026])}

    # CAMERA dataset
    for subset in ['val']:
        camera = {}
        for synsetId in ['02876657', '02880940', '02942699', '02946921', '03642806', '03797390']:
            synset_dir = os.path.join(obj_model_dir, subset, synsetId)
            inst_list = sorted(os.listdir(synset_dir))
            for instance in tqdm(inst_list, ncols=80):
                # no vertices and faces
                if instance == 'd3b53f56b4a7b3b3c9f016d57db96408':
                    continue
                path_to_mesh_model = os.path.join(synset_dir, instance, 'model.obj')
                model_points = sample_points_from_mesh(path_to_mesh_model, 1024, fps=True, ratio=3)
                # flip z-axis in CAMERA
                model_points = model_points * np.array([[1.0, 1.0, -1.0]])
                # re-align mug category
                # if synsetId == '03797390':
                #     if instance == 'b9be7cfe653740eb7633a2dd89cec754':
                #         # skip this instance in train set, improper mug model, only influence training.
                #         continue
                #     if instance in special_cases.keys():
                #         shift = special_cases[instance]
                #     else:
                #         shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
                #         shift = np.array([shift_x, 0.0, 0.0])
                #     model_points += shift
                #     size = 2 * np.amax(np.abs(model_points), axis=0)
                #     scale = 1 / np.linalg.norm(size)
                #     model_points *= scale
                #     mug_meta[instance] = [shift, scale]
                camera[instance] = model_points
        with open(os.path.join(obj_model_dir, 'camera_{}.pkl'.format(subset)), 'wb') as f:
            cPickle.dump(camera, f)

    # Real dataset
    for subset in ['real_test']:
        real = {}
        inst_list = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in tqdm(inst_list, ncols=80):
            instance = os.path.basename(inst_path).split('.')[0]
            bbox_file = inst_path.replace('.obj', '.txt')
            bbox_dims = np.loadtxt(bbox_file)
            scale = np.linalg.norm(bbox_dims)
            model_points = sample_points_from_mesh(inst_path, 1024, fps=True, ratio=3)
            model_points /= scale

            # # relable mug category
            # if 'mug' in instance:
            #     shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
            #     shift = np.array([shift_x, 0.0, 0.0])
            #     model_points += shift
            #     size = 2 * np.amax(np.abs(model_points), axis=0)
            #     scale = 1 / np.linalg.norm(size)
            #     model_points *= scale
            #     mug_meta[instance] = [shift, scale]

            real[instance] = model_points #(N,3)
        with open(os.path.join(obj_model_dir, '{}.pkl'.format(subset)), 'wb') as f:
            cPickle.dump(real, f)

    # # save mug_meta information for re-labeling
    # with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'wb') as f:
    #     cPickle.dump(mug_meta, f)



if __name__ == '__main__':
    obj_model_dir = './data/NOCS/obj_models'
    # Save ground truth models for training deform network
    save_nocs_model_to_file(obj_model_dir)


