'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : utils_files.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/11/4 21:27

# @Desciption: 
'''

import os

def log_data_to_file(data, log_file):
    with open(log_file, "a") as data_file:
        data_file.write(str(data) + '\n')

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def create_folders(path):
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
