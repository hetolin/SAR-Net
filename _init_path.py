'''
# -*- coding: utf-8 -*-
# @Project   : code
# @File      : _init_path.py
# @Software  : PyCharm

# @Author    : hetolin
# @Email     : hetolin@163.com
# @Date      : 2021/5/7 18:08

# @Desciption: 
'''

import os
import sys

sys.path.insert(0, os.getcwd())

def add_path(path):
    if path not in sys.path: sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

root_path = os.path.abspath(os.path.join(this_dir, '..'))
add_path(root_path)
