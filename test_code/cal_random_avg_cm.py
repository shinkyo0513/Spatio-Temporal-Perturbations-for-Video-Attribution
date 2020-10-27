import os
from os.path import join, isdir, isfile
import csv
import pandas as pd
from scipy import stats
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

ds_base = 'outs_ucf101'
txt_file = 'del_cm_test.txt'
res_list = []
with open(join(ds_base, txt_file), 'r') as f:
    for line in f.readlines():
        line = line.strip() 
        if line.startswith('Loaded'):
            res_list.append(list())
        else:
            video_name, auc = line.split(': ')
            res_list[-1].append(float(auc))

for res in res_list:
    print(len(res), sum(res)/len(res))