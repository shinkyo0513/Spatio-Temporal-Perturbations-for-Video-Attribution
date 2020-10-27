import os
from os.path import join, isdir, isfile
import csv
import pandas as pd
from scipy import stats
import numpy as np
small_thres = 1e-10

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from path_dict import PathDict
path_dict = PathDict()
proj_root = path_dict.proj_root
ds_root = path_dict.ds_root

dataset_name = 'ucf101'
model_name = 'r2p1d'
ds_base = 'outs_'+dataset_name

metric_mode = 'del'
metric_order = 'morf'

txt_file = f'{model_name}_{metric_mode}_{metric_order}.txt'
video_res_dict = {}
column_labels = []
metric_units = set()
with open(join(ds_base, txt_file), 'r') as f:
    for line in f.readlines():
        line = line.strip() 
        if line.startswith('Loaded'):
            continue
        elif line.startswith('Finished:'):
            labels = line[10:].split(', ')
            metric_unit = labels[0]
            metric_units.add(metric_unit)
            vis_meth = labels[-2]
            column_labels.append(f'{metric_unit}-{vis_meth}')
        elif ': ' in line:
            video_name, auc = line.split(': ')
            video_res_dict[video_name] = video_res_dict.get(video_name, list()) + [float(auc), ]
            
df = pd.DataFrame.from_dict(data=video_res_dict, orient='index', columns=column_labels)
print(column_labels)

# Calculating inter-rater reliability (by Weighted Ranking Correlation)
for metric_unit in metric_units:
    # print('\n')
    sltd_methods = ['random', 'g', 'ig', 'sg', 'sg2', 'grad_cam']
    sltd_labels = [f'{metric_unit}-{method}' for method in sltd_methods]

    data = np.array([df[col_label] for col_label in sltd_labels]).transpose(1,0)   # num_inputs (910) x num_methods

    if f'{metric_mode}-{metric_order}' in ['ins-lerf', 'del-morf']:
        rank_descend = True 
    elif f'{metric_mode}-{metric_order}' in ['ins-morf', 'del-lerf']:
        rank_descend = False
    rank = np.array([[sorted(row, reverse=rank_descend).index(item) for item in row] for row in data])

    row_wgt = (data.shape[1] - 1 - rank[:, 0]) / (data.shape[1] - 1)
    cross_wgt = np.outer(row_wgt, row_wgt)

    # Spearman Correlation
    rho, pvalue = stats.spearmanr(data, axis=1)   # num_inputs x num_inputs
    alpha = np.triu(cross_wgt * (rho + 1), k=1).sum() / (np.triu(cross_wgt, k=1).sum() + 1e-10) - 1
    print(f'{metric_mode}-{metric_order}-{metric_unit}: Alpha={alpha:.4f}')

    # sum_rho = np.triu((rho + 1), k=1).sum()
    # avg_row_wgt = row_wgt.sum() / len(row_wgt)
    # print(f'{metric_mode}-{metric_order}-{metric_unit}: Sum_Rho={sum_rho:.4f}, Avg_Wgt={avg_row_wgt:.3f}')

    # # Kendall Correlation
    # rhos = []
    # wgts = []
    # num_sample = data.shape[0]
    # for idx1 in range(0, num_sample-1):
    #     for idx2 in range(idx1+1, num_sample):
    #         rho, pvalue = stats.kendalltau(data[idx1], data[idx2])
    #         wgt = row_wgt[idx1] * row_wgt[idx2]
    #         rhos.append(rho)
    #         wgts.append(wgt)
    # rhos = np.array(rhos)
    # wgts = np.array(wgts)
    # alpha = np.sum((rhos + 1) * wgts) / (np.sum(wgts) + 1e-10) - 1
    # print(f'{metric_mode}-{metric_order}-{metric_unit}: Alpha={alpha:.4f}')

    # reliab_mat = []
    # for ridx, row in df.iterrows():
    #     sltd_nums = sorted([row[col_label] for col_label in sltd_labels])
    #     rank_dict = {col_label: sltd_nums.index(row[col_label]) for col_label in sltd_labels}
    #     reliab_mat.append(rank_dict)

    # from utils.KrippendorffAlpha import krippendorff_alpha, nominal_metric, interval_metric
    # alpha = krippendorff_alpha(reliab_mat, nominal_metric)
    # print(f'{metric_mode}-{metric_order}-{metric_unit}: Alpha={alpha:.4f}')

    # # Inter-rater reliability (by Krippendorff Alpha)
    # # data is in the format
    # # [
    # #     {meth1:value, meth2:value, ...},  # input 1
    # #     {meth1:value, meth3:value, ...},   # input 2
    # #     ...                            # more inputs
    # # ]
    # from utils.KrippendorffAlpha import krippendorff_alpha, nominal_metric, interval_metric
    # reliab_mat = []
    # for rank_row in rank:
    #     rank_dict = {col_label: rank_row[col_idx] for col_idx, col_label in enumerate(sltd_labels)}
    #     reliab_mat.append(rank_dict)

    # from utils.KrippendorffAlpha import krippendorff_alpha, nominal_metric, interval_metric
    # alpha = krippendorff_alpha(reliab_mat, nominal_metric)
    # print(f'{metric_mode}-{metric_order}-{metric_unit} Krip_Alpha={alpha:.4f}')
