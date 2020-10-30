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

# 测试 AUC_ins_morf - AUC_del_morf 的可信度会不会更高

dataset_name = 'ucf101'
model_name = 'r2p1d'
ds_base = 'outs_'+dataset_name

# metric_mode = 'ins'
metric_order = 'morf'

txt_file_ins = f'{model_name}_ins_{metric_order}.txt'
txt_file_del = f'{model_name}_del_{metric_order}.txt'

ins_video_res_dict = {}
ins_column_labels = []
ins_metric_units = set()
with open(join(ds_base, txt_file_ins), 'r') as f:
    for line in f.readlines():
        line = line.strip() 
        if line.startswith('Loaded'):
            continue
        elif line.startswith('Finished:'):
            labels = line[10:].split(', ')
            metric_unit = labels[0]
            ins_metric_units.add(metric_unit)
            vis_meth = labels[-2]
            ins_column_labels.append(f'{metric_unit}-{vis_meth}')
        elif ': ' in line:
            video_name, auc = line.split(': ')
            ins_video_res_dict[video_name] = ins_video_res_dict.get(video_name, list()) + [float(auc), ]
ins_video_res_dict = {k: v for k, v in sorted(ins_video_res_dict.items(), key=lambda item: item[0])}
ins_df = pd.DataFrame.from_dict(data=ins_video_res_dict, orient='index', columns=ins_column_labels)
# print(ins_column_labels)

del_video_res_dict = {}
del_column_labels = []
del_metric_units = set()
with open(join(ds_base, txt_file_del), 'r') as f:
    for line in f.readlines():
        line = line.strip() 
        if line.startswith('Loaded'):
            continue
        elif line.startswith('Finished:'):
            labels = line[10:].split(', ')
            metric_unit = labels[0]
            del_metric_units.add(metric_unit)
            vis_meth = labels[-2]
            del_column_labels.append(f'{metric_unit}-{vis_meth}')
        elif ': ' in line:
            video_name, auc = line.split(': ')
            del_video_res_dict[video_name] = del_video_res_dict.get(video_name, list()) + [float(auc), ]
del_video_res_dict = {k: v for k, v in sorted(del_video_res_dict.items(), key=lambda item: item[0])}
del_df = pd.DataFrame.from_dict(data=del_video_res_dict, orient='index', columns=del_column_labels)
# print(del_column_labels)
            

# Calculating inter-rater reliability (by Weighted Ranking Correlation)
for metric_unit in ins_metric_units:
    # print('\n')
    sltd_methods = ['random', 'g', 'ig', 'sg', 'sg2', 'grad_cam']
    sltd_labels = [f'{metric_unit}-{method}' for method in sltd_methods]
    # print(sltd_labels)

    ins_data = np.array([ins_df[col_label] for col_label in sltd_labels]).transpose(1,0)   # num_inputs (910) x num_methods
    del_data = np.array([del_df[col_label] for col_label in sltd_labels]).transpose(1,0)   # num_inputs (910) x num_methods
    data = ins_data - del_data

    rank_descend = False
    rank = np.array([[sorted(row, reverse=rank_descend).index(item) for item in row] for row in data])

    row_wgt = (data.shape[1] - 1 - rank[:, 0]) / (data.shape[1] - 1)
    cross_wgt = np.outer(row_wgt, row_wgt)

    # Spearman Correlation
    rho, pvalue = stats.spearmanr(data, axis=1)   # num_inputs x num_inputs
    alpha = np.triu(cross_wgt * (rho + 1), k=1).sum() / (np.triu(cross_wgt, k=1).sum() + 1e-10) - 1
    print(f'minus-{metric_order}-{metric_unit}: Alpha={alpha:.4f}')

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
    # print(f'minus-{metric_order}-{metric_unit}: Alpha={alpha:.4f}')

    # Inter-rater reliability (by Krippendorff Alpha)
    # data is in the format
    # [
    #     {meth1:value, meth2:value, ...},  # input 1
    #     {meth1:value, meth3:value, ...},   # input 2
    #     ...                            # more inputs
    # ]
    from utils.KrippendorffAlpha import krippendorff_alpha, nominal_metric, interval_metric
    reliab_mat = []
    for rank_row in rank:
        rank_dict = {col_label: rank_row[col_idx] for col_idx, col_label in enumerate(sltd_labels)}
        reliab_mat.append(rank_dict)

    from utils.KrippendorffAlpha import krippendorff_alpha, nominal_metric, interval_metric
    alpha = krippendorff_alpha(reliab_mat, nominal_metric)
    print(f'minus-{metric_order}-{metric_unit} Krip_Alpha={alpha:.4f}')
