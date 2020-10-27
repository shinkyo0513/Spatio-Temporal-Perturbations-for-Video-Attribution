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

dataset_name = 'epic'
ds_base = 'outs_'+dataset_name
txt_files = ['ins_cm.txt', 'ins_cm_16.txt', 'ins_scm.txt', 'del_cm.txt', 'del_cm_16.txt', 'del_scm.txt', 'random_cm.txt', 'random_scm.txt']
video_res_dict = {}
column_labels = []
random_iter_id = 0
random_cm_labels = ['ins_cm', 'ins_cm_16', 'del_cm', 'del_cm_16']
random_scm_labels = ['ins_scm', 'del_scm', 'ins_scm', 'del_scm']
for txt_file in txt_files:
    with open(join(ds_base, txt_file), 'r') as f:
        for line in f.readlines():
            line = line.strip() 
            if (': ' in line) and (not line.startswith('Average')):
                video_name, auc = line.split(': ')
                video_res_dict[video_name] = video_res_dict.get(video_name, list()) + [float(auc), ]
            elif line.startswith('Loaded'):
                label = txt_file[:-4] + '_' + line.split('/')[-1][:-3]
                label = label.replace('_full', '')
                label = label.replace('_test', '')
                label = label.replace('_summed', '')

                if 'random_cm' in txt_file:
                    label = label.replace('random_cm', random_cm_labels[random_iter_id])
                    random_iter_id = random_iter_id + 1 if random_iter_id < 3 else 0
                elif 'random_scm' in txt_file:
                    label = label.replace('random_scm', random_scm_labels[random_iter_id])
                    random_iter_id = random_iter_id + 1 if random_iter_id < 3 else 0
                column_labels.append(label)

df = pd.DataFrame.from_dict(data=video_res_dict, orient='index', columns=column_labels)
# df.to_csv(join(ds_base, 'all.csv'), header=True)

# # Calculating t-stat and p-value for hypothesis validation
# ins_cm_random = np.array(df['ins_cm_ucf101_r2p1d_random'])
# ins_cm_perturb = np.array(df['ins_cm_ucf101_r2p1d_perturb'])
# tstat, pvalue = stats.ttest_rel(ins_cm_random, ins_cm_perturb)
# print('ins_cm_perturb', tstat, pvalue)

# del_cm_random = np.array(df['del_cm_ucf101_r2p1d_random'])
# del_cm_perturb = np.array(df['del_cm_ucf101_r2p1d_perturb'])
# tstat, pvalue = stats.ttest_rel(del_cm_random, del_cm_perturb)
# print('del_cm_perturb', tstat, pvalue)

# ins_cm_16_random = np.array(df['ins_cm_16_ucf101_r2p1d_random'])
# ins_cm_16_perturb = np.array(df['ins_cm_16_ucf101_r2p1d_perturb'])
# tstat, pvalue = stats.ttest_rel(ins_cm_16_random, ins_cm_16_perturb)
# print('ins_cm_16_perturb', tstat, pvalue)

# del_cm_16_random = np.array(df['del_cm_16_ucf101_r2p1d_random'])
# del_cm_16_perturb = np.array(df['del_cm_16_ucf101_r2p1d_perturb'])
# tstat, pvalue = stats.ttest_rel(del_cm_16_random, del_cm_16_perturb)
# print('del_cm_16_perturb', tstat, pvalue)

# ins_scm_random = np.array(df['ins_scm_ucf101_r2p1d_random'])
# ins_scm_perturb = np.array(df['ins_scm_ucf101_r2p1d_perturb'])
# tstat, pvalue = stats.ttest_rel(ins_scm_random, ins_scm_perturb)
# print('ins_scm_perturb', tstat, pvalue)

# del_scm_random = np.array(df['del_scm_ucf101_r2p1d_random'])
# del_scm_perturb = np.array(df['del_scm_ucf101_r2p1d_perturb'])
# tstat, pvalue = stats.ttest_rel(del_scm_random, del_scm_perturb)
# print('del_scm_g', tstat, pvalue)

# # Calculating inter-rater reliability (by Krippendorff Alpha)
# model_name = 'r2p1d'
# metric_types = ['ins_cm', 'del_cm', 'ins_cm_16', 'del_cm_16', 'ins_scm', 'del_scm']
# for metric_type in metric_types:
#     # print('\n')
#     label_pref = f'{metric_type}_{dataset_name}_{model_name}'
#     sltd_methods = ['random', 'ig', 'sg2', 'grad_cam', 'perturb']
#     sltd_labels = [f'{label_pref}_{method}' for method in sltd_methods]

#     sltd_columns = [np.array(df[col_label]) for col_label in sltd_labels]
#     # sltd_columns = [np.random.randn(100) for col_label in sltd_labels]
    
#     # reliab_mat = [{col_label: row[col_label] for col_label in sltd_labels} for ridx, row in df.iterrows()]
#     reliab_mat = []
#     for ridx, row in df.iterrows():
#         sltd_nums = sorted([row[col_label] for col_label in sltd_labels])
#         rank_dict = {col_label: sltd_nums.index(row[col_label]) for col_label in sltd_labels}
#         reliab_mat.append(rank_dict)

#     from utils.KrippendorffAlpha import krippendorff_alpha, nominal_metric, interval_metric
#     alpha = krippendorff_alpha(reliab_mat, nominal_metric)
#     print(f'{label_pref} Alpha={alpha:.4f}')

# Calculating inter-rater reliability (by Weighted Ranking Correlation)
model_name = 'r50l'
metric_types = ['ins_cm', 'del_cm', 'ins_cm_16', 'del_cm_16', 'ins_scm', 'del_scm']
# metric_types = ['ins_cm']
for metric_type in metric_types:
    # print('\n')
    label_pref = f'{metric_type}_{dataset_name}_{model_name}'
    sltd_methods = ['random', 'g', 'ig', 'sg2', 'grad_cam', 'perturb']
    # sltd_methods = ['random', 'g', 'sg', 'sg2', 'perturb']
    sltd_labels = [f'{label_pref}_{method}' for method in sltd_methods]

    data = np.array([df[col_label] for col_label in sltd_labels]).transpose(1,0)   # num_inputs (910) x num_methods
    # print(data[501:505, :])
    # data = np.array([[0, 1, 2], [0, 1, 2], [1, 0, 2]])

    rank_descend = True if 'del' in metric_type else False
    rank = np.array([[sorted(row, reverse=rank_descend).index(item) for item in row] for row in data])
    # print(rank[501:505, :])
    # print('rank:', rank.shape)

    row_wgt = (data.shape[1] - 1 - rank[:, 0]) / (data.shape[1] - 1)
    # row_wgt = (rank[:, 0] == 0)
    # print('row_wgt:', row_wgt)
    # print('row_wgt:', row_wgt.shape)
    cross_wgt = np.outer(row_wgt, row_wgt)
    # print('cross_wgt:', cross_wgt.shape)
    # print('cross_wgt:\n', cross_wgt)

    rho, pvalue = stats.spearmanr(data, axis=1)   # num_inputs x num_inputs
    # print('rho:\n', rho)
    # print('rho:', rho.shape)

    alpha = np.triu(cross_wgt * (rho + 1), k=1).sum() / (np.triu(cross_wgt, k=1).sum() + 1e-10) - 1
    print(f'{label_pref} Alpha={alpha:.4f}')

# # Calculating spearman correlation (inter-method)
# dataset_name = 'ucf101'
# model_name = 'r2p1d'

# metric_types = ['ins_cm', 'del_cm', 'ins_cm_16', 'del_cm_16', 'ins_scm', 'del_scm']
# for metric_type in metric_types:
#     print('\n')
#     label_pref = f'{metric_type}_{dataset_name}_{model_name}'

#     rand_label = f'{label_pref}_random'
#     rand_col = np.array(df[rand_label])

#     comp_methods = ['g', 'sg2', 'perturb']
#     for comp_method in comp_methods:
#         comp_label = f'{label_pref}_{comp_method}'
#         comp_col = np.array(df[comp_label])

#         corr, pvalue = stats.spearmanr(comp_col, rand_col)
#         print(f'{comp_label} VS Random: Rho={corr:.3f}, P={pvalue}')
