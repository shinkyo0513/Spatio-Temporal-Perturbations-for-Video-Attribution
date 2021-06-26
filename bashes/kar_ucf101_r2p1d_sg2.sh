#!/bin/bash

#$ -N ucf_r2p1d
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -o outs_ucf101/r2p1d_kar_sg2.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python model_train/retrain_ucf101_24_model.py --model r2p1d --long_range \
--testlist_idx 1 --multi_gpu --num_epochs 20 \
--vis_method sg2 \
--perturb_ratio 0.05 --perturb_mode keep

python model_train/retrain_ucf101_24_model.py --model r2p1d --long_range \
--testlist_idx 1 --multi_gpu --num_epochs 20 \
--vis_method sg2 \
--perturb_ratio 0.1 --perturb_mode keep

python model_train/retrain_ucf101_24_model.py --model r2p1d --long_range \
--testlist_idx 1 --multi_gpu --num_epochs 20 \
--vis_method sg2 \
--perturb_ratio 0.3 --perturb_mode keep

python model_train/retrain_ucf101_24_model.py --model r2p1d --long_range \
--testlist_idx 1 --multi_gpu --num_epochs 20 \
--vis_method sg2 \
--perturb_ratio 0.5 --perturb_mode keep

python model_train/retrain_ucf101_24_model.py --model r2p1d --long_range \
--testlist_idx 1 --multi_gpu --num_epochs 20 \
--vis_method sg2 \
--perturb_ratio 0.7 --perturb_mode keep

python model_train/retrain_ucf101_24_model.py --model r2p1d --long_range \
--testlist_idx 1 --multi_gpu --num_epochs 20 \
--vis_method sg2 \
--perturb_ratio 0.9 --perturb_mode keep