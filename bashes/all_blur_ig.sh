#!/bin/bash

#$ -N blur_ig
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -o outs_all/blur_ig.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python run_all.py --dataset ucf101 --model r2p1d --only_test --vis_method blur_ig \
--batch_size 4

python run_all.py --dataset ucf101 --model r50l --only_test --vis_method blur_ig \
--batch_size 4

python run_all.py --dataset epic --model r2p1d --only_test --vis_method blur_ig \
--batch_size 4

python run_all.py --dataset epic --model r50l --only_test --vis_method blur_ig \
--batch_size 4

python run_all.py --dataset sthsthv2 --model r2p1d --only_test --vis_method blur_ig \
--batch_size 4

# python run_all.py --dataset epic --model r50l --only_test --vis_method blur_ig \
# --batch_size 4