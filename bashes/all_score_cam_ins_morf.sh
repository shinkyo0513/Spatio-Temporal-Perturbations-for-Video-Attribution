#!/bin/bash

#$ -N score_cam
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -o outs_all/score_cam_ins_morf.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python evaluation/main_causal_metric.py --dataset ucf101 --model r2p1d \
--vis_method score_cam --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --only_test

python evaluation/main_causal_metric.py --dataset ucf101 --model r50l \
--vis_method score_cam --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --only_test

python evaluation/main_causal_metric.py --dataset epic --model r2p1d \
--vis_method score_cam --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --only_test

python evaluation/main_causal_metric.py --dataset epic --model r50l \
--vis_method score_cam --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --only_test

python evaluation/main_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method score_cam --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --only_test
