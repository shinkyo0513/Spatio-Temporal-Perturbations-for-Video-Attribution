#!/bin/bash

#$ -N sthsthv2_r2p1d
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -o outs_sthsthv2/r2p1d_ins_morf_perturb_superpixel.txt
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test --extra_label _core5 

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test --extra_label _core8

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test --extra_label _core11

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method perturb --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test --extra_label _core14
