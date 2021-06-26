#!/bin/bash

#$ -N xrai
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -o outs_all/xrai_ins_morf_superpixel.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

# python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r2p1d \
# --vis_method xrai --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --only_test

# python evaluation/main_superpixel_causal_metric.py --dataset ucf101 --model r50l \
# --vis_method xrai --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --only_test

# python evaluation/main_superpixel_causal_metric.py --dataset epic --model r2p1d \
# --vis_method xrai --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --only_test

# python evaluation/main_superpixel_causal_metric.py --dataset epic --model r50l \
# --vis_method xrai --mode ins --order most_first --num_step 128 --parallel_size 48 \
# --multi_gpu --num_superpixel 256 --only_test

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r2p1d \
--vis_method xrai --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test

python evaluation/main_superpixel_causal_metric.py --dataset sthsthv2 --model r50l \
--vis_method xrai --mode ins --order most_first --num_step 128 --parallel_size 48 \
--multi_gpu --num_superpixel 256 --only_test