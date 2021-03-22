#!/bin/bash

#$ -N sthsthv2_r50l
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -o outs_sthsthv2/r50l_ins_morf_baseline2.txt
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method g --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method ig --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method blur_ig --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method sg --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method la --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method grad_cam --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
--vis_method eb --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --only_test
