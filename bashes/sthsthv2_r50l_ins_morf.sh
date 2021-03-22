#!/bin/bash

#$ -N sthsthv2_r50l
#$ -l rt_F=1
#$ -l h_rt=10:00:00
#$ -o outs_sthsthv2/r50l_ins_morf_perturb2.txt
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python process_perturb_res.py --dataset sthsthv2 --model r50l --only_test --extra_label _core5

python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
--vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
--multi_gpu --new_size 16 --extra_label _core5 --only_test

# python process_perturb_res.py --dataset sthsthv2 --model r50l --only_test

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --only_test

# python process_perturb_res.py --dataset sthsthv2 --model r50l --only_test --extra_label _core8

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core8 --only_test

# python process_perturb_res.py --dataset sthsthv2 --model r50l --only_test --extra_label _core11

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core11 --only_test

# python process_perturb_res.py --dataset sthsthv2 --model r50l --only_test --extra_label _core14

# python evaluation/main_causal_metric.py --dataset sthsthv2 --model r50l \
# --vis_method perturb --mode ins --order most_first --num_step 128 --batch_size 48 \
# --multi_gpu --new_size 16 --extra_label _core14 --only_test