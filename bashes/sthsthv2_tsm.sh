#!/bin/bash

#$ -N sthsthv2_tsm
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -o outs_sthsthv2/tsm_baselines.txt
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method random --batch_size 4
python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method g --batch_size 4
python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method la --batch_size 4
python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method ig --batch_size 4
python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method sg --batch_size 4
python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method grad_cam --batch_size 4
python run_all.py --dataset sthsthv2 --model tsm --only_test --vis_method eb --batch_size 4