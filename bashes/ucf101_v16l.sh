#!/bin/bash

#$ -N ucf101_v16l
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -o taskout/ucf101_v16l
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python run_all.py --dataset ucf101 --model v16l --vis_method ig 
python run_all.py --dataset ucf101 --model v16l --vis_method sg
python run_all.py --dataset ucf101 --model v16l --vis_method sg2
python run_all.py --dataset ucf101 --model v16l --vis_method grad_cam