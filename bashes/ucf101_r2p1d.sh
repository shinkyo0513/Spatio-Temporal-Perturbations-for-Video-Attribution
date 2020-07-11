#!/bin/bash

#$ -N ucf101_r2p1d
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -o taskout/ucf101_r2p1d
#$ -j y
#$ -cwd
source /etc/profile.d/modules.sh

source ${HOME}/.bashrc
conda activate pytorch1.1

python run_all.py --dataset ucf101 --model r2p1d --vis_method ig 
python run_all.py --dataset ucf101 --model r2p1d --vis_method sg
python run_all.py --dataset ucf101 --model r2p1d --vis_method sg2
python run_all.py --dataset ucf101 --model r2p1d --vis_method grad_cam