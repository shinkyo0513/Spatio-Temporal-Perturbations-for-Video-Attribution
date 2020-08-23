#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset ucf101 --model r50l --num_gpu 1 --vis_method g 
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset ucf101 --model r50l --num_gpu 1 --vis_method ig 
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset ucf101 --model r50l --num_gpu 1 --vis_method sg
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset ucf101 --model r50l --num_gpu 1 --vis_method sg2
CUDA_VISIBLE_DEVICES=0 python run_all.py --dataset ucf101 --model r50l --num_gpu 1 --vis_method grad_cam