#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python run_all.py --dataset cat_ucf --model r50l --vis_method perturb \
--perturb_niter 1200 --batch_size 10 --master_addr 127.0.1.1 --master_port 29501
