# # %%
# import pickle
# from PIL import Image
# import os
# from os.path import join, isdir
# from os import listdir
# from glob import glob

# import cv2
# import moviepy.editor as mpy
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# import math

# # from ucf101_24_dataset import UCF101_24_Dataset

# crt_dir = os.path.dirname(os.path.realpath(__file__))

# root_dir = '/home/acb11711tx/lzq/dataset/UCF101_24'
# with open(join(root_dir, 'annotations/pyannot.pkl'), 'rb') as f:
#     video_dict = pickle.load(f)

# video_names = list(video_dict.keys())
# num_video = len(video_names)
# for video_idx, video_name in enumerate(video_names):
#     # video_dict[video_name]: (['annotations', 'numf', 'label'])
#     video_annots_list = video_dict[video_name]['annotations']
#     video_numf = video_dict[video_name]['numf']
#     video_label = video_dict[video_name]['label']

#     print(video_name)
#     for video_annot in video_annots_list:
#         # video_annot: dict w (['label', 'ef', 'boxes', 'sf'])
#         seg_sf = video_annot['sf']
#         seg_ef = video_annot['ef']
#         seg_label = video_annot['label']
#         seg_boxes = video_annot['boxes']
#         num_seg_boxes = len(seg_boxes)
#         # print('\t ', seg_sf, seg_ef, seg_label, num_seg_boxes, seg_boxes[0])  
#         print('\t ', type(seg_label), seg_label, num_seg_boxes, seg_boxes[0])

#     # if 'g15_c01' in video_name:
#     #     video_rgb_dir = join(root_dir, 'images', video_name)
#     #     frame_names = sorted(listdir(video_rgb_dir))
#     #     video_frames = [cv2.imread(join(video_rgb_dir, f_name)) for f_name in frame_names]

#     #     video_grounds = [None]*video_numf
#     #     for video_annot in video_annots_list:
#     #         sf = video_annot['sf']
#     #         ef = video_annot['ef']
#     #         for f_idx in range(sf, ef):
#     #             if video_grounds[f_idx] == None:
#     #                 video_grounds[f_idx] = [video_annot['boxes'][f_idx-sf], ]
#     #             else:
#     #                 video_grounds[f_idx].append(video_annot['boxes'][f_idx-sf])

#     #     # Annotation visualization 
#     #     save_dir = join(crt_dir, 'UCF101_24_annot_example', video_name)
#     #     if not isdir(save_dir):
#     #         os.makedirs(save_dir)
#     #         print('Made dir: ', save_dir)
#     #     rec_frames = []
#     #     for f_idx, frame in enumerate(video_frames):
#     #         rec_frame = frame
#     #         if video_grounds[f_idx] != None:
#     #             for ground in video_grounds[f_idx]:
#     #                 x, y, dx, dy = list(ground.astype(np.int16))
#     #                 rec_frame = cv2.rectangle(rec_frame, (x,y), (x+dx,y+dy), 
#     #                             color=(255,0,0), thickness=1)
#     #         cv2.imwrite(join(save_dir, format(f_idx+1, '05d')+'.jpg'), rec_frame)
#     #         rec_frames.append(rec_frame)
#     #     overlap_vid = mpy.ImageSequenceClip(rec_frames, fps=20)
#     #     overlap_vid.write_videofile(join(save_dir, video_name.split('/')[1]+'.mp4'))


#         # # Gaussian perturbation visualization
#         # delete = False
#         # label = 'UCF101_24_annot_delete' if delete else 'UCF101_24_annot_preserv'
#         # save_dir = join(crt_dir, label, video_name)
#         # if not isdir(save_dir):
#         #     os.makedirs(save_dir)
#         #     print('Made dir: ', save_dir)
#         # cmb_frames = []
#         # for f_idx, frame in enumerate(video_frames):
#         #     rsz_frame = cv2.resize(frame, (112,112))
#         #     # rsz_frame = frame
#         #     blur_frame = cv2.GaussianBlur(rsz_frame, (15,15), 21)
#         #     ptb_mask = np.zeros((112,112,1))    #HxWxC
#         #     if video_grounds[f_idx] != None:
#         #         for ground in video_grounds[f_idx]:
#         #             x, y, dx, dy = list(ground.astype(np.int16))
#         #             x = math.floor(112*x / 320)
#         #             y = math.floor(112*y / 240)
#         #             dx = math.ceil(112*dx / 320)
#         #             dy = math.ceil(112*dy / 240)
#         #             ptb_mask[y:y+dy, x:x+dx, :] = 1.0
#         #     if delete:
#         #         cmb_frame = ptb_mask * blur_frame + (1.0-ptb_mask) * rsz_frame
#         #     else:
#         #         cmb_frame = (1.0-ptb_mask) * blur_frame + ptb_mask * rsz_frame
#         #     cmb_frame = cmb_frame.astype(np.uint8)
#         #     cv2.imwrite(join(save_dir, format(f_idx+1, '05d')+'.jpg'), cmb_frame)
#         #     cmb_frames.append(cmb_frame)
#         # cmb_vid = mpy.ImageSequenceClip(cmb_frames, fps=20)
#         # cmb_vid.write_videofile(join(save_dir, video_name.split('/')[1]+'.mp4'))
    

# # %%
# import pickle
# from PIL import Image
# import os
# from os.path import join, isdir
# from os import listdir
# from glob import glob

# root_dir = '/home/acb11711tx/lzq/dataset/UCF101_24'
# with open(join(root_dir, 'annotations/pyannot.pkl'), 'rb') as f:
#     video_dict = pickle.load(f)

# class_names = sorted(os.listdir('/home/acb11711tx/lzq/dataset/UCF101_24/images'))
# class_frame_stat = {key: 0 for key in class_names}

# video_names = list(video_dict.keys())
# for video_idx, video_name in enumerate(video_names):
#     # video_dict[video_name]: (['annotations', 'numf', 'label'])
#     class_name = video_name.split('/')[0]
#     video_annots_list = video_dict[video_name]['annotations']
#     video_numf = video_dict[video_name]['numf']
#     video_label = video_dict[video_name]['label']
#     # print(video_label)

#     class_frame_stat[class_name] = class_frame_stat[class_name] + video_numf

# avg_numf = round(sum(list(class_frame_stat.values())) / len(class_names))

# import matplotlib.pyplot as plt

# colors = []
# for value in list(class_frame_stat.values()):
#     if value > avg_numf:
#         colors.append('blue')
#     else:
#         colors.append('cyan')

# fig, ax = plt.subplots(1,1, figsize=(10, 3))
# ax.bar(list(class_frame_stat.keys()), list(class_frame_stat.values()), color=colors)
# plt.axhline(y=avg_numf, linewidth=1, color='k')
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
# fig.tight_layout()
# plt.show()
# fig.savefig('video_frame_num_stat.png')
# # %%
# 3//4

# # %%
# import random
# ori_testlist_dir = "/home/acb11711tx/lzq/dataset/UCF101_24/splits/testlist01.txt"
# class_video_dict = {}
# with open(ori_testlist_dir, 'r') as f:
#     sample_names = f.readlines()
#     for sample_name in sample_names:
#         sample_name = str(sample_name.rstrip())
#         class_name = sample_name.split("/")[0]
#         class_video_dict[class_name] = class_video_dict.get(class_name, []) + [sample_name, ]
# f.close()

# sltd_testlist = []
# for class_name, video_list in class_video_dict.items():
#     class_list = video_list
#     random.shuffle(class_list)
#     sltd_testlist += class_list[:5]
# # random.shuffle(sltd_testlist)
# # print(sltd_testlist)

# for item in sltd_testlist:
#     print(f"{item}")

# # new_testlist_dir = ori_testlist_dir.replace("01", "02")
# # with open(new_testlist_dir, 'w') as f:
# #     for item in sltd_testlist:
# #         f.write(f"{item}\n")
# # f.close()
# # print(f"Wrote {new_testlist_dir}")
# # %%
# import os
# class_names = sorted(os.listdir('/home/acb11711tx/lzq/dataset/UCF101_24/images'))
# with open("/home/acb11711tx/lzq/VideoPerturb2/datasets/ucf101_24_catName.txt", "w") as f:
#     for class_name in class_names:
#         f.write(f"{class_name}\n")

# %%
import sys
sys.path.append("/home/lzq/lzq/VideoPerturb2")
from model_train.vgg16lstm import vgg16lstm
from model_train.vgg16lstm_caffe import init_vgg16lstm_caffe

import caffe

# caffe_model_dir = "/home/lzq/lzq/VideoPerturb2/model_train"
# init_vgg16lstm_caffe(num_frame=16, num_classes=24, prototxt_save_dir=caffe_model_dir)

caffemodel_save_dir = "/home/lzq/lzq/VideoPerturb2/models/ucf101_24_vgg16lstm_caffe_16_LongRange.caffemodel"
model = vgg16lstm(24)
model.load_weights("/home/lzq/lzq/VideoPerturb2/models/ucf101_24_vgg16lstm_16_LongRange.pt")
model.transfer_to_caffe("/home/lzq/lzq/VideoPerturb2/models/ucf101_24_vgg16lstm_caffe_input16.prototxt", 
                            caffemodel_save_dir=caffemodel_save_dir)

# %%
