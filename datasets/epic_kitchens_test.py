#%%
import os
from os.path import join, isdir
import tarfile
import glob
import tqdm
import shutil
import ast
import math
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import moviepy.editor as mpy
import pandas as pd
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

# import sys
# sys.path.append("..")
# from utils.ImageShow import imsc

#%%
# Uncompress tar files
# ds_root = "/mnt/StorageDevice/epic-kitchens/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train"
# undumped_root = "/mnt/StorageDevice/epic-kitchens/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/undumped_train"
def undump_epic_frames (ds_root, undumped_root):
    if not os.path.isdir(undumped_root):
        os.makedirs(undumped_root)

    # Something wrong with P06_07
    tar_dirs = sorted(glob.glob(ds_root+"/*/*.tar"))[83:]
    # print(tar_dirs)
    for tar_dir in tqdm.tqdm(tar_dirs):
        tar_name = tar_dir[-14:-4]

        undumped_dir = os.path.join(undumped_root, tar_name)
        if not os.path.isdir(undumped_dir):
            os.makedirs(undumped_dir)
        print("Make: ", undumped_dir)

        tar = tarfile.open(tar_dir)
        tar.extractall(path=undumped_dir)
        tar.close()
        print(f"{tar_name} is done.")

# %%
# Downsample video frames with 2FPS
# ori_frames_root = "/mnt/StorageDevice/epic-kitchens/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/undumped_train"
# new_frames_root = "/mnt/StorageDevice/epic-kitchens/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/2fps_train"
def downsample_epic_frames (ori_frames_root, new_frames_root):
    if not os.path.isdir(new_frames_root):
        os.makedirs(new_frames_root)

    video_dirs = sorted(glob.glob(ori_frames_root+"/*/*"))[370:]
    for video_dir in tqdm.tqdm(video_dirs):
        if os.path.isdir(video_dir):
            # print(video_dir)
            # numf = len(os.listdir(video_dir))
            # sltd_fidx = [fidx for fidx in range(numf) if fidx%30==0]

            # new_video_dir = os.path.join(new_frames_root, video_dir[-10:])
            # if not os.path.isdir(new_video_dir):
            #     os.makedirs(new_video_dir)
            # print("Made:", new_video_dir)
            
            # for fidx in sltd_fidx:
            #     frame_name = f"frame_{fidx+1:010d}.jpg"
            #     src_frame_dir = os.path.join(video_dir, frame_name)
            #     dst_frame_dir = os.path.join(new_video_dir, frame_name)
            #     shutil.move(src_frame_dir, dst_frame_dir)
            # print(video_dir[-10:], 'copied.')

            fidces = [int(fname[6:-4]) for fname in sorted(os.listdir(video_dir))]
            sltd_fidces = [fidx for fidx in fidces if fidx%30==1]

            new_video_dir = os.path.join(new_frames_root, video_dir[-10:])
            if not os.path.isdir(new_video_dir):
                os.makedirs(new_video_dir)
                print("Made:", new_video_dir)
            
            for fidx in sltd_fidces:
                frame_name = f"frame_{fidx:010d}.jpg"
                src_frame_dir = os.path.join(video_dir, frame_name)
                dst_frame_dir = os.path.join(new_video_dir, frame_name)
                # shutil.move(src_frame_dir, dst_frame_dir)
                # img = cv2.imread(src_frame_dir)
                # cv2.imwrite(dst_frame_dir, img)
                img = Image.open(src_frame_dir)
                img.save(dst_frame_dir)
            print(f"{video_dir[-10:]}: {len(sltd_fidces)} frames copied.")

# %%
# Video (2fps) to segments
# root_dir = "/mnt/StorageDevice/epic-kitchens/"
# ori_frames_root = os.path.join(root_dir, 
#                     "EPIC_KITCHENS_2018/frames_rgb_flow/rgb/2fps_train")
# new_frames_root = os.path.join(root_dir, 
#                     "EPIC_KITCHENS_2018/frames_rgb_flow/rgb/2fps_seg_train")
# annot_dir = os.path.join(root_dir, "annotations", "EPIC_train_action_labels.pkl")
def segment_frames_2fps (ori_frames_root, annot_dir, new_frames_root):              
    train_labels = pd.read_pickle(annot_dir)

    for ridx, row in train_labels.iterrows():
        if ridx <= 100:
            p_id = row["participant_id"]
            v_id = row["video_id"]
            verb = row["verb"]
            noun = row["noun"]

            seg_name = f"{v_id}_{ridx}-{verb}-{noun}"
            seg_dir = os.path.join(new_frames_root, f"{p_id}/{v_id}", seg_name)
            if not os.path.isdir(seg_dir):
                os.makedirs(seg_dir)
            print("Made", seg_dir)

            ori_video_dir = os.path.join(ori_frames_root, f"{p_id}/{v_id}")
            seg_sf = int(row["start_frame"])
            seg_ef = int(row["stop_frame"])
            copied_fn = 0
            for fidx in range(seg_sf, seg_ef+1, 1):
                if fidx%30==1:
                    # print(fidx)
                    frame_name = f"frame_{fidx:010d}.jpg"
                    if os.path.isfile(os.path.join(ori_video_dir, frame_name)):
                        shutil.move(os.path.join(ori_video_dir, frame_name), 
                                    os.path.join(seg_dir, frame_name))
                        copied_fn += 1
            print(f"{copied_fn} frames were copied to {seg_name}")

def segment_frames (ori_frames_root, annot_dir, new_frames_root):              
    train_labels = pd.read_pickle(annot_dir)

    for ridx, row in train_labels.iterrows():
        if ridx >= 11339:
            p_id = row["participant_id"]
            v_id = row["video_id"]
            verb = row["verb"]
            noun = row["noun"]

            seg_name = f"{v_id}_{ridx}-{verb}-{noun}"
            print(f"{ridx}: {seg_name}...")
            seg_dir = os.path.join(new_frames_root, f"{p_id}/{v_id}", seg_name)
            if not os.path.isdir(seg_dir):
                os.makedirs(seg_dir)
                print("\t Made", seg_dir)

            ori_video_dir = os.path.join(ori_frames_root, f"{p_id}/{v_id}")
            seg_sf = int(row["start_frame"])
            seg_ef = int(row["stop_frame"])
            copied_fn = 0
            for fidx in range(seg_sf, seg_ef+1, 1):
                frame_name = f"frame_{fidx:010d}.jpg"
                src_frame_dir = os.path.join(ori_video_dir, frame_name)
                dst_frame_dir = os.path.join(seg_dir, frame_name)

                if os.path.isfile(src_frame_dir):
                    img = Image.open(src_frame_dir)
                    img.save(dst_frame_dir)
                    copied_fn += 1
            if copied_fn == 0:
                print(f"\t Maybe something wrong with {seg_name}.")
            else:
                print(f"\t {copied_fn} frames were copied to {seg_name}")

# Valid segment: segment with at leat one object bounding box GT
def get_valid_segment (seg_frames_root, annot_root):
    import ast
    obj_annot_dir = os.path.join(annot_root, "EPIC_train_object_labels.csv")
    obj_annot = pd.read_csv(obj_annot_dir)

    seg_annot_dir = os.path.join(annot_root, "EPIC_train_action_labels.csv")
    seg_annot = pd.read_csv(seg_annot_dir)

    # Iterate all segs annotation and save seg_annot in a dictionary with key of v_id
    seg_info_dict = {}
    for ridx, row in seg_annot.iterrows():
        v_id = row["video_id"]
        noun = row["noun"]
        seg_sf = row["start_frame"]
        seg_ef = row["stop_frame"]
        seg_info = [ridx, noun, seg_sf, seg_ef]
        seg_info_dict[v_id] = seg_info_dict.get(v_id, []) + [seg_info]
    print(len(seg_info_dict.keys()))

    # Iterate all object labels to find their corresponding seg and savein a dictionary with key of seg_ridx
    seg_bbox_dict = {}
    for obj_ridx, obj_row in obj_annot.iterrows():
        obj_v_id = obj_row["video_id"]
        obj_noun = obj_row["noun"]
        fidx_wbbox = int(obj_row["frame"])

        bbox_lst = ast.literal_eval(obj_row["bounding_boxes"])
        if len(bbox_lst) == 0:
            continue
        for seg_info in seg_info_dict[obj_v_id]:
            seg_ridx, noun, seg_sf, seg_ef = seg_info
            if fidx_wbbox >= seg_sf and fidx_wbbox <= seg_ef and noun == obj_noun:
                if seg_ridx in seg_bbox_dict:
                    seg_bbox_dict[seg_ridx].update({fidx_wbbox: bbox_lst})
                else:
                    seg_bbox_dict[seg_ridx] = {fidx_wbbox: bbox_lst}
    
    valid_segs = []
    noun_num_dict = {}
    for seg_ridx in sorted(seg_bbox_dict.keys()):
        seg_row = seg_annot.iloc[seg_ridx]
        seg_id = seg_row["uid"]
        v_id = seg_row["video_id"]
        noun = seg_row["noun"]
        verb = seg_row["verb"]
        seg_sf = seg_row["start_frame"]
        seg_ef = seg_row["stop_frame"]
        seg_bbox = seg_bbox_dict[seg_ridx]
        if seg_ef-seg_sf+1 >= 32:
            seg_info = [seg_id, v_id, noun, verb, seg_sf, seg_ef, seg_bbox]
            valid_segs.append(seg_info)
            noun_num_dict[noun] = noun_num_dict.get(noun, 0) + 1

    noun_num_dict = {noun: num for noun, num in sorted(noun_num_dict.items(), key=lambda item: item[1], reverse=True)}
    noun_idx_dict = {noun: idx for idx, noun in enumerate(noun_num_dict.keys())}

    valid_segs_widx = []
    for seg_info in valid_segs:
        seg_info_widx = seg_info
        noun = seg_info[2]
        seg_info_widx.insert(3, int(noun_idx_dict[noun]))
        valid_segs_widx.append(seg_info_widx)

    valid_segs_df = pd.DataFrame.from_records(valid_segs_widx, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    valid_segs_df.to_csv(join(annot_root, "Valid_seg.csv"), index=False)
    print(f"{len(valid_segs)}/{len(seg_annot)}")

    noun_info_list = []
    for noun in noun_num_dict.keys():
        noun_info = [noun, noun_idx_dict[noun], noun_num_dict[noun]]
        noun_info_list.append(noun_info)
    noun_info_df = pd.DataFrame.from_records(noun_info_list, columns=
                    ["noun", "noun_label", "num"])
    noun_info_df.to_csv(join(annot_root, "Valid_seg_noun_labels.csv"), index=False)
    print(f"{len(noun_num_dict.keys())} nouns.")

# ClassThres=50, LengthThres=100, 60 nouns with 7804 segments left
# ClassThres=30, LengthThres=100, 83 nouns with 8695 segments left
def filter_valid_segment (annot_dir, class_thres=30, class_balance=False, 
                                    length_thres=100, save_label=None):
    if save_label == None:
        # save_label = f"Valid_seg_{class_thres}B" if class_balance else f"Valid_seg_{class_thres}UB"
        save_label = f"ClassThres={class_thres}_ClassBlc={class_balance}_LenThres={length_thres}"

    valid_segs_df = pd.read_csv(annot_dir)

    # Filter by segment frame legth and count number of each class
    filtered_segs1 = []
    noun_num_dict1 = {}
    for ridx, row in valid_segs_df.iterrows():
        seg_numf = int(row["stop_frame"]) - int(row["start_frame"]) + 1
        if seg_numf >= length_thres:
            seg_info = list(dict(row).values())
            filtered_segs1.append(seg_info)
            noun = row["noun"]
            noun_num_dict1[noun] = noun_num_dict1.get(noun, 0) + 1
    print(f"After filtering by length, {len(filtered_segs1)} segments left.")

    # Remove class with small number of samples
    sltd_nouns = [noun for noun, num in noun_num_dict1.items() if num >= class_thres]
    noun_label_dict = {noun: noun_label for noun_label, noun in enumerate(sltd_nouns)}

    # Filter samples which belong to removed classes and count number of each left class
    filtered_segs2 = []
    noun_num_dict2 = {}
    if not class_balance:
        for seg_info in filtered_segs1:
            seg_noun = seg_info[2]
            if seg_noun in sltd_nouns:
                new_seg_info = seg_info
                new_seg_info.insert(3, noun_label_dict[seg_noun])
                filtered_segs2.append(new_seg_info)
                noun_num_dict2[seg_noun] = noun_num_dict2.get(seg_noun, 0) + 1
    else:
        min_num = noun_num_dict1[sltd_nouns[-1]]
        shuffled_segs = filtered_segs1
        random.shuffle(shuffled_segs)
        for seg_info in shuffled_segs:
            seg_noun = seg_info[2]
            if seg_noun in sltd_nouns and noun_num_dict2.get(seg_noun, 0) < min_num:
                new_seg_info = seg_info
                new_seg_info.insert(3, noun_label_dict[seg_noun])
                filtered_segs2.append(new_seg_info)
                noun_num_dict2[seg_noun] = noun_num_dict2.get(seg_noun, 0) + 1
    print(f"After filter by class (Class_Thres {class_thres}), \
            {len(sltd_nouns)} nouns with {len(filtered_segs2)} segments left.")

    filtered_segs2_df = pd.DataFrame.from_records(filtered_segs2, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    filtered_segs2_df.to_csv(annot_dir.replace(".csv", f"_{save_label}.csv"), index=False)

    noun_list = [(noun, noun_label_dict[noun], noun_num_dict2[noun]) for noun in sltd_nouns]
    noun_list_df = pd.DataFrame.from_records(noun_list, columns=["noun", "label", "num"])
    noun_list_df.to_csv(annot_dir.replace(".csv", f"_{save_label}_noun_labels.csv"), index=False)

def select_topk (annot_dir, noun_info_dir, topk):
    annot_root = "/".join(annot_dir.split("/")[:-1])
    print(annot_root)

    noun_info_df = pd.read_csv(noun_info_dir)
    seg_info_df = pd.read_csv(annot_dir)

    sltd_nouns = []
    for ridx, row in noun_info_df.iterrows():
        if ridx < topk:
            sltd_nouns.append(row["noun"])
    
    sltd_segs = []
    for ridx, row in seg_info_df.iterrows():
        if row["noun"] in sltd_nouns:
            sltd_segs.append(list(dict(row).values()))

    sltd_segs_df = pd.DataFrame.from_records(sltd_segs, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    sltd_segs_df.to_csv(join(annot_root, f"Valid_seg_top{topk}.csv"), index=False)
    print(f"{len(sltd_segs)}/{len(seg_info_df)}")

def list_nouns (noun_info_dir):
    annot_root = "/".join(annot_dir.split("/")[:-1])
    noun_info_df = pd.read_csv(noun_info_dir)
    noun_list = []
    for ridx, row in noun_info_df.iterrows():
        noun, label, num = list(dict(row).values())
        noun_list.append(noun)

    with open(join(annot_root, "epic_top20_catName.txt"), "w") as f:
        for noun in noun_list:
            f.write(f"{noun}\n")
    f.close()

def split_topk (annot_dir, noun_info_dir):
    annot_root = "/".join(annot_dir.split("/")[:-1])
    noun_info_df = pd.read_csv(noun_info_dir)
    noun_num_dict = {}
    for ridx, row in noun_info_df.iterrows():
        noun, label, num = list(dict(row).values())
        noun_num_dict[noun] = num

    seg_annot_df = pd.read_csv(annot_dir)

    noun_segs_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        seg_noun = row["noun"]
        seg_info = list(dict(row).values())
        noun_segs_dict[seg_noun] = noun_segs_dict.get(seg_noun, []) + [seg_info, ]
    noun_stat = {noun: len(noun_segs) for noun, noun_segs in noun_segs_dict.items()}
    print(noun_stat)

    test_segs100 = []
    test_segs500 = []
    samples_per_class = 25
    for noun, noun_segs in noun_segs_dict.items():
        random.shuffle(noun_segs)
        sltd_noun_segs = []
        for seg_info in noun_segs:
            seg_numf = seg_info[6] - seg_info[5] + 1
            seg_verb = seg_info[4]
            seg_bbox_dict = ast.literal_eval(seg_info[7])
            if len(sltd_noun_segs) < samples_per_class:
                if seg_numf>=100 and seg_numf <= 300 and seg_numf/len(seg_bbox_dict.keys()) <= 35:
                    no_multi_bbox_in_one_frame = True
                    new_seg_bbox_dict = {}
                    for fidx, bbox_list in seg_bbox_dict.items():
                        if len(bbox_list) > 1:
                            no_multi_bbox_in_one_frame = False
                            break
                        else:
                            new_seg_bbox_dict[fidx] = bbox_list[0]
                    if no_multi_bbox_in_one_frame:
                        seg_info[7] = new_seg_bbox_dict
                        sltd_noun_segs.append(seg_info)
            else:
                break
        test_segs500 += sltd_noun_segs
        test_segs100 += sltd_noun_segs[:5]

    test_segs500_segid = [seg_info[0] for seg_info in test_segs500]

    train_segs = []
    for ridx, row in seg_annot_df.iterrows():
        seg_noun = row["noun"]
        seg_info = list(dict(row).values())
        if not seg_info[0] in test_segs500_segid:
            train_segs.append(seg_info)
    
    test_segs100_df = pd.DataFrame.from_records(test_segs100, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs100_df.to_csv(annot_dir.replace(".csv", "_100_val.csv"), index=False)

    test_segs500_df = pd.DataFrame.from_records(test_segs500, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs500_df.to_csv(annot_dir.replace(".csv", "_500_val.csv"), index=False)

    train_segs_df = pd.DataFrame.from_records(train_segs, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    train_segs_df.to_csv(annot_dir.replace(".csv", "_train.csv"), index=False)

# Valid segment in one function: segment with at least one frame having only one bbox,
# And restrict segment length, smallest sample number in each noun-class.
def get_valid_segment_onestep (seg_frames_root, annot_root, save_label=None, 
                        class_thres=30, class_balance=False, length_thres=100,
                        save_unfiltered=False):
    import ast
    obj_annot_dir = os.path.join(annot_root, "EPIC_train_object_labels.csv")
    obj_annot = pd.read_csv(obj_annot_dir)

    seg_annot_dir = os.path.join(annot_root, "EPIC_train_action_labels.csv")
    seg_annot = pd.read_csv(seg_annot_dir)

    seg_info_dict = {}
    for ridx, row in seg_annot.iterrows():
        v_id = row["video_id"]
        noun = row["noun"]
        seg_sf = row["start_frame"]
        seg_ef = row["stop_frame"]
        seg_info = [ridx, noun, seg_sf, seg_ef]
        seg_info_dict[v_id] = seg_info_dict.get(v_id, []) + [seg_info]
    print(len(seg_info_dict.keys()))

    seg_bbox_dict = {}
    for obj_ridx, obj_row in obj_annot.iterrows():
        obj_v_id = obj_row["video_id"]
        obj_noun = obj_row["noun"]
        fidx_wbbox = int(obj_row["frame"])

        bbox = ast.literal_eval(obj_row["bounding_boxes"])
        if len(bbox) != 1:
            continue
        for seg_info in seg_info_dict[obj_v_id]:
            seg_ridx, noun, seg_sf, seg_ef = seg_info
            if fidx_wbbox >= seg_sf and fidx_wbbox <= seg_ef and noun == obj_noun:
                if seg_ridx in seg_bbox_dict:
                    seg_bbox_dict[seg_ridx].update({fidx_wbbox: bbox[0]})
                else:
                    seg_bbox_dict[seg_ridx] = {fidx_wbbox: bbox[0]}

    segs_wbbox = []
    for seg_ridx in sorted(seg_bbox_dict.keys()):
        seg_row = seg_annot.iloc[seg_ridx]
        seg_id = seg_row["uid"]
        v_id = seg_row["video_id"]
        noun = seg_row["noun"]
        verb = seg_row["verb"]
        seg_sf = seg_row["start_frame"]
        seg_ef = seg_row["stop_frame"]
        seg_bbox = seg_bbox_dict[seg_ridx]
        seg_info = [seg_id, v_id, noun, verb, seg_sf, seg_ef, seg_bbox]
        segs_wbbox.append(seg_info)
    
    segs_wbbox = []
    filtered_segs1 = []
    noun_num_dict1 = {}
    for seg_ridx in sorted(seg_bbox_dict.keys()):
        seg_row = seg_annot.iloc[seg_ridx]
        seg_id = seg_row["uid"]
        v_id = seg_row["video_id"]
        noun = seg_row["noun"]
        verb = seg_row["verb"]
        seg_sf = seg_row["start_frame"]
        seg_ef = seg_row["stop_frame"]
        seg_bbox = seg_bbox_dict[seg_ridx]
        seg_info = [seg_id, v_id, noun, verb, seg_sf, seg_ef, seg_bbox]
        segs_wbbox.append(seg_info)
        if seg_ef-seg_sf+1 >= length_thres:
            filtered_segs1.append(seg_info)
            noun_num_dict1[noun] = noun_num_dict1.get(noun, 0) + 1
    noun_num_dict1 = {noun: num for noun, num in sorted(noun_num_dict1.items(), key=lambda item: item[1], reverse=True)}
    print(f" - {len(segs_wbbox)} segments have at least one bbox.")
    print(f" - Among these, {len(filtered_segs1)} segments have more than {length_thres} frames,")
    print(f"    belonging to {len(noun_num_dict1.keys())} noun-classes.")

    if save_unfiltered:
        valid_segs_df = pd.DataFrame.from_records(segs_wbbox, columns=
                        ["seg_id", "video_id", "noun", "verb", "start_frame", "stop_frame", "bounding_boxes"])
        valid_segs_df.to_csv(join(annot_root, "Valid_seg_new.csv"), index=False)

    # Remove class with small number of samples
    sltd_nouns = [noun for noun, num in noun_num_dict1.items() if num >= class_thres]
    noun_label_dict = {noun: noun_label for noun_label, noun in enumerate(sltd_nouns)}

    # Filter samples which belong to removed classes and count number of each left class
    filtered_segs2 = []
    noun_num_dict2 = {}
    if not class_balance:
        for seg_info in filtered_segs1:
            seg_noun = seg_info[2]
            if seg_noun in sltd_nouns:
                new_seg_info = seg_info
                new_seg_info.insert(3, noun_label_dict[seg_noun])
                filtered_segs2.append(new_seg_info)
                noun_num_dict2[seg_noun] = noun_num_dict2.get(seg_noun, 0) + 1
    else:
        min_num = noun_num_dict1[sltd_nouns[-1]]
        shuffled_segs = filtered_segs1
        random.shuffle(shuffled_segs)
        for seg_info in shuffled_segs:
            seg_noun = seg_info[2]
            if seg_noun in sltd_nouns and noun_num_dict2.get(seg_noun, 0) < min_num:
                new_seg_info = seg_info
                new_seg_info.insert(3, noun_label_dict[seg_noun])
                filtered_segs2.append(new_seg_info)
                noun_num_dict2[seg_noun] = noun_num_dict2.get(seg_noun, 0) + 1
    print(f" - {len(sltd_nouns)} noun-classes have at least {class_thres} samples,")
    print(f"    containing {len(filtered_segs2)} segments.")

    if save_label == None:
        save_label = f"Valid_seg_new_ClassThres={class_thres}_ClassBlc={class_balance}_LenThres={length_thres}"

    filtered_segs2_df = pd.DataFrame.from_records(filtered_segs2, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    filtered_segs2_df.to_csv(join(annot_root, save_label+".csv"), index=False)

    noun_list = [(noun, noun_label_dict[noun], noun_num_dict2[noun]) for noun in sltd_nouns]
    noun_list_df = pd.DataFrame.from_records(noun_list, columns=["noun", "label", "num"])
    noun_list_df.to_csv(join(annot_root, save_label+"_noun_labels.csv"), index=False)

def split_segment (annot_dir, train_per=0.7):
    import random
    seg_annot_dir = os.path.join(annot_dir)
    seg_annot = pd.read_csv(seg_annot_dir)

    noun_segs_dict = {}
    for ridx, row in seg_annot.iterrows():
        seg_noun = row["noun"]
        noun_segs_dict[seg_noun] = noun_segs_dict.get(seg_noun, []) + [ridx]
    
    train_seg_ridx = []
    val_seg_ridx = []
    for noun in noun_segs_dict.keys():
        seg_ridx_lst = noun_segs_dict[noun]
        num_train_seg = int( len(seg_ridx_lst) * train_per )
        random.shuffle(seg_ridx_lst)
        train_seg_ridx += seg_ridx_lst[:num_train_seg]
        val_seg_ridx += seg_ridx_lst[num_train_seg:]

    train_segs_df = seg_annot.iloc[train_seg_ridx]
    train_segs_df.to_csv(annot_dir.replace(".", "_train."), index=False)

    val_segs_df = seg_annot.iloc[val_seg_ridx]
    val_segs_df.to_csv(annot_dir.replace(".", "_val."), index=False)

    print(f"Train: {len(train_segs_df)}; Val: {len(val_segs_df)}; All: {len(seg_annot)}")

def select_test_segment (annot_dir, num_class=50, samples_per_class=10):
    seg_annot_df = pd.read_csv(annot_dir)

    noun_segs_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        seg_noun = row["noun"]
        seg_info = list(dict(row).values())
        noun_segs_dict[seg_noun] = noun_segs_dict.get(seg_noun, []) + [seg_info, ]
    noun_stat = {noun: len(noun_segs) for noun, noun_segs in noun_segs_dict.items()}
    print(noun_stat)

    test_segs = []
    train_segs = []
    test_verb_stat = {}
    for noun in noun_segs_dict.keys():
        noun_segs = noun_segs_dict[noun]
        random.shuffle(noun_segs)
        sltd_noun_segs = []
        for seg_info in noun_segs:
            seg_numf = seg_info[6] - seg_info[5] + 1
            seg_bbox = ast.literal_eval(seg_info[7])
            seg_verb = seg_info[4]
            if seg_numf <= 300 and seg_numf / len(seg_bbox.keys()) <= 35 and len(sltd_noun_segs) < samples_per_class:
                    sltd_noun_segs.append(seg_info)
                    test_verb_stat[seg_verb] = test_verb_stat.get(seg_verb, 0) + 1
            else:
                train_segs.append(seg_info)
        print(f"{noun}: {len(sltd_noun_segs)} / {len(noun_segs)}")
        test_segs += sltd_noun_segs

    test_segs_df = pd.DataFrame.from_records(test_segs, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs_df.to_csv(annot_dir.replace(".csv", f"_test.csv"), index=False)

    train_segs_df = pd.DataFrame.from_records(train_segs, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    train_segs_df.to_csv(annot_dir.replace(".csv", f"_train.csv"), index=False)

def select_top20 (annot_dir):
    noun_label_dir = '/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_new_ClassThres=60_ClassBlc=False_LenThres=100_noun_labels.csv'
    noun_label_df = pd.read_csv(noun_label_dir)
    noun_num_dict = {}
    for ridx, row in noun_label_df.iterrows():
        noun, label, num = list(dict(row).values())
        noun_num_dict[noun] = num
    noun_num_dict = {noun: num for noun, num in sorted(noun_num_dict.items(), key=lambda item: item[1], reverse=True)}
    print(noun_num_dict)

    top20_nouns = list(noun_num_dict.keys())[:20]
    
    seg_annot_df = pd.read_csv(annot_dir)

    noun_segs_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        seg_noun = row["noun"]
        seg_info = list(dict(row).values())
        noun_segs_dict[seg_noun] = noun_segs_dict.get(seg_noun, []) + [seg_info, ]
    noun_stat = {noun: len(noun_segs) for noun, noun_segs in noun_segs_dict.items()}
    print(noun_stat)

    test_segs100 = []
    test_segs500 = []
    samples_per_class = 25
    for noun in top20_nouns:
        noun_segs = noun_segs_dict[noun]
        random.shuffle(noun_segs)
        sltd_noun_segs = []
        for seg_info in noun_segs:
            seg_numf = seg_info[6] - seg_info[5] + 1
            seg_bbox = ast.literal_eval(seg_info[7])
            seg_verb = seg_info[4]
            if seg_numf <= 300 and seg_numf / len(seg_bbox.keys()) <= 35 and len(sltd_noun_segs) < samples_per_class:
                    sltd_noun_segs.append(seg_info)
        test_segs500 += sltd_noun_segs
        test_segs100 += sltd_noun_segs[:5]
    
    annot_path = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot"
    test_segs100_df = pd.DataFrame.from_records(test_segs100, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs100_df.to_csv(join(annot_path, "Valid_seg_new_top20_test100.csv"), index=False)

    test_segs500_df = pd.DataFrame.from_records(test_segs500, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs500_df.to_csv(join(annot_path, "Valid_seg_new_top20_test500.csv"), index=False)

# ori_frames_root = "/groups1/gcb50205/wyuxi/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train"
# new_frames_root = "/groups1/gcb50205/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/cat_obj_segs"
def concat_segs (ori_frames_root, annot_dir, new_frames_root):
    seg_annot_df = pd.read_csv(annot_dir)
    left_segs = []
    right_segs = []
    num_seg = len(seg_annot_df)
    for ridx, row in seg_annot_df.iterrows():
        seg_id = row["seg_id"]
        v_id = row["video_id"]
        noun = row["noun"]
        noun_id = row["noun_label"]
        verb = row["verb"]
        bbox = row["bounding_boxes"]

        st = row["start_frame"]
        ed = row["stop_frame"]
        numf = ed - st + 1
        
        delta = int(numf / 4)
        mid = random.randint(st+delta, ed-delta)
        new_st = st + mid - ed
        new_ed = ed - st + mid

        left_seg = [seg_id, v_id, noun, noun_id, verb, new_st, mid, st]
        right_seg = [seg_id, v_id, noun, noun_id, verb, mid, new_ed, ed]

        left_segs.append(left_seg+['left'])
        right_segs.append(right_seg+['right'])
    randk = random.sample(list(range(num_seg)), int(num_seg/2))
    new_segs = []
    for i in range(num_seg):
        if i in randk:
            new_segs.append(left_segs[i])
        else:
            new_segs.append(right_segs[i])

    # annot_path = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot"
    new_annot_dir = annot_dir.replace(".csv", "_cat.csv")
    new_segs_df = pd.DataFrame.from_records(new_segs, columns=
                    ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "middle_frame", "ground"])
    new_segs_df.to_csv(new_annot_dir, index=False)

    for new_seg in new_segs:
        seg_id, v_id, noun, noun_id, verb, st, ed, mid, ground = new_seg

        p_id = v_id.split("_")[0]
        video_name = f"{p_id}/{v_id}"
        ori_video_dir = os.path.join(ori_frames_root, f"{video_name}")

        seg_name = f"{v_id}_{seg_id}-{verb}-{noun}"
        print(f"{seg_name} start ...")
        seg_dir = os.path.join(new_frames_root, seg_name)
        if not isdir(seg_dir):
            os.makedirs(seg_dir)

        copied_fn = 0
        for fidx in range(st, ed+1, 1):
            frame_name = f"frame_{fidx:010d}.jpg"
            src_frame_dir = os.path.join(ori_video_dir, frame_name)
            dst_frame_dir = os.path.join(seg_dir, frame_name)
            if os.path.isfile(dst_frame_dir):
                continue

            if os.path.isfile(src_frame_dir):
                img = Image.open(src_frame_dir)
                img.save(dst_frame_dir)
                copied_fn += 1
        print(f"{seg_name} finished, {copied_fn} frames copied.")

def concat_verb_segs (ori_frames_root, annot_dir, new_frames_root):
    seg_annot_df = pd.read_csv(annot_dir)
    left_segs = []
    right_segs = []
    num_seg = len(seg_annot_df)
    for ridx, row in seg_annot_df.iterrows():
        seg_id = row["seg_id"]
        v_id = row["video_id"]
        noun = row["noun"]
        verb_id = row["verb_label"]
        verb = row["verb"]
        bbox = row["bounding_boxes"]

        st = row["start_frame"]
        ed = row["stop_frame"]
        numf = ed - st + 1
        
        delta = int(numf / 4)
        mid = random.randint(st+delta, ed-delta)
        new_st = st + mid - ed
        new_ed = ed - st + mid

        left_seg = [seg_id, v_id, noun, verb_id, verb, new_st, mid, st]
        right_seg = [seg_id, v_id, noun, verb_id, verb, mid, new_ed, ed]

        left_segs.append(left_seg+['left'])
        right_segs.append(right_seg+['right'])
    randk = random.sample(list(range(num_seg)), int(num_seg/2))
    new_segs = []
    for i in range(num_seg):
        if i in randk:
            new_segs.append(left_segs[i])
        else:
            new_segs.append(right_segs[i])

    # annot_path = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot"
    new_annot_dir = annot_dir.replace(".csv", "_cat.csv")
    new_segs_df = pd.DataFrame.from_records(new_segs, columns=
                    ["seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "middle_frame", "ground"])
    new_segs_df.to_csv(new_annot_dir, index=False)

    for new_seg in new_segs:
        seg_id, v_id, noun, verb_id, verb, st, ed, mid, ground = new_seg

        p_id = v_id.split("_")[0]
        video_name = f"{p_id}/{v_id}"
        ori_video_dir = os.path.join(ori_frames_root, f"{video_name}")

        seg_name = f"{v_id}_{seg_id}-{verb}-{noun}"
        print(f"{seg_name} start ...")
        seg_dir = os.path.join(new_frames_root, seg_name)
        if not isdir(seg_dir):
            os.makedirs(seg_dir)

        copied_fn = 0
        for fidx in range(st, ed+1, 1):
            frame_name = f"frame_{fidx:010d}.jpg"
            src_frame_dir = os.path.join(ori_video_dir, frame_name)
            dst_frame_dir = os.path.join(seg_dir, frame_name)
            if os.path.isfile(dst_frame_dir):
                continue

            if os.path.isfile(src_frame_dir):
                img = Image.open(src_frame_dir)
                img.save(dst_frame_dir)
                copied_fn += 1
        print(f"{seg_name} finished, {copied_fn} frames copied.")
    

def visual_segment ():
    from epic_kitchens_dataset import EPIC_Kitchens_Dataset

    ds_path = "/home/acb11711tx/lzq/dataset/epic-kitchens/"
    epic_ds = EPIC_Kitchens_Dataset(ds_path, frames_per_clip=16, 
                                sample_mode="random", num_clips=1, 
                                frame_rate=2, train=True, 
                                perturb="delete", fade_type="black")
    num_sample = len(epic_ds)
    rand_num = random.randint(0, num_sample-1)

    inp_tensor, label, video_name = epic_ds[rand_num]
    video_name = video_name.split("/")[-1]
    print(inp_tensor.shape, label, video_name)

    for fidx in range(inp_tensor.shape[1]):
        plt.subplot(4, 4, fidx+1)
        imsc(inp_tensor[:,fidx,:,:])
        plt.title(fidx, fontsize=6)
    plt.savefig(f"ds_vis_{video_name}.png")

def visual_grounds ():
    frames_root = "/home/acb11711tx/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/seg_train"
    annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_50.csv"

    save_dir = "/home/acb11711tx/lzq/VideoPerturb2/visual_res/visual_grounds"
    if not isdir(save_dir):
        os.makedirs(save_dir)
        print(f"Made {save_dir}")

    seg_annot_df = pd.read_csv(annot_dir)
    num_seg = len(seg_annot_df)

    num_show = 5
    for i in range(num_show):
        seg_ridx = random.randint(0, num_seg-1)
        # seg_ridx = 10
        seg_row_dic = dict(seg_annot_df.iloc[seg_ridx])

        seg_id, video_id, noun, noun_label, verb, seg_sf, seg_ef, seg_bbox = list(seg_row_dic.values())
        p_id = video_id.split("_")[0]
        seg_name = join(p_id, video_id, f"{video_id}_{seg_id}-{verb}-{noun}")
        seg_frames_root = join(frames_root, seg_name)

        seg_bbox = ast.literal_eval(seg_bbox)
        seg_numf = seg_ef - seg_sf + 1
        seg_grounds = [None]*seg_numf
        for fidx in range(seg_sf, seg_ef+1):
            for fidx_wbbox in list(seg_bbox.keys()):
                if abs(fidx-fidx_wbbox)<=15:
                    seg_grounds[fidx-seg_sf] = seg_bbox[fidx_wbbox]
        # print(seg_grounds)
        
        seg_frames = []
        seg_frames_wgrd = []
        for fidx in range(seg_sf, seg_ef+1):
            fdir = join(seg_frames_root, f"frame_{fidx:010d}.jpg")
            fimg = cv2.resize(cv2.imread(fdir), (224, 224))
            seg_frames.append(fimg)

            if seg_grounds[fidx-seg_sf] != None:
                y, x, dy, dx = seg_grounds[fidx-seg_sf]
                x = math.floor(224*x / 1920)
                y = math.floor(224*y / 1080)
                dx = math.ceil(224*dx / 1920)
                dy = math.ceil(224*dy / 1080)
                fimg_wgrd = cv2.rectangle(fimg, (x,y), (x+dx,y+dy), 
                                    color=(255,0,0), thickness=1)
            else:
                fimg_wgrd = fimg
            fimg_wgrd = cv2.cvtColor(fimg_wgrd, cv2.COLOR_BGR2RGB)
            seg_frames_wgrd.append(fimg_wgrd)
            # cv2.imwrite(join(save_dir, f"{fidx}.jpg"), fimg_wgrd)
        # print(seg_frames[0].shape)
        overlap_vid = mpy.ImageSequenceClip(seg_frames_wgrd, fps=60)
        overlap_vid.write_videofile(join(save_dir, seg_name.split('/')[-1]+'.mp4'))

def class_segment_statistic (annot_dir):
    seg_annot = pd.read_csv(annot_dir)
    class_dict = {}
    for ridx, row in seg_annot.iterrows():
        seg_numf = row["stop_frame"] - row["start_frame"] + 1
        noun = row["noun"]
        class_dict[noun] = class_dict.get(noun, []) + [seg_numf]
    class_stat = {}
    for noun, seg_numf_lst in class_dict.items():
        numf_lst = sorted(seg_numf_lst)
        min = seg_numf_lst[0]
        max = seg_numf_lst[-1]
        avg = sum(seg_numf_lst) / len(seg_numf_lst)
        print(f"{noun}: {min}; {max}; {avg:.2f}")

def remove_short_segment (annot_dir, numf_thres):
    seg_annot = pd.read_csv(annot_dir)
    '''
    All number of segments: 15053
    Thres of frame number: 96: 8777 segments left.
    Thres of frame number: 100: 8498 segments left.
    Thres of frame number: 108: 7886 segments left.
    Thres of frame number: 120: 7216 segments left.
    Thres of frame number: 150: 5846 segments left.
    '''
    print(f"All number of segments: {len(seg_annot)}")
    seg_num_thres_dict = {}
    for thres in numf_thres:
        seg_num_thres_dict[thres] = 0
        for ridx, row in seg_annot.iterrows():
            seg_sf = row["start_frame"]
            seg_ef = row["stop_frame"]
            seg_numf = seg_ef - seg_sf + 1
            if seg_numf >= thres:
                seg_num_thres_dict[thres] += 1
        print(f"Thres of frame number: {thres}: {seg_num_thres_dict[thres]} segments left.")

def select_top20_verb (annot_dir):
    seg_annot_df = pd.read_csv(annot_dir)
    verb_num_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        verb = row["verb"]
        verb_num_dict[verb] = verb_num_dict.get(verb, 0) + 1
    verb_num_dict = {verb: num for verb, num in sorted(verb_num_dict.items(), key=lambda item: item[1], reverse=True)}
    top20_verbs = list(verb_num_dict.keys())[:20]
    verb_idx_dict = {verb: idx for idx, verb in enumerate(top20_verbs)}
    
    verb_segs_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        seg_verb = row["verb"]
        if seg_verb in top20_verbs:
            seg_id, video_id, noun, noun_label, verb, start_frame, stop_frame, bounding_boxes = list(dict(row).values())
            verb_id = verb_idx_dict[verb]
            seg_info = [seg_id, video_id, noun, verb_id, verb, start_frame, stop_frame, bounding_boxes]
            verb_segs_dict[verb] = verb_segs_dict.get(verb, []) + [seg_info, ]

    verb_labels = [[verb, verb_idx_dict[verb], verb_num_dict[verb]] for verb in top20_verbs]

    noun_labels_df = pd.read_csv("/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_top20_noun_labels.csv")
    top20_nouns = [row["noun"] for ridx, row in noun_labels_df.iterrows()]

    test_segs100 = []
    test_segs500 = []
    train_segs = []
    samples_per_class = 25
    for verb in top20_verbs:
        verb_segs = verb_segs_dict[verb]
        random.shuffle(verb_segs)
        sltd_verb_segs = []
        for seg_info in verb_segs:
            seg_numf = seg_info[6] - seg_info[5] + 1
            seg_bbox = ast.literal_eval(seg_info[7])
            seg_verb = seg_info[4]
            seg_noun = seg_info[2]
            if len(sltd_verb_segs) < samples_per_class and seg_noun in top20_nouns:
                sltd_verb_segs.append(seg_info)
            else:
                train_segs.append(seg_info)
        test_segs500 += sltd_verb_segs
        test_segs100 += sltd_verb_segs[:5]

    annot_path = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot"
    test_segs100_df = pd.DataFrame.from_records(test_segs100, columns=
                    ["seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs100_df.to_csv(join(annot_path, "Valid_seg_top20_verb_100_val.csv"), index=False)

    test_segs500_df = pd.DataFrame.from_records(test_segs500, columns=
                    ["seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs500_df.to_csv(join(annot_path, "Valid_seg_top20_verb_500_val.csv"), index=False)

    train_segs_df = pd.DataFrame.from_records(train_segs, columns=
                    ["seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    train_segs_df.to_csv(join(annot_path, "Valid_seg_top20_verb_train.csv"), index=False)

    verb_labels_df = pd.DataFrame.from_records(verb_labels, columns=["verb", "verb_label", "num"])
    verb_labels_df.to_csv(join(annot_path, "Valid_seg_top20_verb_labels.csv"), index=False)

def select_verb_top20_100 (top20_500_annot_dir):
    seg_annot_df = pd.read_csv(top20_500_annot_dir)
    seg_info_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        seg_info = list(dict(row).values())
        v_id = row["video_id"]
        seg_id = row["seg_id"]
        verb = row["verb"]
        noun = row["noun"]
        seg_name = f"{v_id}_{seg_id}-{verb}-{noun}"
        seg_info_dict[seg_name] = seg_info

    import torch
    r3d_pred_dict = torch.load("/home/acb11711tx/lzq/VideoPerturb2/epic_verb_r2plus1d_preds.pt")
    vgg_pred_dict = torch.load("/home/acb11711tx/lzq/VideoPerturb2/epic_verb_vgg16lstm_preds.pt")
    video_pred_dict = {name: [r3d_pred_dict[name], vgg_pred_dict[name]] for name in r3d_pred_dict.keys()}

    top100_seg_info = []
    for video_name in list(video_pred_dict.keys()):
        if video_pred_dict[video_name][0] > 0.5 and video_pred_dict[video_name][1] > 0.5:
            top100_seg_info.append(seg_info_dict[video_name])
    
    verb_seg_dict = {}
    for seg_info in top100_seg_info:
        verb = seg_info[4]
        verb_seg_dict[verb] = verb_seg_dict.get(verb, 0) + 1
    print(len(verb_seg_dict.keys()), verb_seg_dict)

    annot_path = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot"
    test_segs100_df = pd.DataFrame.from_records(top100_seg_info[:100], columns=
                    ["seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs100_df.to_csv(join(annot_path, "Valid_seg_top20_verb_100_val.csv"), index=False)

# %%
if __name__ == "__main__":
    print("Start...")
    
    # Downsampling EPIC frames to 2fps
    # ori_frames_root = "/groups1/gcb50205/wyuxi/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train"
    # new_frames_root = "/home/acb11711tx/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/2fps_train"
    # downsample_epic_frames(ori_frames_root, new_frames_root)

    # # Extracting segments 
    # ori_frames_root = "/groups1/gcb50205/wyuxi/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train"
    # new_frames_root = "/home/acb11711tx/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/seg_train"
    # annot_dir = "/home/acb11711tx/lzq/dataset/epic-kitchens/annotations/EPIC_train_action_labels.pkl"
    # segment_frames(ori_frames_root, annot_dir, new_frames_root)

    # frames_root = "/home/acb11711tx/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/seg_train"
    # annot_root = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot"
    # get_valid_segment(frames_root, annot_root)
    # get_valid_segment_onestep(frames_root, annot_root, class_thres=60, class_balance=False, length_thres=100)

    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_new_ClassThres=60_ClassBlc=False_LenThres=100.csv"
    # select_test_segment(annot_dir)
    # select_top20(annot_dir)

    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_50UB.csv"
    # split_segment(annot_dir)
    # remove_short_segment(annot_dir, [96, 100, 108, 120, 150])
    # class_segment_statistic(annot_dir)

    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg.csv"
    # noun_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_noun_labels.csv"
    # select_topk(annot_dir, noun_dir, 20)
    # filter_valid_segment(annot_dir, class_thres=30, class_balance=False, length_thres=100)
    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_ClassThres=50_ClassBlc=False_LenThres=100.csv"
    # split_segment(annot_dir)

    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_top20.csv"
    # noun_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_top20_noun_labels.csv"
    # split_topk(annot_dir, noun_dir)
    # list_nouns(noun_dir)

    # visual_segment()
    # visual_grounds()

    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg.csv"
    # select_top20_verb(annot_dir)

    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_top20_verb_500_val.csv"
    # select_verb_top20_100(annot_dir)

    # ori_frames_root = "/groups1/gcb50205/wyuxi/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train"
    # new_frames_root = "/groups1/gcb50205/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/cat_obj_segs"
    # annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_top20_100_val.csv"
    # concat_segs(ori_frames_root, annot_dir, new_frames_root)

    ori_frames_root = "/groups1/gcb50205/wyuxi/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train"
    new_frames_root = "/groups1/gcb50205/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/cat_verb_segs"
    annot_dir = "/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_top20_verb_100_val.csv"
    concat_verb_segs(ori_frames_root, annot_dir, new_frames_root)

#%%

# noun_name_df = pd.read_csv("/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_ClassThres=50_ClassBlc=False_LenThres=100_noun_labels.csv")
# noun2id = {}
# for ridx, row in noun_name_df.iterrows():
#     noun2id[row["noun"]] = int(row["label"])
# print(noun2id)
# noun_lst = list(noun2id.keys())
# with open("epic_nounName.txt", "w") as f:
#     for noun in noun_lst:
#         f.write(f"{noun}\n")
# f.close()

# top20_500_df = pd.read_csv("/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_new_top20_500_val.csv")
# seg_lst = []
# for ridx, row in top20_500_df.iterrows():
#     seg_info = list(dict(row).values())
#     seg_info[3] = noun2id[seg_info[2]]
#     seg_lst.append(seg_info)

# seg_lst_df = pd.DataFrame.from_records(seg_lst, columns=
#                 ["seg_id", "video_id", "noun", "noun_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
# seg_lst_df.to_csv("/home/acb11711tx/lzq/VideoPerturb2/my_epic_annot/Valid_seg_new_top20_500_val.csv", index=False)
