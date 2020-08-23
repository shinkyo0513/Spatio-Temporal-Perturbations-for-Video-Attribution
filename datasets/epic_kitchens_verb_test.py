from PIL import Image
from IPython.display import display, HTML
import os
import numpy as np
import random
import pandas as pd
import shutil
import ast

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt


def plot_sequence_images(image_array, save_name):
    ''' Display images sequence as an animation in jupyter notebook
    
    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    dpi = 72.0
    xpixels, ypixels = image_array[0].shape[:2]
    fig = plt.figure(figsize=(ypixels/dpi, xpixels/dpi), dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    print(len(image_array))
    anim = animation.FuncAnimation(fig, animate, frames=len(
        image_array), interval=17, repeat_delay=1, repeat=True)
    # display(HTML(anim.to_html5_video()))
    anim.save(f"./{save_name}.gif", writer="imagemagick")


def vis_verb_videos(video_root):
    # epic_root = '/home/lzq/dataset/epic-kitchens/frames_rgb_flow/rgb/cat_verb_segs/'
    video_names = os.listdir(video_root)
    for video_name in video_names:
        frame_names = os.listdir(os.path.join(video_root, video_name))

        if len(frame_names) < 500:
            frame_names = sorted(frame_names)

            frames = [np.asarray(Image.open(os.path.join(
                video_root, video_name, fn))) for fn in frame_names]
            # print(len(frames))
            # frames = frames
            plot_sequence_images(frames, video_name)
        else:
            print(f"{video_name} has {len(frame_names)} frames.")


# verb_classess_dir = "/home/lzq/dataset/epic-kitchens/annotations/EPIC_verb_classes.csv"
def get_verb_remapping(verb_classess_dir):
    verb_remapping = {}
    verb_classes = pd.read_csv(verb_classess_dir)
    for ridx, row in verb_classes.iterrows():
        class_key = row["class_key"]
        verbs = ast.literal_eval(row["verbs"])
        # print(class_key, verbs)
        for verb in verbs:
            verb_remapping[verb] = class_key
    # print(verb_remapping)
    return verb_remapping

def select_top20_verb(annot_dir, verb_remapping=None):
    seg_annot_df = pd.read_csv(annot_dir)
    verb_num_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        verb = row["verb"]
        if verb_remapping != None:
            verb_keep = verb
            verb = verb_remapping.get(verb, None)
            if verb == None:
                # print(f"{verb_keep} is dropped.")
                continue
        verb_num_dict[verb] = verb_num_dict.get(verb, 0) + 1
    verb_num_dict = {verb: num for verb, num in sorted(
        verb_num_dict.items(), key=lambda item: item[1], reverse=True)}
    top20_verbs = list(verb_num_dict.keys())[:20]
    verb_idx_dict = {verb: idx for idx, verb in enumerate(top20_verbs)}

    verb_segs_dict = {}
    for ridx, row in seg_annot_df.iterrows():
        seg_verb = row["verb"]
        if seg_verb in top20_verbs:
            seg_id, video_id, noun, noun_label, verb, start_frame, stop_frame, bounding_boxes = list(
                dict(row).values())
            verb_id = verb_idx_dict[verb]
            seg_info = [seg_id, video_id, noun, verb_id,
                        verb, start_frame, stop_frame, bounding_boxes]
            verb_segs_dict[verb] = verb_segs_dict.get(verb, []) + [seg_info, ]

    verb_labels = [[verb, verb_idx_dict[verb], verb_num_dict[verb]]
                   for verb in top20_verbs]

    noun_labels_df = pd.read_csv(
        "/home/lzq/ExpEval/my_epic_annot/Valid_seg_top20_noun_labels.csv")
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
            # and seg_noun in top20_nouns:
            if seg_numf < 250 and len(sltd_verb_segs) < samples_per_class:
                sltd_verb_segs.append(seg_info)
            else:
                train_segs.append(seg_info)
        test_segs500 += sltd_verb_segs
        test_segs100 += sltd_verb_segs[:5]
        print(f"{verb}: {len(sltd_verb_segs)}")
    print(len(test_segs500))
    print(len(train_segs))

    annot_path = "/home/lzq/ExpEval/my_epic_annot"
    test_segs100_df = pd.DataFrame.from_records(test_segs100, columns=[
        "seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs100_df.to_csv(
        os.path.join(annot_path, "Valid_seg_top20_verb_100_val_new.csv"), index=False)

    test_segs500_df = pd.DataFrame.from_records(test_segs500, columns=[
        "seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    test_segs500_df.to_csv(
        os.path.join(annot_path, "Valid_seg_top20_verb_500_val_new.csv"), index=False)

    train_segs_df = pd.DataFrame.from_records(train_segs, columns=[
        "seg_id", "video_id", "noun", "verb_label", "verb", "start_frame", "stop_frame", "bounding_boxes"])
    train_segs_df.to_csv(
        os.path.join(annot_path, "Valid_seg_top20_verb_train_new.csv"), index=False)

    verb_labels_df = pd.DataFrame.from_records(
        verb_labels, columns=["verb", "verb_label", "num"])
    verb_labels_df.to_csv(
        os.path.join(annot_path, "Valid_seg_top20_verb_labels_new.csv"), index=False)

if __name__ == "__main__":
    # annot_dir = "/home/lzq/ExpEval/my_epic_annot/Valid_seg.csv"
    # verb_calsses_dir = "/home/lzq/dataset/epic/annotations/EPIC_verb_classes.csv"
    # verb_remapping = get_verb_remapping(verb_calsses_dir)
    # select_top20_verb(annot_dir, verb_remapping)

    new_cat_root = "/home/lzq/dataset/epic/new_cat_verb_segs_val_500"
    vis_verb_videos(new_cat_root)
