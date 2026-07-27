import os
import numpy as np
import tifffile as tiff
from cellpose import models, train
import cv2
import torch

def create_tif_stack(image_path, mask_path):
    imgs = tiff.imread(image_path)
    masks = tiff.imread(mask_path)
   
    image_list = []
    mask_list = []
   
    for t in range(imgs.shape[0]):
        image_list.append(imgs[t])
        mask_list.append(masks[t])
   
    return image_list, mask_list

def build_pairs(frames, masks, video_starts):
    paired_imgs = []
    paired_masks = []
    for i, (curr, m) in enumerate(zip(frames, masks)):
        if i in video_starts:
            prev = curr
        else:
            prev = frames[i - 1]

        prev_2d = np.squeeze(prev)
        curr_2d = np.squeeze(curr)

        prev_3ch = np.stack([prev_2d, prev_2d, prev_2d], axis=0)  # (3, H, W)
        curr_3ch = np.stack([curr_2d, curr_2d, curr_2d], axis=0)  # (3, H, W)

        stacked = np.concatenate([prev_3ch, curr_3ch], axis=0).astype(np.float32)

        paired_imgs.append(stacked)
        paired_masks.append(m)
    return paired_imgs, paired_masks

images = []
labels = []

data_dir_one = "/home/up202108889/i3s_one_final"

for f in os.listdir(data_dir_one):
    if f == "1_frame_0000.tif":
        img_path = os.path.join(data_dir_one, f)
        mask_path = img_path.replace("1_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_two = "/home/up202108889/i3s_two_final"

for f in os.listdir(data_dir_two):
    if f == "2_frame_0000.tif":
        img_path = os.path.join(data_dir_two, f)
        mask_path = img_path.replace("2_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_three = "/home/up202108889/i3s_three_final"

for f in os.listdir(data_dir_three):
    if f == "3_frame_0000.tif":
        img_path = os.path.join(data_dir_three, f)
        mask_path = img_path.replace("3_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_four = "/home/up202108889/i3s_four_final"

for f in os.listdir(data_dir_four):
    if f == "4_frame_0000.tif":
        img_path = os.path.join(data_dir_four, f)
        mask_path = img_path.replace("4_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

'''data_dir_five = "/home/up202108889/five_comparison_seg_final"

for f in os.listdir(data_dir_five):
    if f == "5_frame_0000.tif":
        img_path = os.path.join(data_dir_five, f)
        mask_path = img_path.replace("5_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)'''

data_dir_aug = "/home/up202108889/augmented_data_complete"

for f, m in zip(sorted(os.listdir(data_dir_aug + "/flip_base")), sorted(os.listdir(data_dir_aug + "/flip_masks"))):
    if (f.startswith("4") and m.startswith("4")):
        continue
    print(f)
    print(m)
    img = cv2.imread(os.path.join(data_dir_aug, "flip_base", f), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(data_dir_aug, "flip_masks", m), cv2.IMREAD_UNCHANGED)
    images.append(img)
    labels.append(mask)

for f, m in zip(sorted(os.listdir(data_dir_aug + "/app_base")), sorted(os.listdir(data_dir_aug + "/app_masks"))):
    if (f.startswith("4") and m.startswith("4")):
        continue
    img = cv2.imread(os.path.join(data_dir_aug, "app_base", f), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(data_dir_aug, "app_masks", m), cv2.IMREAD_UNCHANGED)
    images.append(img)
    labels.append(mask)

print(f"Total frames: {len(images)}")

paired_train_imgs, paired_train_masks = build_pairs(
    images, labels, video_starts=(0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121)
)

print("train_imgs_paired[0].shape:", paired_train_imgs[0].shape) # (6, 2720, 2720)

model = models.CellposeModel(
    pretrained_model='cpsam',
    gpu=True
)

train.train_seg(
    model.net,
    train_data=paired_train_imgs,
    train_labels=paired_train_masks,
   
    normalize=True,
    n_epochs=300,
    learning_rate=1e-4,
    weight_decay=1e-5,
    channel_axis=0,
   
    batch_size=2,
   
    save_path="/home/up202108889",
    model_name="cellpose_new_modification_loocv_four",
   
    save_every=50
)

print("Training process is complete")