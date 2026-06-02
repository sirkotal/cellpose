import os
import numpy as np
import tifffile as tiff
from cellpose import models, train
import cv2

def create_tif_stack(image_path, mask_path):
    imgs = tiff.imread(image_path)
    masks = tiff.imread(mask_path)
   
    image_list = []
    mask_list = []
   
    for t in range(imgs.shape[0]):
        image_list.append(imgs[t])
        mask_list.append(masks[t])
   
    return image_list, mask_list

images = []
labels = []

data_dir_one = "/home/miguel/Desktop/i3s_one_final"

for f in os.listdir(data_dir_one):
    if f == "1_frame_0000.tif":
        img_path = os.path.join(data_dir_one, f)
        mask_path = img_path.replace("1_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_two = "/home/miguel/Desktop/i3s_two_final"

for f in os.listdir(data_dir_two):
    if f == "2_frame_0000.tif":
        img_path = os.path.join(data_dir_two, f)
        mask_path = img_path.replace("2_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_three = "/home/miguel/Desktop/i3s_three_final"

for f in os.listdir(data_dir_three):
    if f == "3_frame_0000.tif":
        img_path = os.path.join(data_dir_three, f)
        mask_path = img_path.replace("3_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_four = "/home/miguel/Desktop/i3s_four_final"

for f in os.listdir(data_dir_four):
    if f == "4_frame_0000.tif":
        img_path = os.path.join(data_dir_four, f)
        mask_path = img_path.replace("4_frame_0000", "auto_segmentation")
       
        imgs, masks = create_tif_stack(img_path, mask_path)
        images.extend(imgs)
        labels.extend(masks)

data_dir_aug = "/home/miguel/Desktop/augmented_data"

for f, m in zip(sorted(os.listdir(data_dir_aug + "/flip_base")), sorted(os.listdir(data_dir_aug + "/flip_masks"))):
    img = cv2.imread(os.path.join(data_dir_aug, "flip_base", f), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(data_dir_aug, "flip_masks", m), cv2.IMREAD_UNCHANGED)
    images.append(img)
    labels.append(mask)

for f, m in zip(sorted(os.listdir(data_dir_aug + "/app_base")), sorted(os.listdir(data_dir_aug + "/app_masks"))):
    img = cv2.imread(os.path.join(data_dir_aug, "app_base", f), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(data_dir_aug, "app_masks", m), cv2.IMREAD_UNCHANGED)
    images.append(img)
    labels.append(mask)

print(f"Total frames: {len(images)}")

model = models.CellposeModel(
    pretrained_model='cpsam',
    gpu=True
)

train.train_seg(
    model.net,
    train_data=images,
    train_labels=labels,
   
    normalize=True,
    n_epochs=300,
    learning_rate=1e-4,
    weight_decay=1e-5,
   
    batch_size=1,
   
    save_path="/home/miguel/Desktop/",
    model_name="cellpose_augmented",
   
    save_every=50
)

print("Training process is complete")