import numpy as np
import glob
import tifffile as tiff
from PIL import Image

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union <= 0:
        return 0
    return inter / union # jaccard

def dice(a, b):
    inter = np.logical_and(a, b).sum()

    if a.sum() + b.sum() <= 0:
        return 0
    return (2 * inter) / (a.sum() + b.sum()) # dice score

def calculate_seg_metrics(ground_truth, seg_mask):
    ground_truth_bool = ground_truth.astype(bool)
    seg_mask_bool = seg_mask.astype(bool)

    iou_score = iou(ground_truth_bool, seg_mask_bool)
    dice_score = dice(ground_truth_bool, seg_mask_bool)

    return iou_score, dice_score

# ground_truth_masks = sorted(glob.glob("*_gt_masks.png"))
ground_truth_masks = tiff.imread("auto_segmentation.tif")
seg_masks = sorted(glob.glob("*_masks.png"))

seg_masks_stack = np.stack([np.array(Image.open(mask)) for mask in seg_masks])

results = []

for i in range(len(seg_masks) - 1):
    score = calculate_seg_metrics(ground_truth_masks[i], seg_masks_stack[i])
    results.append(score)

print(results)