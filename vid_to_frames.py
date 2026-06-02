from nd2reader import ND2Reader
import os
import numpy as np
from skimage.io import imsave
from skimage.exposure import equalize_adapthist
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import img_as_float, img_as_ubyte

output_dir = "five_output_images"
os.makedirs(output_dir, exist_ok=True)

with ND2Reader("5.nd2") as frames:
    for i, frame in enumerate(frames):
        frame = np.array(frame)

        frame_to_float = img_as_float(frame)

        sigma_est = np.mean(estimate_sigma(frame_to_float))

        smoothed = denoise_nl_means(
            frame_to_float,
            h=1.1 * sigma_est,
            fast_mode=True,
            patch_size=5,
            patch_distance=6
        )

        equalized = equalize_adapthist(smoothed)

        output_path = os.path.join(output_dir, f"5_frame_{i:04d}.png")
        imsave(output_path, img_as_ubyte(equalized))

        print(f"Saved frame {i + 1} to {output_path}")
