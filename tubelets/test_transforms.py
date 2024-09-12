from . import build_transform
import numpy as np
from typing import List, Dict

def main():
    transform_params = [
        dict(type='GroupScale', scales=[(298, 224), (342, 256), (384, 288)]),
        dict(type='GroupFlip', flip_prob=0.5),
        dict(type='GroupRandomCrop', out_size=224)
    ]

    transform_params2=[
            dict(
                type='Tubelets',
                region_sampler=dict(
                    scales=[32, 48, 56, 64, 96, 128],
                    ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                    scale_jitter=0.18,
                    num_rois=2,
                ),
                key_frame_probs=[0.5, 0.3, 0.2],
                loc_velocity=5,
                rot_velocity=6,
                shear_velocity=0.066,
                size_velocity=0.0001,
                label_prob=1.0,
                motion_type='gaussian',
                patch_transformation='rotation',
            ),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]

    composed_transform = build_transform(transform_params)
    num_frames = 8        # Number of frames
    frame_height = 224      # Height of each frame
    frame_width = 224       # Width of each frame
    num_channels = 3        # Number of color channels (RGB)

    # Generate random pixel values between 0 and 255 for each frame
    frames = [np.random.randint(0, 256, 
                (frame_height, frame_width, num_channels), 
                dtype=np.uint8)
                for _ in range(num_frames)]
    transformed_frames, _ = composed_transform.apply_image(frames, return_transform_param=True)
    transformed_frames = np.stack(transformed_frames)
    print(transformed_frames.shape, transformed_frames.dtype)


if __name__ == "__main__":
    main()
