from transforms import *
import numpy as np

def build_transform(params: List[Dict]):
    transform_list = []
    for param in params:
        if param['type'] == 'GroupScale':
            transform_list.append(GroupScale(scales=param['scales']))
        elif param['type'] == 'GroupFlip':
            transform_list.append(GroupFlip(flip_prob=param['flip_prob']))
        elif param['type'] == 'GroupRandomCrop':
            transform_list.append(GroupRandomCrop(out_size=param['out_size']))


    return Compose(transform_list)

def main():
    transform_params = [
        dict(type='GroupScale', scales=[(298, 224), (342, 256), (384, 288)]),
        dict(type='GroupFlip', flip_prob=0.5),
        dict(type='GroupRandomCrop', out_size=224)
    ]

    composed_transform = build_transform(transform_params)
    num_frames = 8        # Number of frames
    frame_height = 224      # Height of each frame
    frame_width = 224       # Width of each frame
    num_channels = 3        # Number of color channels (RGB)

    # Generate random pixel values between 0 and 255 for each frame
    frames = np.random.randint(0, 256, 
                                (num_frames, frame_height, frame_width, num_channels), 
                                dtype=np.uint8)
    
    transformed_frames = apply_transform_on_frames(frames)
    print(transformed_frames)


if __name__ == "__main__":
    main()
