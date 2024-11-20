import os
import lmdb
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob

def folder_to_lmdb(image_folder, lmdb_path, resize=None):
    """
    Convert a folder of images to an LMDB dataset.

    Args:
        image_folder (str): Path to the folder containing images.
        lmdb_path (str): Path to save the LMDB database.
        resize (tuple): Optional (width, height) to resize images.
    """
    # List all image files in the folder
    image_paths = [
        os.path.join(image_folder, fname) 
        for fname in os.listdir(image_folder) 
        if fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))
    ]
    
    # Initialize LMDB environment
    map_size = len(image_paths) * 1024 * 1024 * 5  # Estimate the size of the database
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    with env.begin(write=True) as txn:
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            # Load image
            with Image.open(image_path) as img:
                if resize:
                    img = img.resize(resize, Image.ANTIALIAS)
                img = np.array(img)
            
            # Serialize data and store in LMDB
            key = f"{idx:08d}".encode("ascii")
            value = img.tobytes()
            meta = {
                "shape": img.shape,
                "dtype": str(img.dtype),
                "key": key.decode("ascii")
            }
            txn.put(key, value)
            txn.put(f"meta_{key.decode('ascii')}".encode("ascii"), str(meta).encode("ascii"))

    print(f"LMDB dataset created at {lmdb_path}")

def proc(folder):
    lmdb_path = os.path.join(lmdb_root, os.path.join(lmdb_root, os.path.basename(folder)))
    folder_to_lmdb(folder, lmdb_path, resize=(224, 224))
# Example usage
input_folders = os.path.join(os.getenv("SLURM_TMPDIR"), "epic_kitchens/frames_orig/*")
input_folders = glob.glob(f"{input_folders}")
lmdb_root = os.path.join(os.getenv("SLURM_TMPDIR"), "epic_kitchens/frames_lmdb/")
os.makedirs(lmdb_root, exist_ok=True)
from multiprocessing import Pool
pool = Pool(24)
pool.map(proc, input_folders)
