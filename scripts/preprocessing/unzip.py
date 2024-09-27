import os
from argparse import ArgumentParser
from multiprocessing import Pool
import glob


def unzip_sample(sample, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    if sample.endswith(".tar"):
        os.system(f"tar xf {sample} -C {output_folder}")
    

def unzip_ek():
    input_folder = os.path.join(os.getenv("SLURM_TMPDIR"), "epic_kitchens/EPIC-KITCHENS/")
    output_folder = os.path.join(os.getenv("SLURM_TMPDIR"), "epic_kitchens/frames_orig/")
    input_tar_files = glob.glob(f"{input_folder}/*/rgb_frames/*tar")
    in_args = []
    for input_file in input_tar_files:
        input_file_no_extn = os.path.basename(input_file).split(".")[0]
        output_file = os.path.join(output_folder, input_file_no_extn)
        in_args.append((input_file, output_file))
    pool = Pool(24)
    pool.starmap(unzip_sample, [x for x in in_args])
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to unzip")
    args = parser.parse_args()
    if args.dataset == "ek":
        unzip_ek()