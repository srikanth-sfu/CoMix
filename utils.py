# Utilities for CoMix
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from models import *
import torchvision
import params
import zipfile
from typing import Set
import glob
from zip_backend import ZipBackend 


def print_line():
    print('-'*100)


def make_variable(tensor, gpu_id=0, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor, gpu_id):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(
        os.path.join(params.model_root, filename)))

def save_model_warmstart(net, filename):
    """Save trained warmstart model."""
    if not os.path.exists(params.warmstart_graph_checkpoint):
        os.makedirs(params.warmstart_graph_checkpoint)
    torch.save(net.state_dict(),
               os.path.join(params.warmstart_graph_checkpoint, filename))
    print("save pretrained model to: {}".format(
        os.path.join(params.warmstart_graph_checkpoint, filename)))


class SrcFiles:
    def __init__(self, type, value):
        self.type = type
        self.value = value
        self.paths = []

def select_folders_to_zip(src:SrcFiles, dst: str):
    # src: contains zipfile obj under value field and paths to copy from it under paths field
    # dst: destination zipfile to copy paths
    num_files = len(src.paths)
    update_freq = 1
    files = []
    print(f"Total of {num_files} videos")
    for fileid, file in enumerate(src.paths):
        if not os.path.exists(file):
            print(file)
        frames = os.listdir(file)
        files.extend([os.path.join(file, x) for x in frames])
    #with open("filelist.txt", "w") as f:
    #    f.write("\n".join(files))
    print("Starting zip operation")
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            print('Adding', file)
            zipf.write(file)

def select_folders_zip(src_zip_filename: str, tgt_filename: str, paths: Set):
    # paths: set of folder names (video frame folders) to copy
    # src_zip_filename: zipped dataset frame folders
    # tgt_filename: zip file where frame folders are stored
    zip_f = zipfile.ZipFile(src_zip_filename, 'r')
    src = SrcFiles("zip", zip_f)
    all_files = src.value.namelist()
    folder_root = all_files[0]
    all_files = [os.path.join(folder_root, x) for x in paths] 
    src.paths = all_files
    select_folders_to_zip(src,tgt_filename)


def main_select_folders():
    # Load dataset only part of domain adaptation sub-component. BG folder list used in this case 
    root = "%s"%(os.getenv('SLURM_TMPDIR'))
    bg_file = "hmdb_BG.zip"
    bg_file_no_ext = bg_file.split(".")[0]
    vid_file = "hmdb_videos_modified.zip"
    src_list = zipfile.ZipFile(f"{root}/{bg_file}", 'r').namelist()
    src_list = set([os.path.basename(x[:-1]) for x in src_list if x[-1] == "/" and x != f"{bg_file_no_ext}/"])
    src_list = [os.path.basename(x) for x in src_list]
    tgt_zip = f"{root}/{vid_file}"
    dst = "hmdb_filtered.zip"
    select_folders_zip(tgt_zip, dst, paths=src_list)


def test_zip_folder(root_dir, zip_fmt, frame_format):
    backend = ZipBackend(zip_fmt=zip_fmt, frame_fmt=frame_format, data_dir=root_dir)
    vids = os.listdir(root_dir)
    for vid in vids:
        if vidid % 100 == 2:
            print('Checked {} videos'.format(vidid))
        out = backend.open(vid)

def main_zip_test():
    root_dir = os.path.join("{}/jester/20bn-jester-v1/".format(os.getenv("SLURM_TMPDIR")))
    zip_fmt = "{}"
    frame_format = "{:05d}.jpg" 
    test_zip_folder(root_dir, zip_fmt, frame_format)


def test_tar_folder(root_dir, zip_fmt, frame_format):
    backend = ZipBackend(zip_fmt=zip_fmt, frame_fmt=frame_format, data_dir=root_dir)
    vids = os.listdir(root_dir)
    for vid in vids:
        if vidid % 100 == 2:
            print('Checked {} videos'.format(vidid))
        out = backend.open(vid)

def main_tar_test():
    root_dir = os.path.join("{}/EPIC-KITCHENS/P01/rgb_frames/".format(os.getenv("SLURM_TMPDIR")))
    tar_fmt = "{}"
    frame_format = "frame_{:10d}.jpg" 
    test_tar_folder(root_dir, tar_fmt, frame_format)

if __name__ == "__main__":
    main_tar_test()
