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


class SrcFiles:
    def __init__(self, type, value):
        self.type = type
        self.value = value
        self.paths = []

def select_folders_to_zip(src:SrcFiles, dst: str):
    with zipfile.ZipFile(dst, 'w') as output:
        num_files = len(src.paths)
        update_freq = 1
        print(f"Writing total of {num_files} files")
        import ipdb; ipdb.set_trace()
        for fileid, file in enumerate(src.paths):
            # Check if the file exists in the original zip
            # Extract the file to a temporary location
            # Add the extracted file to the new zip
            fn = os.path.basename(file)
            os.system(f"zip -qq -r {fn}.zip {file}")
            if fileid % update_freq == 0:
                print(f"Writing {fileid}:{file}")
            output.write(f"{fn}.zip", arcname=file)
            # Remove the extracted file
            if src.type == "zip":
                os.remove(extracted_path)
    
def select_folders_zip(src_zip_filename: str, tgt_filename: str, paths: Set):
    from multiprocessing import Pool
    zip_f = zipfile.ZipFile(src_zip_filename, 'r')
    src = SrcFiles("zip", zip_f)
    all_files = src.value.namelist()
    #pool = Pool(32)
    #src.paths = pool.map(get_paths, paths)
    all_files = [os.path.join(os.path.basename(src_zip_filename).split('.')[0], x) for x in paths] 
    src.paths = all_files
    select_folders_to_zip(src,tgt_filename)

if __name__ == "__main__":
    root = "%s"%(os.getenv('SLURM_TMPDIR'))
    src_list = zipfile.ZipFile(f"{root}/ucf_BG.zip", 'r').namelist()
    src_list = set([os.path.basename(x[:-1]) for x in src_list if x[-1] == "/" and x != "ucf_BG/"])
    src_list = [os.path.basename(x) for x in src_list]
    tgt_zip = f"{root}/ucf_videos.zip"
    dst = "ucf_modified.zip"
    select_folders_zip(tgt_zip, dst, paths=src_list)