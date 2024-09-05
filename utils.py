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
        self.paths = None

def select_folders_to_zip(src:SrcFiles, dst: str):
    with zipfile.ZipFile(dst, 'w') as output:
        for file in src.paths:
            # Check if the file exists in the original zip
            if src.type == "zip":
                ver_exp = file in src.value.namelist()
            if ver_exp:
                # Extract the file to a temporary location
                if src.type == "zip":
                    extracted_path = src.value.extract(file)
                # Add the extracted file to the new zip
                output.write(extracted_path, arcname=file)
                # Remove the extracted file
                if src.type == "zip":
                    os.remove(extracted_path)
            else:
                print("Entry not found")
    
def select_folders_zip(src_zip_filename: str, tgt_filename: str, path: str=None, path_tgt: str=None):
    from multiprocessing import Pool
    zip_f = zipfile.ZipFile(src_zip_filename, 'r')
    src = SrcFiles("zip", zip_f)
    all_files = src.value.namelist()
    paths = ['%s/%s'%(path_tgt, os.path.basename(os.path.dirname(file))) for file in all_files if file.startswith(path)]
    def get_paths(x):
        return [y for y in all_files if y.startswith('path') and not y[-1] == "/"]
    pool = Pool(32)
    src.paths = pool.map(get_paths, paths)
    import ipdb; ipdb.set_trace()
    select_folders_to_zip(src,tgt_filename)

if __name__ == "__main__":
    fn = "ucf_hmdb.zip"
    src_env = "%s/%s"%(os.getenv('SLURM_TMPDIR'), fn)
    dst = "ucf_modified.zip"
    select_folders_zip(src_env, dst, path="ucf_BG", path_tgt="ucf_videos")