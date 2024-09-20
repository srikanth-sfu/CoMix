import os
import cv2
import numpy as np
import tarfile
from typing import List


class TarItem(object):
    """ Tar storage item for loading images from a video.
    Each video clip has one corresponding tar file, which stores
    the video frames (like 00001.jpg, 00002.jpg, ...) or flow information.
    """

    def __init__(self, video_info: str, tar_fmt: str, frame_fmt: str):
        """
        Example
            tar_fmt='20bn-jester-v1/{}.tar',
            frame_fmt='{:05d}.jpg',
        """
        self.tar_path = tar_fmt.format(video_info)
        self.frame_fmt = frame_fmt
        self.frame_tar_fid = None

    def __len__(self):
        if self.frame_tar_fid is None:
            self._check_available(self.tar_path)
            self.frame_tar_fid = tarfile.open(self.tar_path, 'r')
        namelist = self.frame_tar_fid.getnames()
        namelist = [name for name in namelist if name.endswith('.jpg')]
        return len(namelist)

    def filelist(self):
        if self.frame_tar_fid is None:
            self._check_available(self.tar_path)
            self.frame_tar_fid = tarfile.open(self.tar_path, 'r')
        namelist = self.frame_tar_fid.getnames()
        namelist = [name for name in namelist if name.endswith('.jpg')]
        namelist.sort()
        return namelist

    def close(self):
        if self.frame_tar_fid is not None:
            self.frame_tar_fid.close()

    def get_frame(self, indices: List[int]) -> List[np.ndarray]:
        """ Load image frames from the given tar file.
        Args:
            indices: frame index list (0-based index)
        Returns:
            img_list: the loaded image list, each element is a np.ndarray in shape of [H, W 3]
        """
        if isinstance(indices, int):
            indices = [indices]
        img_list = []
        if self.frame_tar_fid is None:
            self._check_available(self.tar_path)
            self.frame_tar_fid = tarfile.open(self.tar_path, 'r')
        filelist = self.filelist()
        for idx in indices:
            file_name = filelist[idx]
            img = self.load_image_from_tar(self.frame_tar_fid, file_name, cv2.IMREAD_COLOR)
            img_list.append(img)
        return img_list

    @staticmethod
    def load_image_from_tar(tar_fid, file_name, flag=cv2.IMREAD_COLOR):
        file_content = tar_fid.extractfile(file_name).read()
        img = cv2.imdecode(np.fromstring(file_content, dtype=np.uint8), flag)
        return img

    @staticmethod
    def _check_available(tar_path):
        if tar_path is None:
            raise ValueError("There is not file path defined in video annotations")
        if not os.path.isfile(tar_path):
            raise FileNotFoundError("Cannot find tar file {}".format(tar_path))


class TarBackend(object):

    def __init__(self,
                 tar_fmt: str,
                 frame_fmt: str = 'img_{:05d}.jpg',
                 data_dir: str = None):
        if data_dir is not None:
            tar_fmt = os.path.join(data_dir, tar_fmt)
        self.tar_fmt = tar_fmt
        self.frame_fmt = frame_fmt

    def open(self, video_info, frame_inds=None) -> TarItem:
        storage_obj = TarItem(video_info, self.tar_fmt, self.frame_fmt)
        return self.get_single_clip(storage_obj,frame_inds=frame_inds)

    def get_single_clip(self, storage_obj, frame_inds):
        """ Get single video clip according to the video_info query."""
        if frame_inds is None:
            frame_inds = list(range(storage_obj.__len__()))  
        img_list = storage_obj.get_frame(frame_inds)
        return img_list