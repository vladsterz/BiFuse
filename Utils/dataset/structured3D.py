import torch
import cv2
import glob
import os
import numpy as np
from torch.utils.data import Dataset


class S3D(Dataset):
    def __init__(self, root_path, width, height, subset = None):
        super().__init__()
        self._root_path = root_path
        self._width = width
        self._height = height
        self._paths = glob.glob(f"{root_path}\\*\\*\\*\\*")
        if subset is not None:
            self._paths = self._paths[:int(len(self._paths) * subset)]
        

        self._resize = (width != 1024) or (height != 512) #S3D

    def __getitem__(self, i):
        path = os.path.join(self._paths[i], "full")
        rgb = cv2.imread(os.path.join(path, "rgb_rawlight.png"))
        depth = cv2.imread(os.path.join(path, "depth.png"), -1).astype(np.float)
        if self._resize:
            rgb = cv2.resize(rgb, (self._width, self._height), cv2.INTER_CUBIC)
            depth = cv2.resize(depth, (self._width, self._height), cv2.INTER_NEAREST)
        return torch.from_numpy(rgb).float().permute(2,0,1) / 255, torch.from_numpy(depth).float().unsqueeze(0) / 1000

    def __len__(self):
        return len(self._paths)






if __name__ == "__main__":
    d = S3D(r"D:\VCL\Users\vlad\Datasets\Structure3D\Structure3D\Structured3D_splited\train", 512,256)
    rgb, dpeth = d[0]
    z = True