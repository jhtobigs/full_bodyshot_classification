import glob
import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class StyleDataset(Dataset):
    """ Dataset class
    {0:pass, 1:fail}
    """
    def __init__(self, path, mode, transform=None):
        super(Dataset, self).__init__()
        self.mode = mode
        self.path = os.path.join(path, mode)
        img_pass = glob.glob(os.path.join(self.path, 'pass/*.jpg'))
        img_fail = glob.glob(os.path.join(self.path, 'fail/*.jpg'))

        self.images = img_pass + img_fail
        self.labels = [0]*len(img_pass) + [1]*len(img_fail)
        if len(self.images) != len(self.labels):
            raise ValueError('You should match image/label data number.')

        self.transform = transform

    def __getitem__(self, idx):
        imgpath, label = self.images[idx], self.labels[idx]
        image = Image.open(imgpath)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.labels)
