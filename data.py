
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from scipy.stats import bernoulli
import random
import numpy as np
from skimage import transform
from skimage.transform import resize
import os
import glob
import json
import io
import base64
from PIL import Image


def b64utf82ndarr(b_string):
    b64_barr = b_string.encode('utf-8')
    content = base64.b64decode(b64_barr)
    img = Image.open(io.BytesIO(content))
    inp_np = np.asarray(img)
    return inp_np


class TopologyDataset(data.Dataset):
    def __init__(self, in_dir, transform=None):
        self.json_paths = glob.glob(os.path.join(in_dir+'/json_image', "*.json")) 
        self.transform = transform

    def __len__(self):
        return len(self.json_paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item_path = self.json_paths[idx]
        with open(item_path, 'r') as f:
            in_json = json.load(f)

            sample = list()
            label = list()
            sample.append(b64utf82ndarr(in_json['image']))
            label.append(in_json['label'])
        sample = np.asarray(sample)
        sample = sample.reshape(sample.shape[0]*sample.shape[1],sample.shape[2],sample.shape[3])

        if self.transform:
            sample = self.transform(sample)

        return sample, label[0]

class Normalize(object):
    def __init__(self, mean=0.5,std=0.5):
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        sample = sample / 255.

        sample[sample<0.5]=0.0


        sample = (sample - self.mean) / self.std

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #print("before",sample.shape)
        sample = np.transpose(sample, (2, 0, 1))
        #print("after",sample.shape)
        return sample

class Rescale(object):
    """주어진 사이즈로 샘플크기를 조정합니다.

    Args:
        output_size(tuple or int) : 원하는 사이즈 값
            tuple인 경우 해당 tuple(output_size)이 결과물(output)의 크기가 되고,
            int라면 비율을 유지하면서, 길이가 작은 쪽이 output_size가 됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        sample = transform.resize(sample, (new_h, new_w))

        return sample

class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.

    Args:
        output_size (tuple or int): 줄이고자 하는 크기입니다.
                        int라면, 정사각형으로 나올 것 입니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample = sample[top: top + new_h,
                      left: left + new_w]

        return sample

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        if bernoulli.pmf(random.randrange(1, 11)%2,self.p) == 0.3:
            sample = np.fliplr(sample)

        return sample

if __name__ == "__main__":
    dataset = TopologyDataset(in_dir='./data',transform=transforms.Compose([Normalize(), Rescale(256), RandomCrop(224),ToTensor(), RandomHorizontalFlip(p=0.3)]))
    for i in range(len(dataset)):
        sample, label = dataset[i]

        print(i, sample.shape, label)
