  
from __future__ import absolute_import, division, print_function

import os
import io
import sys
import glob
import base64
import json
import argparse

from tqdm import tqdm
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

def ndarr2b64utf8(img):
    img_t = Image.fromarray(img)
    with io.BytesIO() as output:
        img_t.save(output, format="png")
        content = output.getvalue()
        b64_barr = base64.b64encode(content)
        b_string = b64_barr.decode('utf-8')
        return b_string

def b64utf82ndarr(b_string):
    b64_barr = b_string.encode('utf-8')
    content = base64.b64decode(b64_barr)
    img = Image.open(io.BytesIO(content))
    inp_np = np.asarray(img)
    return inp_np


def create_name_pair(inp_paths):
    pair_dict = dict()
    for inp_path in inp_paths:
        base_name, _ = os.path.splitext(os.path.basename(inp_path))
        pair_dict[base_name] = inp_path
    return pair_dict

def save_image(img_path, label_path, output_dir):
    img = Image.open(img_path, "r")
    img_np = np.asarray(img)
    if len(img_np.shape) == 3:
        with open(label_path,'r') as f:
            label = int(f.read())
            if label == 3:
                label = 0
            elif label == 4:
                label = 1
            elif label == 5:
                label = 2
            elif label == 6:
                label = 3
            elif label == 7:
                label = 4
            elif label == 8:
                label = 5
        if img_np.shape[2]==4:
            img_np = img_np[:,:,:3]

        base_name, _ = os.path.splitext(os.path.basename(img_path))

        out_dict = dict()
        out_dict['img_f_name'] = img_path
        out_dict['label_f_name'] = label_path
        b_string = ndarr2b64utf8(img_np)
        out_dict['image'] = b_string
        out_dict['label'] = label
        with open(os.path.join(output_dir, base_name + ".json"), "w") as f:
            json.dump(out_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="\
        path to image input directory. (<input dir>/*.png)")
    parser.add_argument("-l", required=True, type=str, help="\
        path to label input directory. (<input dir>/*.txt)")
    parser.add_argument("-o", default="output_dir")
    args = parser.parse_args()

    img_paths = glob.glob(os.path.join(args.i, "*.png"))
    label_paths = glob.glob(os.path.join(args.l, "*.txt"))
    out_dir = args.o

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


    # crate pair
    name_image = create_name_pair(img_paths)
    name_label = create_name_pair(label_paths)

    pair = dict()
    for k, v in name_image.items():
        if k in name_image:
            pair[k] = {'image': v, 'label': v[:4]+'/labeling/'+k+'.txt'}
    

    for k, v in tqdm(pair.items(), desc="image label pair"):
        save_image(v['image'], v['label'], out_dir)