import os
import json
from glob import glob
import cv2
from tqdm import tqdm


def odgt(img_path):
    img = cv2.imread(img_path)
    _h, _w, _ = img.shape

    odgt_dic = {}
    odgt_dic["fpath_img"] = img_path
    odgt_dic["width"] = _w
    odgt_dic["height"] = _h
    return odgt_dic


if __name__ == "__main__":
    dir_path = os.path.join(os.getcwd(), "./demo/test_images/")
    f_validation = open(f'data/demo.odgt', mode='wt', encoding='utf-8')

    img_list = sorted(glob(dir_path + '*.jpg'))

    for i, img in enumerate(tqdm(img_list)):
        a_odgt = odgt(img)
        f_validation.write(f'{json.dumps(a_odgt)}\n')
