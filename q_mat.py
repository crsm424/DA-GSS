import re
import os.path as osp
from scipy.io import loadmat
import numpy as np
from PIL import Image
import os

root = '/media/huibing/G/PRW'
img_prefix = osp.join(root, 'frames')

query_info = osp.join(root, 'query_info.txt')
with open(query_info,'rb') as f:
    raw = f.readlines()

queries = []
for line in raw:
    linelist = str(line, 'utf-8').split()
    pid = int(linelist[0])
    x,y,w,h = (
        float(linelist[1]),
        float(linelist[2]),
        float(linelist[3]),
        float(linelist[4]),
    )
    roi = np.array([x,y,x+w,y+h]).astype(np.int32)
    roi = np.clip(roi, 0, None)
    img_name = linelist[5][:] + '.jpg'
    query_path = './img/query/' + str(pid)
    if not osp.isdir(query_path):
        os.mkdir(query_path)
    img_path = osp.join(img_prefix,img_name)
    img = Image.open(img_path)
    crop_image = img.crop(roi)
    crop_image.save(osp.join(query_path, str(pid) + '_' + img_name), quality=95, subsampling=0)