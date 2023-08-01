import re
import os.path as osp
from scipy.io import loadmat
import numpy as np
from PIL import Image
import os

root = '/media/base/PRW'
img_prefix = osp.join(root, 'frames')
annotation = []
def load_split_name():
    imgs = loadmat(osp.join(root,'frame_test.mat'))['img_index_test']
    return [img[0][0] + '.jpg' for img in imgs]

def get_cam_id(img_name):
    match = re.search(r'c\d', img_name).group().replace('c','')
    return int(match)

imgs = load_split_name()
for img_name in imgs:
    anno_path = osp.join(root, 'annotations', img_name)
    anno = loadmat(anno_path)
    box_key = 'box_new'
    if box_key not in anno.keys():
        box_key = 'anno_file'
    if box_key not in anno.keys():
        box_key = 'anno_previous'
    rois = anno[box_key][:, 1:]
    ids = anno[box_key][:, 0]
    rois = np.clip(rois, 0, None)

    assert len(rois) == len(ids)

    rois[:, 2:] += rois[:,:2]
    ids[ids == -2] = 5555
    annotation.append(
        {
            'img_name': img_name,
            'img_path': osp.join(img_prefix, img_name),
            'boxes': rois.astype(np.int32),
            'pids': ids.astype(np.int32),
            'cam_id': get_cam_id(img_name),
        }
    )

for ann in annotation:
    for i, pid in enumerate(ann['pids']):
        if pid != 5555:
            train_path = './img/gallery/' + str(pid)
            if not osp.isdir(train_path):
                os.mkdir(train_path)
            img = Image.open(ann['img_path'])
            crop_image = img.crop(ann['boxes'][i])
            crop_image.save(osp.join(train_path,str(pid) + '_' + ann['img_name']),quality=95, subsampling= 0)

