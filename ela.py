import json

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os

from skimage.util import view_as_windows
from sklearn.metrics import f1_score

aut = 'testing/authentic'
doc = 'testing/doctored'
config_dir = 'configs'
temp = 'resave_images'

auts = os.listdir(aut)
auts.sort()

for aut_img in auts:
    configs = json.load(open(config_dir + '/' + aut_img.split('.')[0][1:] + '.json', 'r'))

    aut_im = Image.open(aut + '/' + aut_img)
    aut_im.save(temp + '/aut_temp.jpg', 'JPEG', quality=90)
    aut_resaved_im = Image.open(temp + '/aut_temp.jpg')
    aut_ela_im = np.abs(np.array(aut_im) - np.array(aut_resaved_im))

    doc_im = Image.open(doc + '/' + aut_img)
    doc_im.save(temp + '/doc_temp.jpg', 'JPEG', quality=90)
    doc_resaved_im = Image.open(temp + '/doc_temp.jpg')
    doc_ela_im = np.abs(np.array(doc_im) - np.array(doc_resaved_im))
    for config in configs:
        point = config['point']
        h = config['h']
        w = config['w']
        ori_y = point[1] + 1
        ori_x = point[0] + 1
        size_h = min(h - 2, 32)
        size_w = min(w - 2, 32)
        mean_aut = np.mean(aut_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :])
        mean_doc = np.mean(doc_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :])
        aut_ela_im_window = view_as_windows(aut_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :], (size_h, size_w, 3), step=1)
        doc_ela_im_window = view_as_windows(doc_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :], (size_h, size_w, 3), step=1)
        aut_ela_im_window = aut_ela_im_window.mean(axis=5).mean(axis=4).mean(axis=3).flatten()
        doc_ela_im_window = doc_ela_im_window.mean(axis=5).mean(axis=4).mean(axis=3).flatten()
        gt = (aut_ela_im_window > mean_aut).astype(np.int)
        pred = (doc_ela_im_window > mean_doc).astype(np.int)
        print(f1_score(gt, pred))
    break