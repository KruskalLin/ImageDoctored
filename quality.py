import json

import numpy as np
import pywt
from PIL import Image
import os
from collections import Counter

from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from skimage.metrics import *


aut = 'original_images'
doc = 'images'
config_dir = 'configs'
temp = 'resave_images'

docs = os.listdir(doc)
docs.sort()

for doc_img in docs:
    configs = json.load(open(config_dir + '/' + doc_img.split('.')[0][1:] + '.json', 'r'))

    aut_im = Image.open(aut + '/' + doc_img[1:])
    aut_im.save(temp + '/aut_temp.jpg', 'JPEG', quality=90)
    aut_resaved_im = Image.open(temp + '/aut_temp.jpg')
    aut_ela_im = np.abs(np.array(aut_im, dtype=np.float) - np.array(aut_resaved_im, dtype=np.float))

    doc_im = Image.open(doc + '/' + doc_img)
    doc_im.save(temp + '/doc_temp.jpg', 'JPEG', quality=90)
    doc_resaved_im = Image.open(temp + '/doc_temp.jpg')
    doc_ela_im = np.abs(np.array(doc_im, dtype=np.float) - np.array(doc_resaved_im, dtype=np.float))
    for config in configs:
        point = config['point']
        h = config['h']
        w = config['w']
        ori_y = point[1]
        ori_x = point[0] + 1
        print('ssim')
        print(structural_similarity(aut_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :],
                                    doc_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :], win_size=11,
                                    multichannel=True))

        size_h = 8
        size_w = 8
        mean_aut = np.mean(aut_ela_im[ori_y:ori_y + h, ori_x:ori_x + w, :])
        mean_doc = np.mean(doc_ela_im[ori_y:ori_y + h, ori_x:ori_x + w, :])
        aut_ela_im_window = view_as_windows(aut_ela_im[ori_y:ori_y + h, ori_x:ori_x + w, :],
                                            (size_h, size_w, 3), step=1)
        doc_ela_im_window = view_as_windows(doc_ela_im[ori_y:ori_y + h, ori_x:ori_x + w, :],
                                            (size_h, size_w, 3), step=1)
        aut_ela_im_window = aut_ela_im_window.mean(axis=5).mean(axis=4).mean(axis=3).flatten()
        doc_ela_im_window = doc_ela_im_window.mean(axis=5).mean(axis=4).mean(axis=3).flatten()
        gt = (aut_ela_im_window > mean_aut).astype(np.int)
        pred = (doc_ela_im_window > mean_doc).astype(np.int)
        print('ela')
        f1 = f1_score(gt, pred)
        print(f1)

        size = 5

        diff_image = abs(aut_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :] - doc_ela_im[ori_y:ori_y + h - 2, ori_x:ori_x + w - 2, :]).mean(2)
        diff_windows = view_as_windows(diff_image, (size, size), step=size).reshape((-1, size * size))
        diff_windows_mean = diff_windows.mean(0)
        diff_windows = diff_windows - diff_windows_mean
        pca = PCA()
        pca.fit(diff_windows)
        PC = pca.components_
        diff_windows_feature = view_as_windows(diff_image, (size, size), step=3).reshape((-1, size * size))
        FVS = np.dot(diff_windows_feature, PC)
        FVS = FVS - diff_windows_mean
        kmeans = KMeans(3, verbose=0)
        kmeans.fit(FVS)
        output = kmeans.predict(FVS)
        count = Counter(output)
        least_index = min(count, key=count.get)
        print('pca')
        print((output != least_index).sum() / output.size)

        coeffs2 = pywt.dwt2(doc_ela_im[ori_y:ori_y + h, ori_x:ori_x + w, :], 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        HH_blocks = view_as_windows(HH, (size, size, 4), step=size).reshape((-1, size * size))
        HH_std = HH_blocks.std(axis=1).reshape(-1, 1)
        kmeans = KMeans(2, verbose=0)
        kmeans.fit(HH_std)
        output = kmeans.predict(HH_std)
        count = Counter(output)
        least_index = min(count, key=count.get)
        print('noise')
        print((output != least_index).sum() / output.size)


