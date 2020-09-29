import json
import os
from PIL import Image
import numpy as np


def generate_patches(src, dst, conf):
    auts = os.listdir(src + '/authentic')
    for aut in auts:
        aut_img = np.array(Image.open(src + '/authentic/' + aut))
        doc_img = np.array(Image.open(src + '/doctored/' + aut))
        index = aut.split('.')[0][1:]
        configs = json.load(open(conf + '/' + index + '.json', 'r'))
        for i, config in enumerate(configs):
            point = config['point']
            h = config['h']
            w = config['w']
            ori_y = point[1]
            ori_x = point[0]
            ori_y_32 = ori_y // 32 * 32
            ori_x_32 = ori_x // 32 * 32
            right_32 = ((ori_y + h) // 32 + 1) * 32
            bottom_32 = ((ori_x + w) // 32 + 1) * 32
            if right_32 < aut_img.shape[0] and bottom_32 < aut_img.shape[1]:
                Image.fromarray(aut_img[ori_y_32:right_32, ori_x_32:bottom_32, :])\
                    .save(dst + '/authentic/' + index + '_' + str(i) + '.jpg', 'JPEG', quality=100, subsampling=0)
                Image.fromarray(doc_img[ori_y_32:right_32, ori_x_32:bottom_32, :]) \
                    .save(dst + '/doctored/' + index + '_' + str(i) + '.jpg', 'JPEG', quality=100, subsampling=0)


def repaint(src, dst, conf):
    docs = os.listdir(src + '/doctored')
    for doc in docs:
        doc_img = np.array(Image.open(src + '/doctored/' + doc))
        index = doc.split('.')[0][1:]
        configs = json.load(open(conf + '/' + index + '.json', 'r'))
        for i, config in enumerate(configs):
            patch = np.array(Image.open(dst + '/attack/' + index + '_' + str(i) + '.jpg'))
            point = config['point']
            h = config['h']
            w = config['w']
            ori_y = point[1]
            ori_x = point[0]
            ori_y_32 = ori_y // 32 * 32
            ori_x_32 = ori_x // 32 * 32
            right_32 = ((ori_y + h) // 32 + 1) * 32
            bottom_32 = ((ori_x + w) // 32 + 1) * 32
            if right_32 < doc_img.shape[0] and bottom_32 < doc_img.shape[1]:
                doc_img[ori_y_32:right_32, ori_x_32:bottom_32, :] = patch
        Image.fromarray(doc_img).save(src + '/attack/' + doc, 'JPEG', quality=100, subsampling=0)




generate_patches('testing', 'patches', 'configs')
# repaint('testing', 'patches', 'configs')
