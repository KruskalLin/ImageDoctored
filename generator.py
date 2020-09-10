import os
import numpy as np
from PIL import Image
import cv2
import json
import random

image_boxing_dir = 'area'
image_origin_dir = 'class'
image_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def generate_configs():
    areas = os.listdir(image_boxing_dir)
    for area in areas:
        img = np.array(Image.open(image_boxing_dir + '/' + area))
        data = []
        img_area = img.shape[0] * img.shape[1]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([0, 200, 100])
        high_hsv = np.array([2, 255, 255])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rect = cv2.minAreaRect(c)
            box_ = cv2.boxPoints(rect)
            h = abs(box_[3, 1] - box_[1, 1])
            w = abs(box_[3, 0] - box_[1, 0])
            if h * w < img_area * 0.0005 or 1 < abs(rect[2]) < 89:
                continue
            data.append({'point': [int(round(box_.min(0)[0])), int(round(box_.min(0)[1]))], 'h': int(round(h)), 'w': int(round(w))})
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 0, 255), 3)
        Image.fromarray(img).save('bounding_detect/' + area)
        json.dump(data, open('configs/' + area.split('.')[0] + '.json', 'w'))


def generate_removal():
    for c in image_classes:
        imgs = os.listdir(image_origin_dir + '/' + c + '/')
        for img_name in imgs:
            img_num = img_name.split('.')[0][1:]
            img = np.array(Image.open(image_origin_dir + '/' + c + '/' + img_name))
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            configs = json.load(open('configs/' + img_num + '.json', 'r'))
            masks_num = random.randint(1, 5)
            for config in configs:
                for _ in range(masks_num):
                    point = config['point']
                    h = config['h']
                    w = config['w']
                    rand_y = random.randint(0, h - 1)
                    rand_x = random.randint(0, w - 1)
                    ori_y = point[1] + rand_y
                    ori_x = point[0] + rand_x
                    rand_h = random.randint(1, h - rand_y)
                    rand_w = random.randint(1, w - rand_x)
                    img[ori_y:ori_y + rand_h, ori_x:ori_x + rand_w, :] = 0
                    mask[ori_y:ori_y + rand_h, ori_x:ori_x + rand_w] = 0
            if not os.path.exists('dataset_removal'):
                os.makedirs('dataset_removal')
            if not os.path.exists('dataset_removal/images'):
                os.makedirs('dataset_removal/images')
            if not os.path.exists('dataset_removal/masks'):
                os.makedirs('dataset_removal/masks')
            if not os.path.exists('dataset_removal/images/' + c):
                os.makedirs('dataset_removal/images/' + c)
            if not os.path.exists('dataset_removal/masks/' + c):
                os.makedirs('dataset_removal/masks/' + c)
            Image.fromarray(img).save('dataset_removal/images/' + c + '/' + img_name)
            Image.fromarray(mask).save('dataset_removal/masks/' + c + '/' + img_name)


def generate_copymove():
    for c in image_classes:
        imgs = os.listdir(image_origin_dir + '/' + c + '/')
        for img_name in imgs:
            img_num = img_name.split('.')[0][1:]
            img = np.array(Image.open(image_origin_dir + '/' + c + '/' + img_name))
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            configs = json.load(open('configs/' + img_num + '.json', 'r'))
            masks_num = random.randint(1, 5)
            for config in configs:
                for _ in range(masks_num):
                    point = config['point']
                    h = config['h']
                    w = config['w']
                    rand_y = random.randint(0, h - 1)
                    rand_x = random.randint(0, w - 1)
                    ori_y = point[1] + rand_y
                    ori_x = point[0] + rand_x
                    rand_h = random.randint(1, h - rand_y)
                    rand_w = random.randint(1, w - rand_x)
                    copy_y = random.randint(0, img.shape[0] - rand_h)
                    copy_x = random.randint(0, img.shape[1] - rand_w)
                    img[ori_y:ori_y + rand_h, ori_x:ori_x + rand_w, :] = img[copy_y:copy_y + rand_h, copy_x:copy_x + rand_w, :]
                    mask[ori_y:ori_y + rand_h, ori_x:ori_x + rand_w] = 0
            if not os.path.exists('dataset_copymove'):
                os.makedirs('dataset_copymove')
            if not os.path.exists('dataset_copymove/images'):
                os.makedirs('dataset_copymove/images')
            if not os.path.exists('dataset_copymove/masks'):
                os.makedirs('dataset_copymove/masks')
            if not os.path.exists('dataset_copymove/images/' + c):
                os.makedirs('dataset_copymove/images/' + c)
            if not os.path.exists('dataset_copymove/masks/' + c):
                os.makedirs('dataset_copymove/masks/' + c)
            Image.fromarray(img).save('dataset_copymove/images/' + c + '/' + img_name)
            Image.fromarray(mask).save('dataset_copymove/masks/' + c + '/' + img_name)


def generate_slicing():
    for c in image_classes:
        imgs = os.listdir(image_origin_dir + '/' + c + '/')
        for i, img_name in enumerate(imgs):
            img_num = img_name.split('.')[0][1:]
            img = np.array(Image.open(image_origin_dir + '/' + c + '/' + img_name))
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            configs = json.load(open('configs/' + img_num + '.json', 'r'))
            masks_num = random.randint(1, 5)
            rand = 0
            while rand == i:
                rand = random.randint(0, len(imgs) - 1)
            img_ano = np.array(Image.open(image_origin_dir + '/' + c + '/' + imgs[rand]))
            for config in configs:
                for _ in range(masks_num):
                    point = config['point']
                    h = config['h']
                    w = config['w']
                    rand_y = random.randint(0, h - 1)
                    rand_x = random.randint(0, w - 1)
                    ori_y = point[1] + rand_y
                    ori_x = point[0] + rand_x
                    rand_h = random.randint(1, h - rand_y)
                    rand_w = random.randint(1, w - rand_x)
                    if img_ano.shape[0] - rand_h < 0 or img_ano.shape[1] - rand_w < 0:
                        continue
                    copy_y = random.randint(0, img_ano.shape[0] - rand_h)
                    copy_x = random.randint(0, img_ano.shape[1] - rand_w)
                    img[ori_y:ori_y + rand_h, ori_x:ori_x + rand_w, :] = img_ano[copy_y:copy_y + rand_h, copy_x:copy_x + rand_w, :]
                    mask[ori_y:ori_y + rand_h, ori_x:ori_x + rand_w] = 0
            if not os.path.exists('dataset_slicing'):
                os.makedirs('dataset_slicing')
            if not os.path.exists('dataset_slicing/images'):
                os.makedirs('dataset_slicing/images')
            if not os.path.exists('dataset_slicing/masks'):
                os.makedirs('dataset_slicing/masks')
            if not os.path.exists('dataset_slicing/images/' + c):
                os.makedirs('dataset_slicing/images/' + c)
            if not os.path.exists('dataset_slicing/masks/' + c):
                os.makedirs('dataset_slicing/masks/' + c)
            Image.fromarray(img).save('dataset_slicing/images/' + c + '/' + img_name)
            Image.fromarray(mask).save('dataset_slicing/masks/' + c + '/' + img_name)

generate_slicing()