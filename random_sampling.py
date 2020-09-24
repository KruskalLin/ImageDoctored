import numpy as np
import os
from PIL import Image

image_boxing_dir = 'area'
image_dir = 'dataset_copymove/images/'
image_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


for clazz in image_classes:
    imgs = os.listdir(image_dir + clazz)
    images = np.random.choice(imgs, size=2, replace=False)
    for image in images:
        Image.open(image_dir + clazz + '/' + image).save('images/' + image, 'JPEG', quality=100, subsampling=0)