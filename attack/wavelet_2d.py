#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pywt import WaveletPacket2D


arr = np.array(Image.open('test.jpg').convert('L'))
wp2 = WaveletPacket2D(arr, 'db2', 'symmetric', maxlevel=2)
datas = []
for row in wp2.get_level(2, 'freq'):
    for node in row:
        datas.append(node.data)

init = np.sqrt(np.abs(datas[-1]))
for i in range(15, int(len(datas) - 1)):
    init += np.sqrt(np.abs(datas[i]))
print(init.max())
Image.fromarray(init).save('init.tiff')