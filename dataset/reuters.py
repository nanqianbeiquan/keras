# -*- coding: utf-8 -*-

import numpy as np


imfile = 'E:\pest1\\retuers\\reuters.npz'
npzfile = np.load(imfile)
print(npzfile.files)
print(npzfile['y'])
print(npzfile['x'])
