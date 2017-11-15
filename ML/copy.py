# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:14:09 2017

@author: jingwang.tian
"""

import matplotlib.pyplot as plt
import numpy as np

gold,chihh = 400,400
gold_height = 40*10*np.random.randn(gold)
chihh_height = 25*6*np.random.randn(chihh)

plt.hist([gold_height,chihh_height],stacked = True,color = ['r','b'])
plt.show()