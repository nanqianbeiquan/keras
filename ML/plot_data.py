# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.Series(np.random.randn(1000),index = np.arange(1000))

cdata = data.cumsum()

cdata.plot()

plt.show()

data = pd.DataFrame(
        np.random.randn(1000,4),
        index = np.arange(1000),
        columns=list("ABCD"))
cdata = data.cumsum()

cdata.plot()
plt.show()

ax = cdata.plot.scatter(x='A',y='B',color='DarkGreen',label='Class1')

cdata.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)

