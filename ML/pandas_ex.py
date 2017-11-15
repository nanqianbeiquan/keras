# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime

now = datetime.now()
#datetime以毫秒形式储存时间
print (now,now.year,now.month,now.day,now.microsecond,'\n')
#print datetime(2015,12,17,20,00,01,555555) #设置一个时间
#datetime.timedelta表示两个datetime对象之间的时间差
#换句话说，datetime格式可以相相减
delta = datetime(2017,9,6) - datetime(2009,5,16)
print (delta)
#把注意下面是days And seconds
print (dt.timedelta(3400,912415))
print (delta.days)
print (delta.seconds)
#下面是错误的
#print delta.minutes
start = datetime(2009,5,16)
#参数分别为days,seconds,microseconds(微秒),milliseconds（毫秒）,minutes,hours,weeks,除了微秒小数自动四舍五入之外，其他的都能自动转换为其他度量
print (start + dt.timedelta(1,20,0.5,5,10,10,0))
print (start + dt.timedelta(3400))

print (datetime(2017,9,6) + dt.timedelta(265))


