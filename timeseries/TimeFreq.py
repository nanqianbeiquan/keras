# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pandas import Series,DataFrame
from datetime import datetime
from dateutil.parser import parse
import time
from pandas.tseries.offsets import Hour,Minute,Day,MonthEnd
import pytz

#下面的'A-DEC'是年第12月底最后一个日历日
p = pd.Period('2016',freq = 'A-DEC')
#Period可以直接加减
print (p + 5)
#相同频率的Period可以进行加减,不同频率是不能加减的
rng = pd.Period('2015',freq = 'A-DEC') - p
print (rng)
rng = pd.period_range('1/1/2000','6/30/2000',freq = 'M')
#类型是<class 'pandas.tseries.period.PeriodIndex'>，形式上是一个array数组
#注意下面的形式已经不是书上的形式，而是float类型，但是做索引时，还是日期形式
print (rng)
print (type(rng))
print (Series(np.random.randn(6),index = rng),'\n')
#PeriodIndex类的构造函数还允许直接使用一组字符串
values = ['2001Q3','2002Q2','2003Q1']
index = pd.PeriodIndex(values,freq = 'Q-DEC')
#下面index的
print (index)

print (p.asfreq('M',how = 'start'))
print (p.asfreq('M',how = 'end'))

#高频率转换为低频率时，超时期是由子时期所属位置决定的,例如在A-JUN频率中，月份“2007年8月”实际上属于“2008年”
p = pd.Period('2007-08','M')
print (p.asfreq('A-JUN'),'\n')

#PeriodIndex或TimeSeries的频率转换方式也是如此：
rng = pd.period_range('2006','2009',freq = 'A-DEC')
ts = Series(np.random.randn(len(rng)),index = rng)
print (ts)
print (ts.asfreq('M',how = 'start'))
print (ts.asfreq('B',how = 'end'),'\n')

print('-'*78)
p = pd.Period('2012Q4',freq = 'Q-JAN')
print (p)
#在以1月结束的财年中，2012Q4是从11月到1月
print (p.asfreq('D','start'))
print (p.asfreq('D','end'),'\n')
#因此，Period之间的运算会非常简单，例如，要获取该季度倒数第二个工作日下午4点的时间戳
p4pm = (p.asfreq('B','e') - 1).asfreq('T','s') + 16 * 60
print (p4pm)
print (p4pm.to_timestamp(),'\n')

#period_range还可以用于生产季度型范围，季度型范围的算数运算也跟上面是一样的：
#要非常小心的是Q-JAN是什么意思
rng = pd.period_range('2011Q3','2012Q4',freq = 'Q-JAN')
print (rng.to_timestamp())
ts = Series(np.arange(len(rng)),index = rng)
print (ts,'\n')
new_rng = (rng.asfreq('B','e') - 1).asfreq('T','s') + 16 * 60
ts.index = new_rng.to_timestamp()
print (ts,'\n')
print('-'*78)
#下面生成1分钟线
rng = pd.date_range('1/1/2000',periods = 12,freq = 'T')
ts = Series(range(0,12),index = rng)
print (ts,'\n')
#下面聚合到5min线
print (ts.resample('5min',how = 'sum'))
#传入的频率将会以“5min”的增量定义面元。默认情况下，面元的有边界是包含右边届的，即00:00到00:05是包含00:05的
#传入closed = 'left'会让左边的区间闭合
print (ts.resample('5min',how = 'sum',closed = 'left'))
#最终的时间序列默认是用右侧的边界标记，但是传入label = 'left'可以转换为左边标记
print (ts.resample('5min',how = 'sum',closed = 'left',label = 'left'),'\n')
#最后，你可能需要对结果索引做一些位移，比如将右边界减去一秒更容易明白到底是属于哪一个区间
#通过loffset设置一个字符串或者日期偏移量即可实现此目的,书上作者没有加left是矛盾的，当然也可以调用shift来进行时间偏移
print (ts.resample('5min',how = 'sum',closed = 'left',loffset = '-1s'))
print('-'*78)
rng = pd.date_range('1/1/2000',periods = 100,freq = 'D')
ts = Series(np.arange(100),index = rng)
print (ts.groupby(lambda x:x.month).mean())  #作真是越写越省事了……
print (ts.groupby(lambda x:x.weekday).mean())