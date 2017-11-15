# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pandas import Series,DataFrame
from datetime import datetime
from dateutil.parser import parse
import numpy as np
from pandas.tseries.offsets import Hour,Minute,Day,MonthEnd
import pytz

dates = [datetime(2011,1,2),datetime(2011,1,5),datetime(2011,1,7),
datetime(2011,1,8),datetime(2011,1,10),datetime(2011,1,12)]
print (dates)
ts = Series(np.random.randn(6),index = dates)
print (ts,'\n')
#这些datetime对象实际上是被放在一个DatetimeIndex中的。现在，变量ts就成为了TimeSeries了。
print (type(ts))
print (ts.index,'\n')
#没必要显示使用TimeSeries的构造函数。当创建一个带有DatetimeIndex的Series时，pandas就会知道该对象是一个时间序列
print (ts + ts[::2])
#pandas用NumPy的datetime64数据类型以纳秒形式存储时间戳：
print (ts.index.dtype)
#DatetimeIndex中的各个标量值是pandas的Timestamp
stamp = ts.index[0]
print (stamp)

#只要有需要，TimeStamp可以随时自动转换为datetime对象。此外，还可以存储频率信息，且知道如何执行时区转换以及其他操作


#pandas.date_range会生成指定长度的DatetimeIndex
index = pd.date_range('4/1/2015','6/1/2015')
print(index)
#默认情况下，date_range产生按天计算的时间点，当然可以传入开始或结束日期，还得传入一个表示一段时间的数字
print (pd.date_range('1/1/2016',periods = 31),'\n')
#开始和结束定义了日期索引的严格边界，如果你想要生成一个由每月最后一个工作日组成的日期索引，可以传入‘BM’（business end of month）
#这样就只会包含时间间隔内（或者放好在时间边界上）符合频率要求的日期：
print (pd.date_range('12/18/2015','1/1/2016',freq = 'BM'),'\n')
#date_range默认保留起始和结束时间戳信息
print (pd.date_range('5/2/2015 12:12:12',periods = 5))
#有时，虽然起始和结束带有时间信息，但是可以用normalize = True把它们吧变为00:00:00
print (pd.date_range('5/2/2015 12:12:12',periods = 5,normalize = True))

print('-'*50)
#pandas中的频率是由一个基础频率和一个乘数组成的。基础的频率由字符串表示，比如‘M’表示月，‘H’表示小时
#对于每个基础频率，都有一个被称为日期偏移量（date offset）的对象与之对应。
hour = Hour()
print (hour) #感觉这个形式比较霸气
#传入整数可以自定义偏移量倍数
four_hours = Hour(4)
print (four_hours)
#一般而言，并不需要显示创建偏移量，只需创建时间序列时传入'H'或者'4h'即可
print (pd.date_range('1/1/2016','1/2/2016',freq = '4h'),'\n')
#偏移量可以拼接
print (Hour(1) + Minute(30))
#传入频率字符串（'2h30min'）,这种字符串可以被高效地解析为等效的表达式
print (pd.date_range('1/1/2016',periods = 10,freq = '1h30min'),'\n')
#有些频率所描述的时间点并不是均匀分隔的。例如'M'和'BM'就取决于每月的天数，对于后者，还要考虑月末是不是周末，将这些成为锚点偏移量（anchored offset）
#WOM(Week Of Month)日期是一个非常常用的频率，以WOM开头，能产生诸如“每月第三个星期五”之类的信息
rng = pd.date_range('1/1/2016','9/1/2016',freq = 'WOM-3FRI')
print (rng)

print('-'*50)
ts = Series(np.random.randn(4),index = pd.date_range('1/1/2016',periods = 4,freq = 'M'))
print(ts)
print (ts.shift(2))
print (ts.shift(-2),'\n')
#可以看到，shift通常用于计算一个时间序列或多个时间序列（如DataFrame列）中的百分比变化。
print (ts / ts.shift(1) - 1)
#单纯的移位操作不会修改索引，所以部分数据会被丢弃，如果频率已知，则可以将其传给shift以实现对时间戳进行位移而不是只对数据移位
print (ts.shift(2,freq = 'M'))  #时间戳移动，而数据不动
print (ts.shift(3,freq = 'D'),'\n')
print (ts.shift(1,freq = '3D'))
print (ts.shift(1,freq = '90T'))
print('-'*50)
now = datetime(2011,11,29)
print (now + Day(3),'\n')
#如果加的是锚点偏移量，第一次增量会将原日期向前滚动到符合频率规则的下一个日期
#如果本来就是锚点，那么下一个就是下一个锚点
print (now + MonthEnd(),'\n')
print (now + MonthEnd(2),'\n')
#通过锚点偏移量的rollforward和rollback方法，可显示地将日期向前或向后“滚动”
offset = MonthEnd()
print (offset.rollforward(now),'\n')
print (offset.rollback(now),'\n')
#日期偏移量还有一个巧妙的用法，即结合groupby使用这两个“滚动”方法
ts = Series(np.random.randn(20),index = pd.date_range('1/15/2000',periods = 20,freq = '4M'))

print (ts,'\n')
#注意下面的方式，很隐晦
print (ts.groupby(offset.rollforward).mean(),'\n')
#当然，更简单快速的方式是使用resample
print (ts.resample('M',how = 'mean'))
print('-'*50)
print (pytz.common_timezones[-5:])
tz = pytz.timezone('US/Eastern')

rng = pd.date_range('3/9/2012 9:30',periods = 6,freq = 'D')
ts = Series(np.random.randn(len(rng)),index = rng)
print ('ts',ts,'\n')
print (ts.index.tz,'\n')  #默认的时区字段为None
#在生成日期范围的时候还可以加上一个时区集
print (pd.date_range('3/9/2012',periods = 10,freq = 'D',tz = 'UTC'),'\n')
#从单纯到本地化的转换是通过tz_localize方法处理的：
ts_utc = ts.tz_localize('US/Pacific')  #转换为美国太平洋时间
print (ts_utc,'\n')
print (ts_utc.index,'\n')
#一旦被转换为某个特定时期，就可以用tz_convert将其转换到其他时区了
print (ts_utc.tz_convert('US/Eastern'),'\n')
#tz_localize和tz_convert是DatetimeIndex的实例方法，可以把一个DatetimeIndex转化为特定时区
print (ts.index.tz_localize('Asia/Shanghai'))

print('-'*50)
stamp = pd.Timestamp('2011-03-12 04:00')
print (type(stamp),'\n')
stamp_utc = stamp.tz_localize('UTC')
print (stamp_utc,'\n')
print (stamp_utc.tz_convert('US/Eastern'),'\n')
stamp_moscow = pd.Timestamp('2011-03-12 04:00',tz = 'Europe/Moscow')
print (stamp_moscow)
#时区意识型Timestamp对象在内部保存了一个UTC时间戳值（自1970年1月1日起的纳秒数），这个UTC值在时区转换过程中是不会变化的
print (stamp_utc.value)
print (stamp_utc.tz_convert('US/Eastern').value,'\n')
#当使用pandas的DataOffset对象执行运算时，会自动关注“夏时令”…………

rng = pd.date_range('3/7/2012',periods = 10,freq = 'B')
ts = Series(np.random.randn(len(rng)),index = rng)
print (ts)
ts1 = ts[:7].tz_localize('Europe/London')
#注意naive是不能直接转换为时区的，必须先转换为localize再进行转换
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
#转换为UTC
print (result.index)