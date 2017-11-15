# -*- coding: utf-8 -*-


from datetime import datetime
import datetime as dt
from dateutil.parser import parse
import pandas as pd

now = datetime.now()

#datetime以毫秒形式储存时间

print (now,now.year,now.month,now.day,now.microsecond,'\n')
#print datetime(2015,12,17,20,00,01,555555) #设置一个时间

#datetime.timedelta表示两个datetime对象之间的时间差
#换句话说，datetime格式可以相相减
delta = datetime(2017,9,6) - datetime(2009,5,16,8,15)
print (delta)
#把注意下面是days And seconds
print (dt.timedelta(3034,56700))
print (delta.days)
print (delta.seconds)

start = datetime(2009,5,16)
#参数分别为days,seconds,microseconds(微秒),milliseconds（毫秒）,minutes,hours,weeks
#,除了微秒小数自动四舍五入之外，其他的都能自动转换为其他度量
print (start + dt.timedelta(1,20,0.5,5,10,10,0))

stamp = datetime(2011,1,3)
print(stamp)
print (stamp.strftime('%Y-%m-%d'))
print (stamp.strftime('%Y-%m-%d'),'\n')
value = '2011-01-03'
print (datetime.strptime(value,'%Y-%m-%d'))
#注意这是datetime函数的函数，不是模块的函数
datestrs = ['7/6/2011','8/6/2011']
print ([datetime.strptime(x,'%m/%d/%Y') for x in datestrs])
#上面将字符串转化为最常用的格式，但是每次都自己写出来有点麻烦，
#可以用dateutil这个第三方包中的parser.parse方法

print (parse('20110103'))
print (parse('2011/01/03'))
print (parse('Jan 31 1997 10:45 PM'))
#国际通用格式中，日出现在月的前面，传入dayfirst = True即可
print (parse('6/12/2011',dayfirst = True),'\n')
#pandas通常是用于处理成组日期的，不管这些日期是DataFrame的行还是列。
print (pd.to_datetime(datestrs),'\n')
idx = pd.to_datetime(datestrs + [None])
print (idx)
print (idx[2]) #这里应该是NaT（Not a Time）
print (pd.isnull(idx))
#parse是一个不完美的工具，比如下面
print (parse('42'))