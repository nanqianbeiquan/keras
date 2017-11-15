# -*- coding: utf-8 -*-

import datetime
from dateutil.parser import parse

def get_today():
    return str(datetime.date.today())

def get_tomorrow():
    today = datetime.date.today()
    delta = datetime.timedelta(days=1)
    tomorrow = today+delta
    return str(tomorrow)

def get_yesterday():
    today = datetime.date.today()
    delta = datetime.timedelta(days=1)
    yesterday = today-delta
    return str(yesterday)

def paese_date(dt):
    try:
        dt=datetime.datetime.strptime(dt, '%Y年%m月%d日').strftime('%Y-%m-%d')
        
    except:
        pass
    date = parse(dt)
    date = date.strftime('%Y-%m-%d')
    return(date)
    
print(get_today())
print(get_tomorrow())
print(get_yesterday())
print(paese_date('20170812'))
print(paese_date('2017-08-12'))
print(paese_date('2016-09-12 09:00'))
print(paese_date('2015/6/17'))
print(paese_date('Oct 8 2016 12:33:08 PM'))
print(paese_date('Apr 15, 2010 12:00:00 AM'))
print(paese_date('2017年8月1日'))



