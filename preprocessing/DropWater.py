# -*- coding: utf-8 -*-

from itertools import groupby
from PIL import Image
import cv2
import numpy as np
import time
import random

COLOR_RGB_BLACK = (0, 0, 0)
COLOR_RGB_WHITE = (255, 255, 255)
COLOR_RGBA_BLACK = (0, 0, 0, 255)
COLOR_RGBA_WHITE = (255, 255, 255, 255)

BORDER_LEFT = 0
BORDER_TOP = 1
BORDER_RIGHT = 2
BORDER_BOTTOM = 3

RAW_DATA_DIR = 'captcha/'
PROCESSED_DATA_DIR = 'processed/'
LABELS_DIR = 'labels/'

NORM_SIZE = 20

def binarizing(img,threshold):
    """传入image对象进行灰度、二值处理"""
    img = img.convert("L") # 转灰度
    pixdata = img.load()
    w, h = img.size
    # 遍历所有像素，大于阈值的为黑色
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img

def vertical(img):
    """传入二值化后的图片进行垂直投影"""
    pixdata = img.load()
    w,h = img.size
    result = []
    for x in range(w):
        black = 0
        for y in range(h):
            if pixdata[x,y] == 0:
                black += 1
        result.append(black)
    return result

def get_start_x(hist_width):
    """根据图片垂直投影的结果来确定起点
 hist_width中间值 前后取4个值 再这范围内取最小值
 """
    mid = len(hist_width) // 2 # 注意py3 除法和py2不同
    temp = hist_width[mid-4:mid+5]
    return mid - 5 + temp.index(min(temp))

def get_nearby_pix_value(img_pix,x,y,j):
    """获取临近5个点像素数据"""
    if j == 1:
        return 0 if img_pix[x-1,y+1] == 0 else 1
    elif j ==2:
        return 0 if img_pix[x,y+1] == 0 else 1
    elif j ==3:
        return 0 if img_pix[x+1,y+1] == 0 else 1
    elif j ==4:
        return 0 if img_pix[x+1,y] == 0 else 1
    elif j ==5:
        return 0 if img_pix[x-1,y] == 0 else 1
    else:
        raise Exception("get_nearby_pix_value error")


def get_end_route(img,start_x,height):
    """获取滴水路径"""
    left_limit = 0
    right_limit = img.size[0] - 1
    end_route = []
    cur_p = (start_x,0)
    last_p = cur_p
    end_route.append(cur_p)

    while cur_p[1] < (height-1):
        sum_n = 0
        max_w = 0
        next_x = cur_p[0]
        next_y = cur_p[1]
        pix_img = img.load()
        for i in range(1,6):
            cur_w = get_nearby_pix_value(pix_img,cur_p[0],cur_p[1],i) * (6-i)
            sum_n += cur_w
            if max_w < cur_w:
                max_w = cur_w
        if sum_n == 0:
            # 如果全黑则看惯性
            max_w = 4
        if sum_n == 15:
            max_w = 6

        if max_w == 1:
            next_x = cur_p[0] - 1
            next_y = cur_p[1]
        elif max_w == 2:
            next_x = cur_p[0] + 1
            next_y = cur_p[1]
        elif max_w == 3:
            next_x = cur_p[0] + 1
            next_y = cur_p[1] + 1
        elif max_w == 5:
            next_x = cur_p[0] - 1
            next_y = cur_p[1] + 1
        elif max_w == 6:
            next_x = cur_p[0]
            next_y = cur_p[1] + 1
        elif max_w == 4:
            if next_x > cur_p[0]:
                # 向右
                next_x = cur_p[0] + 1
                next_y = cur_p[1] + 1
            if next_x < cur_p[0]:
                next_x = cur_p[0]
                next_y = cur_p[1] + 1
            if sum_n == 0:
                next_x = cur_p[0]
                next_y = cur_p[1] + 1
        else:
            raise Exception("get end route error")

        if last_p[0] == next_x and last_p[1] == next_y:
            if next_x < cur_p[0]:
                max_w = 5
                next_x = cur_p[0] + 1
                next_y = cur_p[1] + 1
            else:
                max_w = 3
                next_x = cur_p[0] - 1
                next_y = cur_p[1] + 1
        last_p = cur_p

        if next_x > right_limit:
            next_x = right_limit
            next_y = cur_p[1] + 1
        if next_x < left_limit:
            next_x = left_limit
            next_y = cur_p[1] + 1
        cur_p = (next_x,next_y)
        end_route.append(cur_p)
    return end_route

def get_projection_x(img):
    p_x = [0 for _ in range(img.size[0])]
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if img.getpixel((x, y)) == 0:
                p_x[x] = 1
                break
    return p_x

def is_joint(split_len):
    """
		以字符宽度统计值判断当前split_len是否是两个字符的长度
		返回True需要进一步进行滴水算法分割
		"""
    return True if split_len >= 30 else False
    
def get_split_seq(projection_x):
    split_seq = []
    start_x = 0
    length = 0
    for pos_x, val in enumerate(projection_x):
        if val == 0 and length == 0:
            continue
        elif val == 0 and length != 0:
            split_seq.append([start_x, length])
            length = 0
        elif val == 1:
            if length == 0:
                start_x = pos_x
            length += 1
        else:
            raise Exception('generating split sequence occurs error')
    # 循环结束时如果length不为0，说明还有一部分需要append
    if length != 0:
        split_seq.append([start_x, length])
    return split_seq


def do_split(source_image, starts, filter_ends):
    """
 具体实行切割
 : param starts: 每一行的起始点 tuple of list
 : param ends: 每一行的终止点
 """
    cnt = len(filter_ends)
    left = starts[0][0]
    top = starts[0][1]
    right = filter_ends[0][0]
    bottom = filter_ends[cnt-1][1]
    pixdata = source_image.load()
    for i in range(len(starts)):
        left = min(starts[i][0], left)
        top = min(starts[i][1], top)
        right = max(filter_ends[i][0], right)
        bottom = max(filter_ends[i][1], bottom)
    width = right - left + 1
    height = bottom - top + 1
    image = Image.new('RGB', (width, height), (255,255,255))
    for i in range(height):
        start = starts[i]
        end = filter_ends[i]
        for x in range(start[0], end[0]+1):
            if pixdata[x,start[1]] == 0 or pixdata[x,start[1]] == (0,0,0):
                image.putpixel((x - left, start[1] - top), (0,0,0))
    return image

def cut_image(img):
    """
    切割图片为单个字符
    """
    projection_x = get_projection_x(img)
    split_seq = get_split_seq(projection_x)
    croped_images = []
    height = img.size[1]
    for start_x, width in split_seq:
        # 同时去掉y轴上下多余的空白
#        begin_row = 0
#        end_row = height - 1
        for row in range(height):
            flag = True
            for col in range(start_x, start_x + width):
                if img.getpixel((col, row)) == 0:
                    flag = False
                    break
            if not flag: # 如果在当前行找到了黑色像素点，就是起始行
#                begin_row = row
                break
        for row in reversed(range(height)):
            flag = True
            for col in range(start_x, start_x + width):
                if img.getpixel((col, row)) == 0:
                    flag = False
                    break
            if not flag:
#                end_row = row
                break
        croped_images.append(img.crop((start_x, 0, start_x + width, 50)))
#        croped_images.append(img.crop((start_x, begin_row, start_x + width, end_row + 1)))
	 
    # 没考虑一个source image出现多个粘连图片的情况
    need_drop_fall = False
    for idx, split_info in enumerate(split_seq):
        # split_info: (start_x, length)
        print('idx, split_info',idx, split_info)       
        imgx=img.crop((split_info[0], 0, split_info[0]+split_info[1], 50))
        print('%d' % idx+'.png')       
        if is_joint(split_info[1]):
            need_drop_fall = True
            print ("找到一张粘连图片: %d" % idx)
            split_images = drop_fall(croped_images[idx])
#            break
        else:
           imgx.save('%d' % idx+'.png') 
    if need_drop_fall:
        del croped_images[idx]
        croped_images.insert(idx, split_images[0])
        croped_images.insert(idx + 1, split_images[1])

    return croped_images

def drop_fall(img):
    """滴水分割"""
    width,height = img.size
    # 1 二值化
    b_img = binarizing(img,180)
    # 2 垂直投影
    hist_width = vertical(b_img)
    # 3 获取起点
    start_x = get_start_x(hist_width)

    # 4 开始滴水算法
    start_route = []
    for y in range(height):
        start_route.append((0,y))
    end_route = get_end_route(img,start_x,height)
    filter_end_route = [max(list(k)) for _,k in groupby(end_route,lambda x:x[1])] # 注意这里groupby
    img1 = do_split(img,start_route,filter_end_route)
    if img1.size[0] > 30:
        drop_fall(img1)
    else:      
        img1.save('%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png')
    start_route = list(map(lambda x : (x[0]+1,x[1]),filter_end_route)) # python3中map不返回list需要自己转换
    end_route = []
    for y in range(height):
        end_route.append((width-1,y))
    img2 = do_split(img,start_route,end_route)    
    if img2.size[0] > 30:
        drop_fall(img2)
    else:
        img2.save('%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png')

if __name__ == '__main__':
    p = Image.open("118.png")
    cut_image(p)
#    drop_fall(p)