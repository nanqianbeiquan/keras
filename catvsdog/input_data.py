# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

IMAGE_SIZE = 64

# 读取训练数据
images = []
labels = []

def resize_image(image,height = IMAGE_SIZE, width = IMAGE_SIZE):
    top,bottom,left,right = (0,0,0,0)
    
    # 获取图像尺寸
    h,w,_ = image.shape
    # 长短不一致的图片取长边
    longest_edge = max(h,w)
    # 计算短边需要增加量加上像素宽度使与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    
    # RGB颜色
    BLACK = [0,0,0]
    # 给图片增加边界，使长宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    #调整图像大小并返回
    return cv2.resize(constant, (height, width))

# 读取数据
images = []
labels = []
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name,dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)              # 如果是文件夹继续递归调用
        else:
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image,IMAGE_SIZE,IMAGE_SIZE)
                # 放开这个代码，可以看到resize_image()函数的实际调用效果
                cv2.imwrite('1.jpg',image)
                images.append(image)
                labels.append(path_name)
    return images,labels

# 从指定路径读取训练数据
def load_dataset(path_name):
    images,labels = read_path(path_name)
    #将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
    #我和闺女两个人共1200张图片，IMAGE_SIZE为64，故对我来说尺寸为1200 * 64 * 64 * 3
    #图片为64 * 64像素,一个像素3个颜色值(RGB)
    images = np.array(images)
    labels = np.array([0 if label.endswith('0') else 1 for label in labels])
    return images,labels

if __name__ == '__main__':
    path_name='E:\pest1\catanddog\catvsdog'
    images,labels=load_dataset(path_name)
    print(images)
    for i in range(10):
        index = random.randint(0, 10)
        plt.subplot(2, 5, i+1)
        plt.title(images[index])
        plt.imshow(labels[index])
        plt.axis('off')