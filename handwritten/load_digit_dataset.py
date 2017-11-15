# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

IMAGE_SIZE = 28

def resize_image(image, height = IMAGE_SIZE,width = IMAGE_SIZE):
    top, bottom, left, right=(0,0,0,0)
    h,w = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge -w
        left = dw // 2
        right = dw - left
    else:
        pass
    
    BLACK = [0,0,0]
    constant = cv2.copyMakeBorder(image,top,bottom,left,right,
                                  cv2.BORDER_CONSTANT,value = BLACK)
    return cv2.resize(constant,(height,width))
    
images = []
labels = []

def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name,dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('png'):
                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                image = resize_image(image,IMAGE_SIZE,IMAGE_SIZE)
                cv2.imwrite('1.png',image)
                images.append(image)
                labels.append(int(path_name.split('\\')[3]))
    return images,labels
        
def load_dataset(path_name):
    images,labels = read_path(path_name)
    images = np.array(images)
    labels = np.array(labels)
    return images,labels

    
if __name__ == '__main__':
    path_name = 'E:\pest1\mnist'
    load_dataset(path_name)
        

    
