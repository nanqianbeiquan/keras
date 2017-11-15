# coding=utf-8

import numpy as np
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 64

# 按指定大小调整图片
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
	top,bottom,left,right=(0, 0, 0, 0)

	# 获取图像尺寸
	h, w, _ = image.shape

	# 对于图片长和宽不一致的一边，找到最长的一边
	longest_edge = max(h,w)

	#计算短边需要增加多少像素才可以与长边一致
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
	BLACK = [0, 0, 0]

	# 给图片增加边界，使图片的长宽等长，cv2.BORDER_CONSTANT指定边界颜色由value决定
	constant = cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value = BLACK)

	# 调整图片大小并返回
	return cv2.resize(constant,(height,width))

# 读取训练数据
images = []
labels = []

def read_path(path_name):
	for dir_item in os.listdir(path_name):
		# print path_name
		# print dir_item
		# index, label in enumerate(labels)
		# 从初始路径开始叠加，合成可以识别的操作路径
		full_path = os.path.abspath(os.path.join(path_name, dir_item))
		if os.path.isdir(full_path): # 如果是文件夹，继续递归调用
			read_path(full_path)
		else: # 文件
			if dir_item.endswith('.jpg'):
				image = cv2.imread(full_path)
				image = resize_image(image,IMAGE_SIZE,IMAGE_SIZE)
				# 放开这个代码，可以看到resize_image的实际调用
				cv2.imwrite('1.jpg',image)
				images.append(image)
				labels.append(int(path_name.split('\\')[3]))
#	print ('images:',images)
	# print 'labels:',labels
	return images,labels

# 从指定路径读取训练数据
def load_dataset(path_name):
	images,labels = read_path(path_name)
	# 将训练数据分成四维数组，尺寸为（图片数量*IMAGE_SIZE*IMAGE_SIEZ*3）
	#  图片为64*64像素，一个像素有3个颜色值(RGB)
	images = np.array(images)
	print (images.shape)
	# print labels
	# 标注数据
	labels = np.array(labels)
	# print 'images:',images
	# print 'labels:',labels
	return images,labels

if __name__ == '__main__':
	path_name = 'E:\pest1\dogs'
	load_dataset(path_name)


