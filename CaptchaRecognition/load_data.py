# coding=utf-8
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random

def load_dataset():
	dataset = []
	labelset = []
	label_map = {}
	base_dir = "E:/pest1/training_set/"
	labels = os.listdir(base_dir)
	for index, label in enumerate(labels):
		image_files = os.listdir(base_dir + label)
		for image_file in image_files:
			image_path = base_dir + label + "/" + image_file
			im = Image.open(image_path).convert('L').resize((28, 28))
			# im.show()
			dataset.append(np.asarray(im, dtype=np.float))
			labelset.append(index)
		label_map[index] = label
	return np.array(dataset), np.array(labelset), label_map

# dataset, labelset, label_map = load_dataset()

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation, :, :]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

# dataset, labelset = randomize(dataset, labelset)

def reformat(dataset, labels, image_size, num_labels):
	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
	return dataset, labels

# train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labelset)

# train_dataset, train_labels = reformat(train_dataset, train_labels, 32, len(label_map))

def check_dataset(dataset, labels, label_map, index):
	data = np.uint8(dataset[index]).reshape((28, 28))
	i = np.argwhere(labels[index] == 1)[0][0]
	im = Image.fromarray(data)
	im.show()
	print ("label:", label_map[i])

def load_model():
	dataset, labelset, label_map = load_dataset()
	dataset, labelset = randomize(dataset, labelset)
	train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labelset)
	train_dataset, train_labels = reformat(train_dataset, train_labels, 28, len(label_map))
	test_dataset, test_labels = reformat(test_dataset, test_labels, 28, len(label_map))
	print ("train_dataset:", train_dataset)
	print ("train_labels:", train_labels)
	print ("test_dataset:", test_dataset)
	print ("test_labels:", test_labels)
	check_dataset(train_dataset, train_labels, label_map, 0)
	return train_dataset, train_labels,test_dataset, test_labels, label_map

def get_name_and_image():
	base_dir = "E:/pest1/training_set/8/"
	all_image = os.listdir(base_dir)
	random_file = random.randint(0,10)
	base = os.path.basename('E:/pest1/training_set/8/'+ all_image[random_file])
	name = os.path.splitext(base)[0]
	image = Image.open('E:/pest1/training_set/8/'+ all_image[random_file])
	image = np.array(image)
	return name,image

IMAGE_HIGHT = 28
IMAGE_WIDTH = 28
MAX_CAPTCHA = 8
CHAR_SET_LEN = 1

def name2vec(name):
	vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
	for i,c in enumerate(name):
		idx = i * 1 + ord(c)-61
		vector[idx] = 1
	return vector

def vec2name(vec):
	name = []
	for i in vec:
		a = chr(i + 61)
		name.append(a)
	return "".join(name)

if __name__ == '__main__':
	load_model()