# coding=utf-8

from load_dog_dataset import load_dataset, resize_image, IMAGE_SIZE
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np


class Dateset():
	def __init__(self, path_name):
		# 训练集
		self.train_images = None
		self.train_labels = None

		# 验证集
		self.valid_images = None
		self.valid_labels = None

		# 测试集
		self.test_images = None
		self.test_labels = None

		# 数据集加载路径
		self.path_name = path_name

		# 当前库才用的维度顺序
		self.input_shape = None

	# 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
	def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
			 img_channels=3, nb_classes=6):
		# 加载数据集到内存
		images, labels = load_dataset(self.path_name)
		# print 'images:',images
		# print 'labels:',labels
		train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels)
		# train_images,valid_images,train_labels,valid_labels = train_test_split(images,labels,
		# 	test_size=0.3, random_state=(0,100))
		_, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5)
		# 当前维度的顺序如果为'th',则输入图片的顺序为：channels,rows,clos,否则为rows,cols,channels
		# 这部分代码是根据keras要求的维度顺序重组训练数据集
		if K.image_dim_ordering() == 'th':
			train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
			valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
			test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
			self.input_shape = (img_channels, img_rows, img_cols)
		else:
			train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
			valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
			test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
			self.input_shape = (img_rows, img_cols, img_channels)

		# 输出训练集，验证集，测试集的数量
		print ('train samples:', train_images.shape[0])
		print ('valid samples:', valid_images.shape[0])
		print ('test samples:', test_images.shape[0])
		print ('self.input_shape:', self.input_shape)

		# 模型使用categorical_crossentropy作为损失函数，因此要根据类别数量nb_classes将
		# 类别标签进行 one-hot编码使其量化，在这里我们的类别只有两种，经过转换后标签数据变为二维
		# print 'train_labels:',train_labels
		train_labels = np_utils.to_categorical(train_labels, nb_classes)
		valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
		test_labels = np_utils.to_categorical(test_labels, nb_classes)
		# print 'train_labels:',train_labels
		# 像素数据集浮点化以便归一化
		train_images = train_images.astype('float32')
		valid_images = valid_images.astype('float32')
		test_images = test_images.astype('float32')

		# 归一化，图像的各像素归一化到[0,1]区间
		train_images /= 255
		valid_images /= 255
		test_images /= 255

		self.train_images = train_images
		self.valid_images = valid_images
		self.test_images = test_images
		self.train_labels = train_labels
		self.valid_labels = valid_labels
		self.test_labels = test_labels
		# print('self.valid_images:',self.valid_images)
		# print('self.test_images:',self.test_images)


# CNN网络模型
class Model():
	def __init__(self):
		self.model = None

	# 建立模型
	def build_model(self, dataset, nb_classes=6):
		# 构建一个空间的网络模型，它是一个线性堆叠模型，各神经网络层被顺序添加，专业名称为
		# 序贯模型或线性堆叠模型
		self.model = Sequential()

		# 以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
		self.model.add(Conv2D(32, kernel_size=(3, 3),
							  padding='same',
							  input_shape=dataset.input_shape,
							  activation='relu'))  # 1. 2维卷积层

		self.model.add(Conv2D(32, (3, 3), activation='relu'))  # 2. 2维卷积层

		self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 3. 2维池化层

		self.model.add(Dropout(0.25))  # 4. Dropout层

		self.model.add(Conv2D(64, (3, 3), padding='same',
							  activation='relu'))  # 5. 2维卷积层

		self.model.add(Conv2D(64, (3, 3), activation='relu'))  # 6. 2维卷积层

		self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 7. 2为池化层

		self.model.add(Dropout(0.25))  # 8.Dropout层

		self.model.add(Flatten())  # 9.Flatten层

		self.model.add(Dense(512, activation='relu'))  # 10.Dense层又被称作全连接层

		self.model.add(Dropout(0.5))  # 11.Dropout层

		self.model.add(Dense(nb_classes, activation='softmax'))  # 12.Dense层，分类层，输出最终结果

		# 输出模型状况
		self.model.summary()

	def train(self, dataset, batch_size=20, nb_epoch=10, data_argumentation=True):
		# 采用SGD+momentum的优化器进行训练，首先生成一个优化对象
		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy',
						   optimizer=sgd,
						   metrics=['accuracy'])  # 完成实际的模型配置工作


		# 不使用数据提升，所谓提升就是从我们提供的训练数据中利用旋转，翻转，加噪声等方法所创造的
		# 训练数据，有意识的提升训练数据的规模，增加模型的训练量
		if not data_argumentation:
			self.model.fit(dataset.train_images,
						   dataset.train_labels,
						   batch_size = batch_size,
						   epochs = nb_epoch,
						   validation_data = (dataset.valid_images,dataset.valid_labels),
						   shuffle = True)
		# 使用实时数据提升
		else:
			# 定义数据生成器，用于数据的提升，且返回一个生成器对象datagen,datagen每被调用一次
			# 其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
			datagen = ImageDataGenerator(
				featurewise_center = False,              # 是否使输入数据去中心化
				samplewise_center = False,              # 是否使输入数据的每个样本均值为0
				featurewise_std_normalization = False,  # 是否数据标准化（输入数据除以数据集的标准差）
				samplewise_std_normalization = False,   # 是否将每个样本数据除以自身的标准差
				zca_whitening = False,                  # 是否对输入数据zca白化
				rotation_range = 20,                    # 是否提升图片随机旋转的角度（范围0~180）
#				width_shift_range = 0.2,                # 数据提升图片水平偏移的角度（宽度为图片宽度的占比，0~1之间的浮点数）
#				height_shift_range = 0.2,               # 同上 ，垂直偏移
				horizontal_flip = True,                 # 是否进行随机水平翻转
				vertical_flip = False                   # 是否进行随机垂直翻转
			)

			# 计算整个训练样本集的数量以用于特征值归一化，ZCA白化等处理
			datagen.fit(dataset.train_images)

			# 利用生成器开始训练模型
			self.model.fit_generator(datagen.flow(dataset.train_images,dataset.train_labels,
									 batch_size = batch_size),
									 samples_per_epoch = dataset.train_images.shape[0],
									 epochs = nb_epoch,
									 verbose = 1,
									 validation_data = (dataset.valid_images,dataset.valid_labels))

	MODEL_PATH = './dog.train.model.h5'

	def save_model(self, file_path = MODEL_PATH):
		self.model.save(file_path)

	def load_model(self, file_path = MODEL_PATH):
		self.model = load_model(file_path)
		# h5py.File(file_path,'r')
		# self.model.load(file_path)

	def evaluate(self, dataset):
		score = self.model.evaluate(dataset.test_images, dataset.test_labels,verbose = 1)
		print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

	# 识别狗
	def dog_predict(self,image):
		# 依然根据后端系统确定维度顺序
		if K.image_dim_ordering() == 'th' and image.shape !=(1, 3, IMAGE_SIZE,IMAGE_SIZE):
			image = resize_image(image) #尺寸必须与测试集一致应该是IMAGE_SIZE * IMAGE_SIZE
			image = image.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE) # 与训练模型不同，这次只是针对一张图片进行预测
		elif K.image_dim_ordering()=='tf' and image.shape !=(1, 3, IMAGE_SIZE, IMAGE_SIZE):
			image = resize_image(image)
			image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
		else:
			print('image.shape:',image.shape)

		# 浮点并归一化
		image = image.astype('float32')
		image /= 255

		# 给出输入属于个类别的概率，我们是二值类别，则该函数会给出输出图像属于0,1的概率各为多少
		result = self.model.predict_proba(image)
		print('result:',self.model.predict(image))

		# 给出类别0或1
		result = self.model.predict_classes(image)
		print('result2:',result)

		# 返回类型的预测结果
		return result[0]

if __name__ == '__main__':
	dataset = Dateset('E:\pest1\dogs')
	dataset.load()
	model = Model()
	model.build_model(dataset)
	model.train(dataset)
	model.save_model(file_path = './dog.train.model.h5')
	model.load_model(file_path = './dog.train.model.h5')
	model.evaluate(dataset)
