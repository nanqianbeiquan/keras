# -*- coding: utf-8 -*-

import random
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from input_data import load_dataset, resize_image, IMAGE_SIZE

class Dataset:
    def __init__(self,path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None
        
        #  验证集
        self.valid_images = None
        self.valid_labels = None
        
        # 测试集
        self.test_images = None
        self.test_labels = None
        
        # 数据集加验证
        self.path_name = path_name
        
        # 当前库采用的顺序
        self.input_shape = None
    
    # 根据加载的数据集按照交叉验证的原则划分数据集并进行相关验证
    def load(self,img_rows = IMAGE_SIZE,img_cols = IMAGE_SIZE,
             img_channels = 3, nb_classes = 2):
        # 加载数据集到内存
        images,labels = load_dataset(self.path_name)
        train_images,valid_images,train_labels,valid_labels = train_test_split(
                images,labels,test_size = 0.3,random_state = random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(
                images, labels, test_size = 0.5, random_state = random.randint(0, 100))
        #当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        #这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)    
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols,img_channels)
        
        #输出训练集、验证集、测试集的数量
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')
        #使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        #类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)
        #像素数据浮点化以便归一化
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')
        #将其归一化,图像的各像素值归一化到0~1区间
        train_images /= 255
        valid_images /= 255
        test_images /= 255  
        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images  = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels  = test_labels
        
# CNN网络类
class Model:
    def __init__(self):
        self.model = None
        
    # 建立模型
    def build_model(self,dataset,nb_classes=2):
        #构建一个空的网络模型，它是一个线性堆叠模型，各神经网络层会被顺序添加，专业名称为序贯模型或线性堆叠模型
        self.model = Sequential()
        #以下代码将顺序添加CNN网络需要的各层，一个add就是一个网络层
        self.model.add(Conv2D(32, kernel_size = (3,3),
                              padding = 'same',
                              input_shape = dataset.input_shape,
                              activation = 'relu'))
        self.model.add(Conv2D(32,(3,3),activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
        self.model.add(Conv2D(64,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
        self.model.add(Conv2D(128,(3,3),activation='relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Flatten())
        self.model.add(Dense(512,activation = 'relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes, activation = 'softmax'))
        
        self.model.summary()
        
    def train(self, dataset, batch_size = 20, np_epoch = 50, data_augmentation = True):
        sgd = SGD(lr = 0.01,decay = 1e-6,momentum = 0.9, nesterov = True)
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = sgd,
                           metrics = ['accuracy'])    

        if not data_augmentation:
            self.model.fit(dataset.train_images,dataset.train_labels,
                           batch_size = batch_size,
                           epochs = np_epoch,
                           verbose = 1,
                           validation_data = (dataset.valid_images,dataset.valid_labels),
                           shuffle = True)
        else:
            datagen = ImageDataGenerator(
                    featurewise_center = False,
                    samplewise_center = False,
                    featurewise_std_normalization = False,
                    samplewise_std_normalization = False,
                    zca_whitening = False,
#                    rotation_range = 10,
#                    width_shift_range = 0.2,
#                    height_shift_range = 0.2,
                    horizontal_flip = False,
                    vertical_flip = False)
            
            datagen.fit(dataset.train_images)
            
            self.model.fit_generator(datagen.flow(dataset.train_images,
                      dataset.train_labels,
                      batch_size = batch_size),
                      samples_per_epoch = dataset.train_images.shape[0],
                      epochs = np_epoch,
                      verbose = 1,
                      validation_data = (dataset.valid_images,dataset.valid_labels))
            
if __name__ == '__main__':
    dataset = Dataset('E:\pest1\catanddog\catvsdog')
    dataset.load()
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
        

