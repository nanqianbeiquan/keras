# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = VGG16(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1./255)

# 生成训练图片
generator = datagen.flow_from_directory(
        'preview/catvsdog',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        shuffle=False)

# 生成验证图片
generator = datagen.flow_from_directory(
        'preview/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        shuffle=False)


#（3）得到bottleneck feature
bottleneck_features_train = model.predict_generator(generator, 50)
# 核心，steps是生成器要返回数据的轮数，每个epoch含有500张图片，与model.fit(samples_per_epoch)相对
np.save(open('bottleneck_features_train.npy', 'wb+'), bottleneck_features_train)

bottleneck_features_validation = model.predict_generator(generator, 10)
# 与model.fit(nb_val_samples)相对，一个epoch有800张图片，验证集
np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
# （1）导入bottleneck_features数据
train_data = np.load(open('bottleneck_features_train.npy' ,'rb'))

train_labels = np.array([0] * 80 + [1] * 80)  # matt,打标签

validation_data = np.load(open('bottleneck_features_validation.npy' ,'rb'))
validation_labels = np.array([0] * 16 + [1] * 16)  # matt,打标签

# （2）设置标签，并规范成Keras默认格式
train_labels = to_categorical(train_labels, 2)
validation_labels = to_categorical(validation_labels, 2)

# （3）写“小网络”的网络结构
model = Sequential()
print(train_data.shape)
print(validation_data.shape)
model.add(Flatten(input_shape=(4,4,512)))# 4*4*512
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(1, activation='sigmoid'))  # 二分类
model.add(Dense(2, activation='softmax'))  # matt,多分类
#model.add(Dense(1))
#model.add(Dense(5)) 
#model.add(Activation('softmax'))

# （4）设置参数并训练
model.compile(loss='categorical_crossentropy',   
# matt，多分类，不是binary_crossentropy
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=50, batch_size=16,
          validation_data=(validation_data, validation_labels))
model.save_weights('bottleneck_fc_model.h5')