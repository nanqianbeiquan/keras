# -*- coding: utf-8 -*-

from load_digit_dataset import load_dataset,resize_image,IMAGE_SIZE
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from keras.optimizers import SGD,Adadelta,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

class Dataset():
    def __init__(self,path_name):
        self.train_images = None
        self.train_labels = None
        
        self.valid_images = None
        self.valid_labels = None
        
        self.test_images = None
        self.test_labels = None
        
        self.path_name = path_name
        
        self.input_shape = None
        
    def load(self,img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE,
             img_channels = 1,nb_classes = 10):
        images,labels = load_dataset(self.path_name)
        
        train_images,valid_images,train_labels, valid_labels = train_test_split(images,labels)
        test_images, _, test_labels, _ = train_test_split(images,labels,test_size = 0.5)

        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0],img_channels,img_rows,img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0],img_channels,img_rows,img_cols)
            test_images = test_images.reshape(test_images.shape[0],img_channels,img_rows,img_cols)
            self.input_shape = (img_channels,img_rows,img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0],img_rows,img_cols,img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0],img_rows,img_cols,img_channels)
            test_images = test_images.reshape(test_images.shape[0],img_rows,img_cols,img_channels)
            self.input_shape =(img_rows,img_cols,img_channels)

        print('train samples:', train_images.shape[0])
        print('valid samples:', valid_images.shape[0])
        print('test samples:', test_images.shape[0])
        print('input shape:', self.input_shape)
        
        train_labels = np_utils.to_categorical(train_labels,nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels,nb_classes)
        test_labels = np_utils.to_categorical(test_labels,nb_classes)
        
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')
        
        train_images /= 255
        valid_images /= 255
        test_images /= 255
        
        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels
        
class Model():
    
    def __init__(self):
        self.model = None
        
    def build_model(self,dataset,nb_classes = 10):
        self.model = Sequential()
        
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
        
    def train(self, dataset, batch_size = 20, np_epoch = 10, data_argumentation = True):
        sgd = SGD(lr = 0.01,decay = 1e-6,momentum = 0.9, nesterov = True)
        self.model.compile(loss = 'categorical_crossentropy',
                           optimizer = sgd,
                           metrics = ['accuracy'])


        if not data_argumentation:
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
                    rotation_range = 20,
                    width_shift_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = False)
            
            datagen.fit(dataset.train_images)
            
            self.model.fit_generator(datagen.flow(dataset.train_images,
                      dataset.train_labels,
                      batch_size = batch_size),
                      samples_per_epoch = dataset.train_images.shape[0],
                      epochs = np_epoch,
                      verbose = 1,
                      validation_data = (dataset.valid_images,dataset.valid_labels))
            
    MODEL_PATH = 'handwritten.train.model.h5'
    
    def save_model(self,file_path = MODEL_PATH):
        self.model.save(file_path)

    def load_model(self,file_path = MODEL_PATH):
        self.model = load_model(file_path) 

    def evaluate(self,dataset):
        score = self.model.evaluate(dataset.test_images,dataset.test_labels,verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100)) 
        
    def digit_predict(self,image):
        if K.image_dim_ordering() == 'th' and image.shape !=(1, 3, IMAGE_SIZE,IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        elif K.image_dim_ordering()=='tf' and image.shape !=(1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        else:
            print('image.shape:',image.shape)
            
        image = image.astype('float32')
        image /= 255
        
        result = self.model.predict_proba(image)
        result = self.model.predict_classes(image)
        
        return result[0]
        
        
if __name__ == '__main__':
    dataset = Dataset('E:\pest1\mnist')
    dataset.load()
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model('handwritten.train.model.h5')
    model.load_model('handwritten.train.model.h5')
    model.evaluate(dataset)
        
