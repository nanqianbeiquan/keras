# -*- coding: utf-8 -*-


import cv2
import os
from digit_train_use_cnn import Model

if __name__ == '__main__':
    model = Model()
    model.load_model(file_path = 'handwritten.train.model.h5')
    image_path = 'E:\pest1\mnist\sj'
    dir_item = os.listdir(image_path)
    for impa in dir_item:
        im_path = os.path.abspath(os.path.join(image_path,impa))
        image = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        digitID = model.digit_predict(image)
        print('digitID:',digitID)