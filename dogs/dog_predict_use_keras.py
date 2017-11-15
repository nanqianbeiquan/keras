from dog_train_use_keras import Model
import cv2
import os

if __name__ == '__main__':
    model = Model()
    model.load_model(file_path = 'dog.train.model.h5')
    image_path = 'E:\pest1\Images\\n02085620-Chihuahua'
    dir_item = os.listdir(image_path)
    for impa in dir_item:
        im_path = os.path.abspath(os.path.join(image_path, impa))
        image = cv2.imread(im_path)
        dogID = model.dog_predict(image)
        print('dogID:',dogID)




