# -*- coding: utf-8 -*-


import cv2
import os
from digit_train_use_cnn import Model
from nacao.PreProcess import PreProcess
import shutil

nacao_dict = {'0':'x',
              '1':'y',
              '2':'2',
              '3':'3',
              '4':'4',
              '5':'5',
              '6':'6',
              '7':'7',
              '8':'8',
              '9':'a',
              '10':'b',
              '11':'c',
              '12':'d',
              '13':'e',
              '14':'f',
              '15':'g',
              '16':'m',
              '17':'n',
              '18':'p',
              '19':'w'
        }

if __name__ == '__main__':
    inpath = 'E:\pest1\\nacao1'
    PP = PreProcess()
    for root,dirs,files in os.walk(inpath):
        for filename in files:
            Img = cv2.imread(root + '/' + filename)
            GrayImage = PP.ConvertToGray(Img, filename)
#            cv2.imshow('image',GrayImage)
#            cv2.waitKey (0)
            Bpp = PP.ConvertToBpp(GrayImage, filename)            
            Bpp_new = PP.InterferLine(Bpp, filename)
            Bpp_r = PP.RemoveLine(Bpp, filename)
            b = PP.CutImage(Bpp,filename)
            model = Model()
            model.load_model(file_path = 'nacao.train.model.h5')
            image_path = 'E:/python/keras/nacao/temp'
            dir_item = os.listdir(image_path)
            yzm_list = []
            yzm = ''
            for impa in dir_item:
                im_path = os.path.abspath(os.path.join(image_path,impa))
                image = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
                digitID = model.digit_predict(image)
        #        print('digitID:',digitID)
                for did in nacao_dict:
        #            print('did:',did)
        #            print("nacao_dict[%s]=" % did,nacao_dict[did])
                    if str(digitID) == str(did):
                        digitID = nacao_dict[did]
                        print('digitID:',digitID)
                        break
                yzm_list.append(digitID)
                yzm = yzm+digitID
            print('yzm_list:',yzm_list)
            print(yzm)
            print (image_path)
            shutil.rmtree(image_path)
            os.mkdir(image_path)
                    