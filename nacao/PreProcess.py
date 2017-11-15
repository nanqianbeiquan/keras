# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import random
import time

class PreProcess(object):
    
    def ConvertToGray(self,Image,filename):
        GrayImage = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        return GrayImage
    
    def ConvertToBpp(self,GrayImage,filename):
        App,Bpp = cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)       
        return Bpp
    
    def RemoveLine(self,Bpp,filename):
        m=1
        n=1
        near_dots = 0
        for x in range(Bpp.shape[0]-1):
            for y in range(Bpp.shape[1]-1):
                pix = Bpp[x][y]
                if pix == Bpp[x-1][y-1]:
                    near_dots += 1
                if pix == Bpp[x-1][y]:
                    near_dots += 1
                if pix == Bpp[x-1][y+1]:
                    near_dots += 1
                if pix == Bpp[x][y-1]:
                    near_dots += 1
                if pix == Bpp[x][y+1]:
                    near_dots += 1
                if pix == Bpp[x+1][y-1]:
                    near_dots += 1
                if pix == Bpp[x+1][y]:
                    near_dots += 1
                if pix == Bpp[x+1][y+1]:
                    near_dots += 1
            if near_dots < 5:
                Bpp[x][y] = Bpp[x][y-1]
        cv2.imwrite('1.jpg', Bpp)
        return Bpp
    
    
    def InterferLine(self,Bpp,filename):
        for i in range(50):
            for j in range(Bpp.shape[0]):
                Bpp[j][i] = 255
        for j in range(171,Bpp.shape[1]):
            for i in range(0,Bpp.shape[0]):
                Bpp[j][i] = 255
        
        m = 1
        n = 1
        for i in range(50, 171):
            while (m < Bpp.shape[0]-1):
                if Bpp[m][i] == 0:
                    if Bpp[m+1][i] == 0:
                        n = m+1
                    elif m>0 and Bpp[m-1][i] == 0:
                        n = m
                        m = n-1
                    else:
                        n = m+1
                    break
                elif m != Bpp.shape[0]:
                    l = 0
                    k = 0
                    ll = m
                    kk = m
                    while(ll>0):
                        if Bpp[ll][i] == 0:
                            ll = ll-1
                            l = l+1
                        else:
                            break
                    while(kk>0):
                        if Bpp[kk][i] == 0:
                            kk = kk-1
                            k = k+1
                        else:
                            break
                    if (l <= k and l != 0) or (k == 0 and l != 0):
                        m = m-1
                    else:
                        m = m+1
                else:
                    break
            if m>0 and Bpp[m-1][i] == 0 and Bpp[n-1][i] == 0:
                continue
            else:
                Bpp[m][i] = 255
                Bpp[n][i] = 255       
#        cv2.imwrite(filename+'1.jpg', Bpp)                               
        return Bpp
        
    def CutImage(self, Bpp, filename):
        outpath = 'E:/python/keras/nacao/temp/'
        
        b1 = np.zeros((Bpp.shape[0],23))
        
        for i in range(57,80):
            for j in range(0,Bpp.shape[0]):
                b1[j][i-57] = Bpp[j][i]
        cv2.imwrite(outpath+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b1)
        
        b2 = np.zeros((Bpp.shape[0],21))
        
        for i in range(81,102):
            for j in range(0,Bpp.shape[0]):
                b2[j][i-81] = Bpp[j][i]
        cv2.imwrite(outpath +'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b2)
                
        b3 = np.zeros((Bpp.shape[0],21))
        for i in range(102,123):
            for j in range(0,Bpp.shape[0]):
                b3[j][i-102] = Bpp[j][i]
        cv2.imwrite(outpath+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b3)

        b4 = np.zeros((Bpp.shape[0],21))
        
        for i in range(124,145):
            for j in range(0,Bpp.shape[0]):
                b4[j][i-124] = Bpp[j][i]
        cv2.imwrite(outpath+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b4)
                
        b5 = np.zeros((Bpp.shape[0],23))
        for i in range(145,168):
            for j in range(0,Bpp.shape[0]):
                b5[j][i-145] = Bpp[j][i]
        cv2.imwrite(outpath+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b5)
        return (b1,b2,b3,b4,b5)
    
    def InterferPoint(self,Bpp,filename):        
        m = 1
        n = 1
        for i in range(0, 20):
            while (m < Bpp.shape[0]-1):
                if Bpp[m][i] == 0:
                    if Bpp[m+1][i] == 0:
                        n = m+1
                    elif m>0 and Bpp[m-1][i] == 0:
                        n = m
                        m = n-1
                    else:
                        n = m+1
                    break
                elif m != Bpp.shape[0]:
                    l = 0
                    k = 0
                    ll = m
                    kk = m
                    while(ll>0):
                        if Bpp[ll][i] == 0:
                            ll = ll-1
                            l = l+1
                        else:
                            break
                    while(kk>0):
                        if Bpp[kk][i] == 0:
                            kk = kk-1
                            k = k+1
                        else:
                            break
                    if (l <= k and l != 0) or (k == 0 and l != 0):
                        m = m-1
                    else:
                        m = m+1
                else:
                    break
            if m>0 and Bpp[m-1][i] == 0 and Bpp[n-1][i] == 0:
                continue
            else:
                Bpp[m][i] = 255
                Bpp[n][i] = 255
        cv2.imwrite('1.jpg', Bpp)                               
        return Bpp


    
if __name__ == '__main__':
    inpath = 'E:\pest1\\nacao'
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
    inpath2 = 'E:\pest1\\nacao1'
    outpath2 = 'E:\pest1\\nacao3\\'
    for root,dirs,files in os.walk(inpath2):
        for filename in files:
            Img = cv2.imread(root + '/' + filename)
            GrayImage = PP.ConvertToGray(Img, filename)
            Bpp = PP.ConvertToBpp(GrayImage, filename)
            p = PP.InterferPoint(Bpp, filename)
            cv2.imwrite(outpath2+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',p)
            