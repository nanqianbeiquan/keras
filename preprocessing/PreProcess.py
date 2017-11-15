# -*- coding: utf-8 -*-

import cv2
import numpy as np

class PreProcess(object):
    
    def ConvertToGray(Image):
        GrayImage = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        return GrayImage
    
    
    def ConvertToBpp(GrayImage):
        App,Bpp = cv2.threshold(GrayImage,130,255,cv2.THRESH_BINARY)
        return Bpp

    
    def ConvertToSalt(Bpp,n):
        for k in range(n):
            i = int(np.random.random() * Bpp.shape[1])
            j = int(np.random.random() * Bpp.shape[0])
            if Bpp.ndim == 2:
                Bpp[j,i] = 255
            elif Bpp.ndim == 3:
                Bpp[j,i,0] = 255
                Bpp[j,i,1] = 255
                Bpp[j,i,2] = 255
        return Bpp
        
    
    def ConvertToDilate(Bpp):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))
        Bpp = cv2.blur(Bpp, (1,1))       
        Bpp = cv2.medianBlur(Bpp,5) 
#        Bpp = PP.ConvertToSalt(Bpp,1000)          
        Bpp = cv2.dilate(Bpp,kernel)
        Bpp = cv2.erode(Bpp,kernel)        
        return Bpp
    
    def InterferLine(Bpp):
        for i in range(56):
            for j in range(Bpp.shape[0]):
                Bpp[j][i] = 255                
        for j in range(171,Bpp.shape[1]):
            for i in range(Bpp.shape[0]):
                Bpp[i][j] = 255
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
#            else:
#                Bpp[m][i] = 255
#                Bpp[n][i] = 255
        return Bpp
    
    def InterferPoint(Bpp):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        Bpp = cv2.dilate(Bpp,kernel)
#        Bpp = cv2.erode(Bpp,kernel)
        return Bpp
       
    
    def CutImage(Bpp):
        TotalBlack = 0
        for j in range(56,170):
            for i in range(50):
                if Bpp[i][j] == 0:
                    TotalBlack += 1
            if TotalBlack < 8:
                cv2.line(Bpp,(j,0),(j,50),(120,120,120),1)
            TotalBlack =0
        return Bpp
   
           
if __name__ == '__main__':
    inpath = 'E:/pest1/nacao/18.png'
    img = cv2.imread(inpath)
    PP = PreProcess
    GrayImage = PP.ConvertToGray(img)
    Bpp = PP.ConvertToBpp(GrayImage)
    Bpp = PP.InterferLine(Bpp)       
    Bpp = PP.ConvertToDilate(Bpp)  
    Bpp = PP.InterferPoint(Bpp)
#    Bpp = PP.CutImage(Bpp)
#    cv2.imshow('img',Bpp)
    cv2.imwrite('118.png',Bpp)
#    cv2.imwrite('img',img)
#    cv2.waitKey(0)
    
