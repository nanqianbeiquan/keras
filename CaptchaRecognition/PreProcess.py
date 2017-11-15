# coding=utf-8

import cv2
import numpy as np
import time
import random
import os

class PreProcess(object):
    """description of class"""
    def ConvertToGray(self,Image,filename):
        GrayImage=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
        return GrayImage

    def ConvertTo1Bpp(self,GrayImage,filename):
        App,Bpp=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
        # cv2.imwrite('E:/pest1/'+'%d' % int(filename.split('.')[0])+'.jpg',Bpp[1])
        return Bpp

    def InterferLine(self,Bpp,filename):
        for i in range(50):
            for j in range(Bpp.shape[0]):
                Bpp[j][i]=255
        for i in range(161,Bpp.shape[1]):
            for j in range(0,Bpp.shape[0]):
                Bpp[j][i]=255
        m=1
        n=1
        for i in range(50,161):
            while(m<Bpp.shape[0]-1):
                if Bpp[m][i]==0:
                    if Bpp[m+1][i]==0:
                        n=m+1
                    elif m>0 and Bpp[m-1][i]==0:
                        n=m
                        m=n-1
                    else:
                        n=m+1
                    break
                elif m!=Bpp.shape[0]:
                    l=0
                    k=0
                    ll=m
                    kk=m
                    while(ll>0):
                        if Bpp[ll][i]==0:
                            ll=11-1
                            l=l+1
                        else:
                            break
                    while(kk>0):
                        if Bpp[kk][i]==0:
                            kk=kk-1
                            k=k+1
                        else:
                            break
                    if (l<=k and l!=0) or (k==0 and l!=0):
                        m=m-1
                    else:
                        m=m+1
                else:
                    break
                #endif
            #endwhile
            if m>0 and Bpp[m-1][i]==0 and Bpp[n-1][i]==0:
                continue
            else:
                 Bpp[m][i]=255
                 Bpp[n][i]=255
            #endif
        #endfor
        cv2.imwrite('E:/pest1/orgyzm/'+'%d' % int(filename.split('.')[0])+'.jpg',Bpp)
        return Bpp

    def CutImage(self,Bpp,filename):
        outpath='E:/pest1/test1'
        b1=np.zeros((Bpp.shape[0],20))

        for i in range(51,71):
            for j in range(0,Bpp.shape[0]):
                b1[j][i-50]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk').encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b1)

        b2=np.zeros((Bpp.shape[0],19))
        for i in range(73,92):
            for j in range(0,Bpp.shape[0]):
                b2[j][i-73]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[0].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b2)

        b3=np.zeros((Bpp.shape[0],19))
        for i in range(94,113):
            for j in range(0,Bpp.shape[0]):
                b3[j][i-113]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[0].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b3)

        b4=np.zeros((Bpp.shape[0],19))
        for i in range(115,134):
            for j in range(0,Bpp.shape[0]):
                b4[j][i-115]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[0].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b4)

        b5=np.zeros((Bpp.shape[0],19))
        for i in range(136,155):
            for j in range(0,Bpp.shape[0]):
                b5[j][i-136]=Bpp[j][i]
        cv2.imwrite(outpath+filename.decode('gbk')[0].encode('gbk')+'_'+'%d' %(time.time()*1000)+str(random.randint(1000,9999))+'.png',b5)
        return (b1,b2,b3,b4,b5)

if __name__ == '__main__':
    inpath = 'E:/pest/'
    PP=PreProcess()
    for root,dirs,files in os.walk(inpath):
        for filename in files:
            Img=cv2.imread(root+'/'+filename)#太坑，此处inpath不能包含中文路径
            GrayImage=PP.ConvertToGray(Img,filename)
            Bpp=PP.ConvertTo1Bpp(GrayImage,filename)
            Bpp_new=PP.InterferLine(Bpp,filename)
            b=PP.CutImage(Bpp_new,filename)

