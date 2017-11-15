# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread('E:/pest1/nacao/11.png')
#cv2.namedWindow("Image")   
cv2.imshow("Image",img)
cv2.waitKey(0)
#
#print(img)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
#腐蚀图像  
eroded = cv2.erode(img,kernel)  
#r,g,b = cv2.split(img)
#cv2.imshow("Image",r)
cv2.imshow("Eroded Image",eroded);  
cv2.waitKey(0)
#膨胀图像  
dilated = cv2.dilate(img,kernel)  
cv2.imshow("Dilated Image",dilated)
cv2.waitKey(0)

eroded = cv2.erode(dilated,kernel)  
#r,g,b = cv2.split(img)
#cv2.imshow("Image",r)
cv2.imshow("Eroded Image",eroded) 
cv2.waitKey(0)

img = cv2.imread('E:/pest1/nacao/11.png',0)  
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))  
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
#闭运算  
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  
#显示腐蚀后的图像  
cv2.imshow("Close",closed)
cv2.waitKey(0)
#开运算 
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  
#显示腐蚀后的图像  
cv2.imshow("Open", opened)
cv2.waitKey(0)

#将两幅图像相减获得边，第一个参数是膨胀后的图像，第二个参数是腐蚀后的图像  
result = cv2.absdiff(dilated,eroded)
#上面得到的结果是灰度图，将其二值化以便更清楚的观察结果  
retval, result = cv2.threshold(result, 60, 255, cv2.THRESH_BINARY)
#反色，即对二值图每个像素取反  
result = cv2.bitwise_not(result)
#显示图像  
cv2.imshow("result",result)
cv2.waitKey(0)
 
result = cv2.blur(img, (5,5))  
cv2.imshow("Blur", result)
cv2.waitKey(0)  

result = cv2.blur(img, (3,3))  
cv2.imshow("Blur", result)
cv2.waitKey(0) 

# 高斯模糊
gaussianResult = cv2.GaussianBlur(img,(5,5),1.5)
cv2.imshow("gaussianResult", gaussianResult)
cv2.waitKey(0) 

# 中值滤波
medianBlur = cv2.medianBlur(img,5)  
cv2.imshow("medianBlur", medianBlur)
cv2.waitKey(0) 

def salt(img, n):    
    for k in range(n):    
        i = int(np.random.random() * img.shape[1]);    
        j = int(np.random.random() * img.shape[0]);    
        if img.ndim == 2:     
            img[j,i] = 255    
        elif img.ndim == 3:     
            img[j,i,0]= 255    
            img[j,i,1]= 255    
            img[j,i,2]= 255    
    return img 
result = salt(img, 500)    
median = cv2.medianBlur(result, 5)  
cv2.imshow("Salt", result)  
cv2.imshow("Median", median)  
cv2.waitKey(0)  

gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
dst = cv2.convertScaleAbs(gray_lap)
cv2.imshow('laplacian',dst)  
cv2.waitKey(0)

img = cv2.GaussianBlur(img,(3,3),0)
canny = cv2.Canny(img, 50, 150)
cv2.imshow("Canny", canny)
cv2.waitKey(0)


#img = cv2.imread("E:/pest1/nacao/11.png", 0)  
#img = cv2.GaussianBlur(img,(3,3),0)  
#edges = cv2.Canny(img, 50, 150, apertureSize = 3)  
#lines = cv2.HoughLines(edges,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值  
#result = img.copy()
#print(lines[0])
#for line in lines:  
#    rho = line[0] #第一个元素是距离rho  
#    theta= line[1] #第二个元素是角度theta 
#    print (rho,theta)
#    if  (theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)): #垂直直线  
#        pt1 = (int(rho/np.cos(theta)),0) 
#        pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])  
#        #绘制一条白线  
#        cv2.line( result, pt1, pt2, (255)) 
#    else: #水平直线 
#        pt1 = (0,int(rho/np.sin(theta)))  
#        pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))  
#        cv2.line(result, pt1, pt2, (255), 1)  
#cv2.imshow("Canny", edges ) 
#cv2.imshow("Result", result)
#cv2.waitKey(0)  

equ = cv2.equalizeHist(img)
cv2.imshow("equ",equ)
cv2.waitKey(0)


img = cv2.imread('E:/pest1/nacao/11.png')  
#gray = cv2.imread('E:/pest1/nacao/11.png',cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(img,contours,-1,(0,0,255),3)
cv2.imshow("img", img)
cv2.imshow("binary", binary)  
cv2.waitKey(0)