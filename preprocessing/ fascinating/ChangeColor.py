# -*- coding: utf-8 -*-

import cv2

image = cv2.imread('E:\\tjpic\\1.png')

image = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)

image[:,:,1] = 127

image = cv2.cvtColor(image,cv2.COLOR_LAB2BGR)

cv2.imwrite('test.jpg',image)

