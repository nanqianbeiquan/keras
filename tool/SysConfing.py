# -*- coding: utf-8 -*-

import sys
import os

def save_image_path():
    return os.path.join(sys.path[0], '..\\data\\' + str(os.getpid()) + ".png")
    
def delete_image_path():
    os.remove(image_path)
    
if __name__ == '__main__':
    image_path = r'E:\pest1\nacao3\15050957670923244.png'
#    delete_image_path()
    save_image_path()