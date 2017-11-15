# -*- coding: utf-8 -*-

import numpy as np
import os
from PIL import Image
import struct

#data_dir = 'E:/pest1/hanzi/handwriting/'
#
#
#train_data_dir = os.path.join(data_dir, '1241-c.gnt')
#test_data_dir = os.path.join(data_dir, '1289-c.gnt')

train_data_dir = 'HWDB1.1tst_gnt'
test_data_dir = '1291.gnt'

#f = open('3500.txt', 'r')
#total_words = f.readlines()[0].decode("utf-8")
#print(total_words)

def read_from_gnt_dir(gnt_dir = train_data_dir):
    def one_file(f):
        header_size = 10
        while True:
            header = np.fromfile(f,dtype='uint8', count = header_size)
            print('header:',header)
            if not header.size:
                break
            sample_size = header[0]+(header[1] << 8)+(header[2] << 16)+(header[3] << 24)
            tagcode = header[5] + (header[4] << 8)
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            print(height)
            if header_size + width * height != sample_size:
                break
            image = np.fromfile(f,dtype='uint8',count = width*height).reshape((height,width))
            yield image,tagcode
            
    for file_name in os.listdir(gnt_dir):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(gnt_dir, file_name)
            with open(file_path, 'rb') as f:
                for image,tagcode in one_file(f):
                    yield image,tagcode
                        
char_set = set()
for image, tagcode in read_from_gnt_dir(gnt_dir=train_data_dir):
#for im,tagcode in read_from_gnt_dir(gnt_dir = train_data_dir):
    tagcode_unicode = struct.pack('>H',tagcode).decode('gb2312')
    char_set.add(tagcode_unicode)

char_list = list(char_set)
char_dict = dict(zip(sorted(char_list),range(len(char_list))))
print(len(char_dict))
        