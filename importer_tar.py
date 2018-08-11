# -*- coding: utf-8 -*-
"""
Created on Fri Aug 03 22:32:23 2018

@author: yonic
"""

import tarfile
import cv2
import numpy as np
from PIL import Image
import os

DB_FOLDER = "/home/yonic/repos/CycleGAN-Tensorflow-PyTorch/db"

class Importer:
    def __init__(self,fname):
        self.fname = os.path.join(DB_FOLDER,fname)
        self.tar_db = tarfile.open(self.fname)
        self.catalog = []
        i = self.tar_db.next()
        while i:
            self.catalog.append(i.get_info())
            i = self.tar_db.next()
        self.catalog = sorted(self.catalog,key=lambda x:x['name'])
            
    def get_full_catalog(self):
        return self.catalog
    
    def get_sorted_image_name(self):
        return [f['name'] for f in self.catalog][1:]#first is a folder name!
    
    def get_stream(self,tar_fname):
        fileobj = self.tar_db.extractfile(tar_fname)
        return fileobj
    
    '''Return extract RGB image
    '''
    def get_image(self,tar_fname):
        image = Image.open(self.get_stream(tar_fname))
        image = np.array(image)
        return image
        #return  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
def test(tar_fname="cameron_images.tgz"):
    i = Importer(tar_fname)
    image_names = i.get_sorted_image_name()
    image = i.get_image(image_names[0])
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    

if __name__ == "__main__":
    test()
    
        
    