import pandas as pd
import numpy as np
import cv2
import os
from shutil import copy
from PIL import Image
import cv2
mappings = pd.read_csv("train.csv")


one_hot_dummies = pd.get_dummies(mappings['Id'])

images =list(mappings['Image'])
Id = list(mappings['Id'])



classes = list(one_hot_dummies.columns)
##
##ids = list(mappings['Image'])
##
##print(one_hot_dummies)

def create_dict_mappings(images,Id):
    dicti = {}
    for i in range(len(images)):
        dicti[images[i]]=Id[i]
    return dicti

map_dict = create_dict_mappings(images,Id)


def create_train_path():
    newpath = './data'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    newpath = './data/train'

    if not os.path.exists(newpath):
        os.makedirs(newpath)


def create_training_folders():
    for class_i in classes:
        newpath = './data/train/{}'.format(class_i)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

    
def add_images_to_training_folders():
    ad
    src_dir = 'C:\\Users\\demon\\Desktop\\Whale Dataset\\train\\{}'
    dest_dir = 'C:\\Users\\demon\\Desktop\\Whale Dataset\\data\\train\\{}\\'
    os.chdir(r'C:\Users\demon\Desktop\Whale Dataset\train')
    map_dict = create_dict_mappings(images,Id)
    for image in map_dict:
        print("Copying :",image,"Folder :",dest_dir.format(map_dict[image]))
        copy(image,
                 dest_dir.format(map_dict[image]))
        print("Moving to approprate folder")
            
         
