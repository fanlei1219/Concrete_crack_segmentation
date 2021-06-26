import os
import torch
import numpy as np
import shutil
import cv2
from torch.utils.data import random_split

def split_train_val(custom_path):
    folder_path = custom_path
    folder_img = folder_path + '/'+ 'Image'
    folder_label = folder_img.replace('Image', 'Label')
    imgs = os.listdir(folder_img)
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    if not os.path.exists('./datasets/ImageSets'):
        os.mkdir('./datasets/ImageSets')
    imgs_to_path = './datasets/JPEGImages'
    labels_to_path = './datasets/SegmentationClass'
    files_path = './datasets/ImageSets/Segmentation'
    if not os.path.exists(imgs_to_path):
        os.mkdir(imgs_to_path)
    if not os.path.exists(labels_to_path):
        os.mkdir(labels_to_path)
    if not os.path.exists(files_path):
        os.mkdir(files_path)
    for img in imgs:
        img_path = folder_img+'/'+img
        img_to_path = imgs_to_path + '/'+img
        shutil.copy(img_path, img_to_path)
        label_path = folder_label+ '/'+img
        label_to_path = labels_to_path+ '/'+ img
        shutil.copy(label_path, label_to_path)
    f1 = open(files_path + '/' + 'train.txt', 'w+')
    f2 = open(files_path + '/' + 'val.txt', 'w+')
    f3 = open(files_path + '/' + 'trianval.txt', 'w+')
    num_files = len(imgs)
    num_train = int(0.9*num_files)
    num_val = num_files - num_train
    train,val = random_split(imgs, [num_train, num_val])
    for all_file in imgs:
        all_file_name = all_file.split('.')[0]
        f3.write(all_file_name + '\n')
    for file in train:
        img_name = file.split('.')[0]
        f1.write(img_name + '\n')
    for file1 in val:
        img_name1 = file1.split('.')[0]
        f2.write(img_name1 + '\n')
    f1.close()
    f2.close()
    f3.close()
if __name__ == '__main__':
    custom_path = './data'
    split_train_val(custom_path)