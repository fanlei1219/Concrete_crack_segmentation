import os
# import cv2
from PIL import Image
import numpy as np

img_path = './test3'
img_save_path = './test3_twovalue'
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)
imgs = os.listdir(img_path)
for img in imgs:
    img_file = img_path + '/' + img
    img_name = img.split('.')[0]
    img = Image.open(img_file)
    img = np.array(img)
    img_ave = np.average(img)
    img = np.where(img > img_ave, 255, 0)
    img_save = img_save_path + img_name + '.png'
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_save)
    print('two value done!')