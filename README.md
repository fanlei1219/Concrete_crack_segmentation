# Concrete_crack_segmentation
代码地址： https://github.com/fanlei1219/Concrete_crack_segmentation   
系统：Windows10  
环境：Pytorch 1.8.1，Numpy  
需安装包：pytorch, numpy, os, pillow, torchvision, argparse, tqdm, tensorboardX, cv2   
训练操作方式：将训练图像及标签放入文件夹'./data'，运行代码'data.py'将图像分为训练集和验证集；运行'train.py'进行训练   
测试操作方式：将测试图像放入文件夹'./datasets/test/ test_imgs'，运行'test.py'代码，测试结果存入文件夹'./datasets/test/ test_pred'   
测试结果指标计算方式：将测试图像标签放入文件夹'./datasets/test/ test_labels'，运行'test_metric.ipynb'   
