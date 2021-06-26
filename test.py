import argparse
import os
import numpy as np
import torch
import tqdm
from deeplab_model.datasets import utils
from PIL import Image
from deeplab_model.deeplab import *
from deeplab_model.metrics import Evaluator
import time


class Tester(object):
    def __init__(self, args):
        if not os.path.isfile(args.weights):
            raise RuntimeError('no checkpoint found at "{}"'.format(args.weights))

        self.args = args
        self.color_map = utils.get_pascal_labels()
        self.nclass = args.num_class

        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)

        self.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.weights)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(device)
        self.evaluator = Evaluator(self.nclass)


    def test(self):
        self.model.eval()
        self.evaluator.reset()
        data_dir = self.args.data_path
        save_dir = self.args.save_path
        for idx, file in enumerate(os.listdir(data_dir)):
            start = time.time()
            test_img = Image.open(os.path.join(data_dir, file)).convert('RGB')
            test_array = np.array(test_img).astype(np.float32)
            image_id = file.split('.')[0]
            test_array /= 255.0
            test_array -= (0.485, 0.456, 0.406)
            test_array /= (0.229, 0.224, 0.225)
            width = test_array.shape[1]
            height = test_array.shape[0]
            inf_img = np.zeros((height, width), dtype=np.float32)

            for i in range(0, height, self.args.crop_size):
                if i + self.args.crop_size > height:
                    i = height - self.args.crop_size
                for j in range(0, width, self.args.crop_size):
                    if j + self.args.crop_size > width:
                        j = width - self.args.crop_size


                    test_crop_array = test_array[i:i + self.args.crop_size, j:j + self.args.crop_size]
                    test_crop_array = test_crop_array.transpose((2, 0, 1))
                    test_crop_array_batch = np.expand_dims(test_crop_array, axis=0)
                    test_crop_tensor = torch.from_numpy(test_crop_array_batch)
                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    test_crop_tensor_cuda = test_crop_tensor.to(device)

        # for i, sample in enumerate(self.test_loader):
        #     image, target = sample['image'], sample['label']
        #     inf_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        #     image = image.to(device)
                    with torch.no_grad():
                        output = self.model(test_crop_tensor_cuda)
                    pred = output.data.cpu().numpy()
                    pred = np.argmax(pred, axis=1)
                    inf_img[i:i+self.args.crop_size, j:j+self.args.crop_size] = pred[0][:, :]

            print('Test ...{}/{}'.format(idx+1, len(os.listdir(self.args.data_path))))
            ave = np.average(inf_img)
            save_label= np.where(inf_img > ave, 255, 0)
            save_img = Image.fromarray(save_label.astype('uint8'))
            save_img.save(os.path.join(self.args.save_path, image_id + '.png'))
            end_1 = time.time() - start
            print('测试时间：%.3f' % end_1)


def main():
    parser = argparse.ArgumentParser(description='Deeplabv3 testing')
    parser.add_argument('--backbone', type=str, default='mobilenet')
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--crop-size', type=int, default=513)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--data_path', type=str, default='./datasets/test/test_imgs')
    parser.add_argument('--save_path', type=str, default='./datasets/test/test_pred')
    parser.add_argument('--weights', type=str, default='./run/model_best.pth.tar')
    args = parser.parse_args()

    tester = Tester(args)

    print('Predicting...')
    tester.test()

if __name__ == '__main__':
    main()
