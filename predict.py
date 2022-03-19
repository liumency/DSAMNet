# coding=utf-8
import os
import torch.utils.data
from torch.utils.data import DataLoader
from data_utils import TestDatasetFromFolder
from model.dsamnet import DSAMNet
import cv2
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Test Change Detection Models')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--model_dir', default='test.pth', type=str)
parser.add_argument('--time1_dir', default='../Data_path/time1/', type=str)
parser.add_argument('--time2_dir', default='../Data_path/time2/', type=str)
parser.add_argument('--label_dir', default='../Data_path/label/', type=str)
parser.add_argument('--save_dir', default='result/mydata/', type=str)

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGES_FORMAT = ['.jpg','.png','.tif']
image_sets = [name for name in os.listdir(opt.time1_dir) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

netCD = DSAMNet(2).to(device, dtype=torch.float)
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs!")
    netCD = torch.nn.DataParallel(netCD, device_ids=range(torch.cuda.device_count()))

netCD.load_state_dict(torch.load(opt.model_dir))

netCD.eval()

if __name__ == '__main__':
    test_set = TestDatasetFromFolder(opt.time1_dir, opt.time2_dir, opt.label_dir, image_sets)
    test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=1, shuffle=True)
    test_bar = tqdm(test_loader)
    inter = 0
    unin = 0

    for image1, image2, label, image_name in test_bar:

        image1 = image1.to(device, dtype=torch.float)
        image2 = image2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)

        prob, _, _  = netCD(image1,image2)

        prob = (prob > 1).float()
        prob = prob.cpu().data.numpy()
        result = np.squeeze(prob)

        cv2.imwrite(opt.save_dir + image_name[0], result*255)
