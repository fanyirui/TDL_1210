import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from skimage import io
import cv2

# video_rain_heavy light_train
class SynData(torch.utils.data.Dataset):
    def __init__(self, origin_img_root_dir = 'F:/dataset/video_rain_dataset/RainSynLight25/train/',  dataset='SynLight'):
        self.origin_img_root_dir = origin_img_root_dir
        self.dataset = dataset
        self.datasetname = 'light'
        self.pathlist = self.loadpath()
        self.count = len(self.pathlist)

    def loadpath(self):
        '''if self.dataset == 'SynLight':
            pathlistfile = 'trainlightfolder.txt'
            self.datasetname = 'light'
        elif self.dataset =='SynHeavy':
            pathlistfile = 'trainheavyfolder.txt'
            self.datasetname = 'heavy'

        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()'''

        if self.dataset == 'SynLight':
            pathlistfile = 'trainlightfolder.txt'
            self.datasetname = 'light'
            fp = open(pathlistfile)
            pathlist = fp.read().splitlines()
            fp.close()
        elif self.dataset == 'SynHeavy':
            pathlistfile = 'trainheavyfolder2.txt'
            self.datasetname = 'heavy'
            fp = open(pathlistfile)
            pathlist = fp.read().splitlines()
            fp.close()
        elif self.dataset == 'NTU':
            self.datasetname = 'train'
            pathlistfile = 'trainNTUfolder.txt'
            self.origin_img_root_dir = 'E:/program/Python/HyperParameter/datatrain/NTU_resize/'
            fp = open(pathlistfile)
            pathlist = fp.read().splitlines()
            fp.close()

        return pathlist

    def __getitem__(self, index):
        frames = []
        path_code = self.pathlist[index]

        N = 3
        for i in range(1, 4):
            # 把plt改为io
            frames.append(io.imread(
                os.path.join(self.origin_img_root_dir,  path_code, 'rfc-%d.png' % (i))))
        frames.append(io.imread(os.path.join(self.origin_img_root_dir, path_code, 'gtc-2.png')))


        # for i in range(1, 3):
        #     print(i)
        #     # 把plt改为io
        #     img = cv2.imread(os.path.join(self.origin_img_root_dir, path_code, 'rfc-%d.png' % (i)))
        #     out = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
        #                         cv2.NORM_MINMAX)  # Convert to normalized floating point
        #     frames.append(out)
        # img = cv2.imread(os.path.join(self.origin_img_root_dir, path_code, 'gtc-2.png'))
        # out = cv2.normalize(img.astype('float'), None, 0.0, 1.0,
        #                     cv2.NORM_MINMAX)  # Convert to normalized floating point
        # frames.append(out)


        frames = np.array(frames) / 255.0


        framex = np.transpose(frames[0:N, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[-1, :, :, :], (2, 0, 1))


        return torch.from_numpy(framex).type(torch.FloatTensor), torch.from_numpy(framey).type(
            torch.FloatTensor)

    def __len__(self):
        return self.count
