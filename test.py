import torch
from torchvision import transforms
import numpy as np
import sys
import getopt
import os
import shutil
import matplotlib.pyplot as plt
import datetime
from PIL import Image
from network import TDL
import warnings

warnings.filterwarnings("ignore", module="matplotlib.pyplot")
# ------------------------------
# I don't know whether you have a GPU.
plt.switch_backend('agg')
# Static

dataset_dir = 'E:/data/ntu_test/real/'
pathlistfile = 'E:/data/ntu_test/real/test.txt'

model_path = 'F:/Lz.pkl'



gpuID = 0
if gpuID == None:
    cuda_flag = False
else:
    cuda_flag = True
    torch.cuda.set_device(gpuID)


# --------------------------------------------------------------

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def vimeo_evaluate(input_dir, out_img_dir, test_codelistfile, cuda_flag=True):
    mkdir_if_not_exist(out_img_dir)

    net = TDL()
    net.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    if cuda_flag:
        net.cuda().eval()
    else:
        net.eval()

    fp = open(test_codelistfile)
    test_img_list = fp.read().splitlines()
    fp.close()

    count = 0
    for k in range(2,59):
        process_index = [k-1,k,k+1]
        str_format = '%05d.png'

        total_count = len(test_img_list)


        pre = datetime.datetime.now()


        for code in test_img_list:
            print('Processing %s...' % code)
            count += 1
            video = code.split('/')[0]

            #mkdir_if_not_exist(os.path.join(out_img_dir, video))

            input_frames = []
            # input_frames = input_frames.unsqueeze(0)
            for i in process_index:
                input_frames.append(
                    plt.imread(os.path.join(input_dir, code, str_format % i).replace('\\', '/')))  # 添加到现有列表里


            input_frames = np.transpose(np.array(input_frames), (0, 3, 1, 2))  # 转置

            if cuda_flag:
                input_frames = torch.from_numpy(input_frames).cuda()
            else:
                input_frames = torch.from_numpy(input_frames)

            input_frames = input_frames.view(1, input_frames.size(0), input_frames.size(1), input_frames.size(2),
                                             input_frames.size(3))  # 一行
            #B, N, C, H, W = input_frames.size()
            #print(input_frames.shape)
            with torch.no_grad():
                img_COURSE= net(input_frames)
            # ims1,a1,img1= net(input_frames)



            '''
            ima1=input_frames[:, 0, :, :, :].squeeze(1)
            ima1 = transforms.ToPILImage()(ima1[0, :, :, :].cpu().data)'''


            # save_images(ims1, './test_result/in-%s.jpg' % (str(video)))
            # save_images(a1, './test_result/out-%s.jpg' % (str(video)))
            # save_images(img1, './test_result/result-%s.jpg' % (str(video)))


            #img.save(os.path.join(out_img_dir, '%d-out_rain25_atNRT100.png'))


            save_images(img_COURSE, 'E:/data/r/%s/result-%d.jpg' % (str(video),k))

        cur = datetime.datetime.now()
        processing_time = (cur - pre).seconds / count
        print('%.2fs per frame.\t%.2fs left.' % (processing_time, processing_time * (total_count - count)))



def save_images(tensor, path):
  image_numpy = tensor[0].detach().cpu().float().numpy()
  image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
  im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
  im.save(path, 'png')


vimeo_evaluate(dataset_dir, './test_result', pathlistfile, cuda_flag=cuda_flag)