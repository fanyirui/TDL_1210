import os
import torch
import matplotlib.pyplot as plt
import sys
import argparse
from read_data import SynData
import torch.nn as nn
import torch.nn.functional as F
from network import TDL
#from lossfunction import L1andL1, DualNegativePSNR, NegativePsnrLoss
import logging
import time
import datetime
from torchvision import transforms

#from test import test, generate_pic, generate_multi_result
import numpy as np
from PIL import Image
import pytorch_ssim
# model_path = './checkpoints/ckpt_20200729_371epoch.pkl'

parser = argparse.ArgumentParser(description="NetWithNIQE Training")
parser.add_argument('-dataset', type=str, default='SynLight',
                    help='Trianing G:/fyr/video_rain_light/train/input/Dataset\n There are 3 options for Training \n SynLight, SynHeavy and NTU'
                    )
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=0.0001, help='init learning rate')
parser.add_argument('-weight_decay', type=float, default=0.0001, help='init learning rate')
parser.add_argument('-name', type=str, default='20200729')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log', 'log_%s.txt' % args.name))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('dataset: {}'.format(args.dataset))
logging.info('epoch: {}'.format(args.epoch))
logging.info('name: {}'.format(args.name))

gpuID=0
torch.cuda.set_device(gpuID)

#check_loss = 2.3
# model_path = 'F:/icme/Lz.pkl'
def main():
    train_data = SynData(dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    check_loss = 2
    model = TDL(cuda_flag=True, height=500, width=889, center=1)
    # model.load_state_dict(torch.load(model_path))
    #
    model.cuda()

    criterion1 = torch.nn.L1Loss()
    criterion2 = pytorch_ssim.SSIM(window_size = 11)
    #criterion2 = GradientLoss()

    optimizer = torch.optim.Adam(
        #filter(lambda p: p.requires_grad, model.parameters()),
        model.parameters(),
        lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    '''for param in model.parameters():
        print(param)'''
    plotx = []
    ploty = []
    print('Training Start')
    time_start = datetime.datetime.now()
    for epoch in range(args.epoch):

        losses = train(train_loader, model, criterion1,criterion2, optimizer, scheduler, epoch)

        #print('epoch: %d' % (epoch+1), 'loss: %f' % losses)
        logging.info('epoch: %d, loss: %f', epoch+1, losses)

        plotx.append(epoch + 1)
        ploty.append(losses)

        # if not os.path.exists('./checkpoints'):
        #     os.mkdir('./checkpoints')

        if (epoch % 10 == 0):
            torch.save(model.state_dict(), './checkpoints/ckpt_%s_%depoch.pkl' % (args.name, epoch + 1))

        if check_loss > losses:
            #print('\n%s Saving the best model temporarily...' % show_time(datetime.datetime.now()))
            # if not os.path.exists(os.path.join('./net_models')):
            #     os.mkdir(os.path.join('./net_models'))
            torch.save(model.state_dict(),
                       os.path.join('./net_models',  '_best_l_params.pkl'))
            print('Saved.\n')
            check_loss = losses



        plt.plot(plotx, ploty)
        plt.savefig('./loss_pic/' + args.name + '.jpg')
    print('Training End')
    time_end = datetime.datetime.now()
    logging.info('Time cost: {}'.format(time_end - time_start))


def train(train_loader, model, criterion1,criterion2,optimizer, scheduler, epoch):
    # model.train()
    losses = 0
    for step, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        #print(input[0,0,0,0,0])

        att1,att2,att,img1,img2,img,image= model(input)
        # img,image= model(input)
        #,att1,att2,att3
        #att = att.cuda()
        #prediction = prediction.cuda()
        #print(prediction.shape)

        loss = (1-criterion2(image, target))+criterion1(image, target)\
               +criterion1(att, target)+0.5*criterion1(img1, target)+0.5*criterion1(img2, target)

        if(epoch%10==0):
             save_images(image, './result/result_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(ims1, './result/in1_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(ims2, './result/in2_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(att1, './result/out1_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(att2, './result/out2_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(att, './result/att_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(img1, './result/r1_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(img2, './result/r2_%s_%s.jpg' % (str(epoch), str(step)))
        # save_images(img_COURSE, './result/rc_%s_%s.jpg' % (str(epoch), str(step)))


        #
        # if (epoch%100== 0):
        #     save_images(ims1, './result/in1_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(ims2, './result/in2_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(ims3, './result/in3_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(att1, './result/out1_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(att2, './result/out2_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(att3, './result/out3_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(att, './result/att_%s_%s.jpg' % (str(epoch), str(step)))
        #     save_images(img, './result/res_%s_%s.jpg' % (str(epoch), str(step)))







        optimizer.zero_grad()

        losses += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    #scheduler.step()

    losses = losses / (step + 1)

    return losses



def save_images(tensor, path):
  image_numpy = tensor[0].detach().cpu().float().numpy()
  image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
  im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
  im.save(path, 'png')


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.compute_image_gradient(output)
        target_gradient_x, target_gradient_y = self.compute_image_gradient(target)

        x_loss = torch.abs(output_gradient_x-target_gradient_x)
        y_loss = torch.abs(output_gradient_y-target_gradient_y)

        mut_loss = (x_loss + y_loss).mean()
        return mut_loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def compute_image_gradient_o(self, x):
        h_x = x.size()[2]
        w_x = x.size()[3]
        grad_x = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        grad_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        return grad_x, grad_y

    def compute_image_gradient(self, x):
        kernel_x = [[0, 0], [-1, 1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[0, -1], [0, 1]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

        grad_x_1 = torch.abs(F.conv2d(x[:, 0:1, :, :], weight_x, padding=1))
        grad_x_2 = torch.abs(F.conv2d(x[:, 1:2, :, :], weight_x, padding=1))
        grad_x_3 = torch.abs(F.conv2d(x[:, 2:3, :, :], weight_x, padding=1))
        grad_x = torch.cat([grad_x_1, grad_x_2, grad_x_3], 1)

        grad_y_1 = torch.abs(F.conv2d(x[:, 0:1, :, :], weight_y, padding=1))
        grad_y_2 = torch.abs(F.conv2d(x[:, 1:2, :, :], weight_y, padding=1))
        grad_y_3 = torch.abs(F.conv2d(x[:, 2:3, :, :], weight_y, padding=1))
        grad_y = torch.cat([grad_y_1, grad_y_2, grad_y_3], 1)

        return grad_x, grad_y

    def rgb_to_gray(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        return gray

if __name__ == '__main__':
    main()
