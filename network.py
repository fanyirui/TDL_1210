import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

# import torch.utils.serialization   # it was removed in torch v1.0.0 or higher version.


arguments_strModel = 'sintel-final'
SpyNet_model_dir = './models'  # The directory of SpyNet's weights


def normalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
    tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
    tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


def denormalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
    tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
    tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


Backward_tensorGrid = {}


# warp
def Backward(tensorInput, tensorFlow, cuda_flag):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        if cuda_flag:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1).cuda()
        else:
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], 1)
    # end
    # 缩放到 [-1, 1] 函数grid_sample的grid参数需要接收[-1, 1]范围的坐标
    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensorInput,
                                           grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2,
                                                                                                                   3,
                                                                                                                   1),
                                           mode='bilinear', padding_mode='border')


# end

class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                # 输入通道数为8，两张RGB图像及一个2通道光流图
                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            # end

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        # 构建由4个moduleBasic组成的网络
        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(4)])

        self.load_state_dict(torch.load(SpyNet_model_dir + '/network-' + arguments_strModel + '.pytorch'), strict=False)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]

        # tenosrFirst 和 Second 是存储 1，1/2，1/4，1/8尺寸RGB图片的列表
        for intLevel in range(3):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        # 初始化光流图
        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear',
                                                              align_corners=True) * 2.0

            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            # input ：[first picture of corresponding level,
            # 		   the output of w with input second picture of corresponding level and upsampling flow,
            # 		   upsampling flow]
            # then we obtain the final flow.
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               Backward(tensorInput=tensorSecond[intLevel],
                                                                        tensorFlow=tensorUpsampled,
                                                                        cuda_flag=self.cuda_flag),
                                                               tensorUpsampled], 1)) + tensorUpsampled
        return tensorFlow


class warp(torch.nn.Module):
    def __init__(self, h, w, cuda_flag=True):
        super(warp, self).__init__()
        self.height = h
        self.width = w
        self.cuda_flag = cuda_flag

    def init_addterm(self):
        n = torch.FloatTensor(list(range(self.width)))
        horizontal_term = n.expand((1, 1, self.height, self.width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(self.height)))
        vertical_term = n.expand((1, 1, self.width, self.height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        self.height = flow.size(2)
        self.width = flow.size(3)
        # print('height: ', self.height)

        if self.cuda_flag:
            self.addterm = self.init_addterm().cuda()
        else:
            self.addterm = self.init_addterm()

        if True:
            flow = flow + self.addterm
        else:
            self.addterm = self.init_addterm()
            flow = flow + self.addterm

        horizontal_flow = flow[0, 0, :, :].expand(1, 1, self.height, self.width)  # 第一个0是batch size
        vertical_flow = flow[0, 1, :, :].expand(1, 1, self.height, self.width)

        horizontal_flow = horizontal_flow * 2 / (self.width - 1) - 1
        vertical_flow = vertical_flow * 2 / (self.height - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)

        return reference_frame


# SpyNet + Warp
class Alignment(nn.Module):
    def __init__(self, cuda_flag=True, height=500, width=889):
        super(Alignment, self).__init__()

        self.cuda_flag = cuda_flag
        self.height = height
        self.width = width

        self.spynet = SpyNet(cuda_flag=self.cuda_flag)
        self.warp = warp(self.height, self.width, cuda_flag=self.cuda_flag)

    def forward(self, rainframes):

        for i in range(rainframes.size(1)):
            rainframes[:, i, :, :, :] = normalize(rainframes[:, i, :, :, :])

        if self.cuda_flag:
            opticalflows = torch.zeros(rainframes.size(0), rainframes.size(1), 2, rainframes.size(3),
                                       rainframes.size(4)).cuda()
            warpframes = torch.empty(rainframes.size(0), rainframes.size(1), 3, rainframes.size(3),
                                     rainframes.size(4)).cuda()
        else:
            opticalflows = torch.zeros(rainframes.size(0), rainframes.size(1), 2, rainframes.size(3),
                                       rainframes.size(4))
            warpframes = torch.empty(rainframes.size(0), rainframes.size(1), 3, rainframes.size(3), rainframes.size(4))

        process_index = [0, 2]
        # process_index = [0, 1, 2, 4, 5, 6]
        for i in process_index:
            opticalflows[:, i, :, :, :] = self.spynet(rainframes[:, 1, :, :, :], rainframes[:, i, :, :, :])
        warpframes[:, 1, :, :, :] = rainframes[:, 1, :, :, :]

        for i in process_index:
            warpframes[:, i, :, :, :] = self.warp(rainframes[:, i, :, :, :], opticalflows[:, i, :, :, :])

        '''for i in range(warpframes.size(1)):
            warpframes[:, i, :, :, :] = denormalize(warpframes[:, i, :, :, :])'''
        return warpframes
        # return warpframes[:, 0, :, :, :], warpframes[:, 1, :, :, :], warpframes[:, 3, :, :, :], warpframes[:, 4, :, :, :]


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # feature extraction
        self.fe1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.fe2 = torch.nn.Conv2d(64, 64, 3, 1, 1)

        # attention
        self.sAtt_1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = torch.nn.Conv2d(64 * 2, 64, 1, 1, bias=True)
        self.sAtt_3 = torch.nn.Conv2d(64, 3, 1, 1, bias=True)
        #self.sAtt_3 = torch.nn.Conv2d(64, 1, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, frame):
        B, C, H, W = frame.size()  # N video frames

        ### feature extraction
        frame_fea = frame.clone().view(-1, C, H, W)
        frame_fea = self.lrelu(self.fe1(frame_fea))
        frame_fea = self.lrelu(self.fe2(frame_fea)).view(B,-1, H, W)

        cor_l = []
        # spatial attention
        att = self.lrelu(self.sAtt_1(frame_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att = F.interpolate(att, size=[H, W],mode='bilinear', align_corners=False)#上采样
        att = self.sAtt_3(att).unsqueeze(1)

        cor_l.append(att)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, C, H, W

        return  cor_prob*frame




class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)

        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(64, 3, 3, 1, 1, bias=True)


        self.sig = nn.Sigmoid()


    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)

        out = F.relu(self.conv3(out), inplace=True)
        out = self.conv4(out)

        x = x[:, :3, :, :]
        identity = out + x
        return identity

class Soft(nn.Module):
    def __init__(self):
        super(Soft, self).__init__()

        self.conv1 = nn.Conv2d(6, 64, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(64, 2, 3, 1, 1, bias=True)


    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        X_exp = out.exp()
        # print(x.shape)
        partition = X_exp.sum(dim=1, keepdim=True)

        return X_exp / partition  # 这里应用了广播机制

class TDL(nn.Module):
    def __init__(self, cuda_flag=True, height=500, width=889, center=1):
        super(TDL, self).__init__()
        self.cuda_flag = cuda_flag
        self.height = height
        self.width = width
        self.center = center

        self.alignment = Alignment(cuda_flag=self.cuda_flag, height=self.height, width=self.width)

        self.att_1 = Attention()
        self.att_2 = Attention()
        self.res =Residual()
        self.soft=Soft()


    def forward(self, rainframes):
        #print(rainframes)
        B, N, C, H, W = rainframes.size()

        imm2 = rainframes[:, 1, :, :, :].clone().squeeze(1)

        warpframes = self.alignment(rainframes)
        #warpframes = rainframes

        imy1 = warpframes[:, 0, :, :, :].clone().squeeze(1)
        imy2 = warpframes[:, 1, :, :, :].clone().squeeze(1)
        imy3 = warpframes[:, 2, :, :, :].clone().squeeze(1)



        ims1 = imy2 - imy1 # BF
        ims1[ims1 < 0] = 0
        ims2 = imy2 - imy3 # BD
        ims2[ims2 < 0] = 0


        so= torch.cat((ims1, ims2), dim=1)



        att1 = self.att_1(ims1)  # B, N, C, H, W
        att2 = self.att_2(ims2)  # B, N, C, H, W


        sod=self.soft(so)
        img1=imm2-att1#
        img2=imm2-att2




        img_COURSE = imm2-(att1*sod[:, 0, :, :] + att2 *sod[:, 1, :, :])

        img_COURSE = img_COURSE.squeeze(1)

        att = (att1*sod[:, 0, :, :] + att2 *sod[:, 1, :, :])  #总的attention
        att = att.squeeze(1)

        imgz=imm2-att
        ims = torch.cat((att, imm2), dim=1)

        img = self.res(ims)  # , 3, H, W

        image=imm2-img




        img1 = img1.squeeze(1)
        img2 = img2.squeeze(1)
        imgz=imgz.squeeze(1)


        ims1 = ims1.squeeze(1)
        ims2 = ims2.squeeze(1)
        ims = ims.squeeze(1)


        att1 = att1.squeeze(1)
        att2 = att2.squeeze(1)
        att = att.squeeze(1)

        ims = ims.squeeze(1)
        img = img.squeeze(1)
        image = image.squeeze(1)


        imy1 =denormalize(imy1)
        imy2= denormalize(imy2)
        imy3= denormalize(imy3)



        return  att1,att2,att,img1,img2,img,image

