#import gflags
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torchvision as tv
import os
import glob
import gzip
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pdb
import pickle
import random
import sys
from skimage import io, transform
from scipy.misc import imresize
import socket
import sklearn
# from gflags import flagvalues
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from eval_metrics import ErrorRateAt95Recall
from tensorboard_logger import configure, log_value
# from matching_classifiers import calculate_dataset_auc, calculate_per_image_mean_auc
import matching_classifiers
# from patchdataset import load_image_matching_dataset, load_datasets

# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class ResNet(nn.Module):

#     def __init__(self, block, layers, opts, num_classes=1000):
#         self.name = None
#         self.opts = opts
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#         self.image_match_classifier = nn.Linear(1024 * block.expansion, 2)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     # def forward(self, im1, im2):
#     def forward(self, im1, im2, rmatches, matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, \
#       spatial_entropy_1_16x16, spatial_entropy_2_16x16, median_reproj_error, median_angle_error, \
#       r33, error_l2, error_l1, error_l2_hsv, error_l1_hsv, error_l2_lab, \
#       error_l1_lab):
#         x1 = self.conv1(im1)
#         x1 = self.bn1(x1)
#         x1 = self.relu(x1)
#         x1 = self.maxpool(x1)

#         x1 = self.layer1(x1)
#         x1 = self.layer2(x1)
#         x1 = self.layer3(x1)
#         x1 = self.layer4(x1)

#         x1 = self.avgpool(x1)
#         x1_ = x1.view(x1.size(0), -1)

#         x2 = self.conv1(im2)
#         x2 = self.bn1(x2)
#         x2 = self.relu(x2)
#         x2 = self.maxpool(x2)

#         x2 = self.layer1(x2)
#         x2 = self.layer2(x2)
#         x2 = self.layer3(x2)
#         x2 = self.layer4(x2)

#         x2 = self.avgpool(x2)
#         x2_ = x2.view(x2.size(0), -1)

#         # x = self.fc(x)

#         # return x
#         #     def forward(self, im1, im2):
#         # x1 = self.features(im1)
#         # x2 = self.features(im2)
        
#         # x1_ = x1.view(x1.size(0), -1)
#         # x2_ = x2.view(x2.size(0), -1)

#         x = torch.cat([x1_, x2_], 1)
#         x_ = self.image_match_classifier(x)
#         return x_


# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
#     return model


# def resnet34(pretrained=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
#     return model


# def resnet50(opts, pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], opts, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
#     return model


# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     model.name = 'ResNet101'
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
#     return model

# def vgg11_bn(opts, pretrained=True, **kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     else:
#         kwargs['init_weights'] = True
#     model = VGG(make_layers(cfg['A'], batch_norm=True), opts, **kwargs)
#     model.name = 'VGG11'
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict=False)
#     return model

# def vgg16_bn(opts, pretrained=True, **kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     else:
#         kwargs['init_weights'] = True
#     model = VGG(make_layers(cfg['D'], batch_norm=True), opts, **kwargs)
#     model.name = 'VGG16'
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']), strict=False)
#     return model

# def vgg19_bn(opts, pretrained=False, **kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     if pretrained:
#         kwargs['init_weights'] = False
#     else:
#         kwargs['init_weights'] = True
#     model = VGG(make_layers(cfg['E'], batch_norm=True), opts, **kwargs)
#     model.name = 'VGG19'
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
#     return model

class Logger(object):
  def __init__(self, run_dir):
    # clean previous logged data under the same directory name
    self._remove(run_dir)

    # configure the project
    configure(run_dir, flush_secs=2)

    self.global_step = 0

  def log_value(self, name, value):
    log_value(name, value, self.global_step)
    return self

  def step(self):
    self.global_step += 1

  @staticmethod
  def _remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
      os.remove(path)  # remove the file
    elif os.path.isdir(path):
      import shutil
      shutil.rmtree(path)  # remove dir and all contains


class NN(nn.Module):

    def __init__(self, opts):#, num_classes=1000, init_weights=True):
        super(NN, self).__init__()
        self.name = 'MLP'
        # self.features = features
        self.opts = opts
        
        # if self.opts['all_features']:
        #   self.image_match_classifier = nn.Sequential(
        #       nn.Linear(1024 * 7 * 7 + 15, 4096),
        #       nn.ReLU(True),
        #       nn.Dropout(),
        #       nn.Linear(4096, 4096),
        #       nn.ReLU(True),
        #       nn.Dropout(),
        #       nn.Linear(4096, 2)
        #       # nn.Linear(4096, 2),
        #       # nn.Softmax()
        #   )
        # else:

        if opts['use_image_features']:
            self.mlp = nn.Sequential(
                # nn.Linear(5015, 1024),
                # nn.Linear(1431, 1024),
                # nn.Linear(1943, 1024),
                nn.Linear(1436, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 2),
                nn.Softmax()
            )
        else:
            self.mlp = nn.Sequential(
                # nn.Linear(919, 2),
                nn.Linear(923, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 2),
                nn.Softmax()
            )

        


        # self.mlp = nn.Sequential(
        #     nn.Linear(1, 32),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(32, 2),
        #     # nn.ReLU(),
        #     # nn.Linear(2, 2),
        #     nn.Softmax()
        # )

        # self.linear_classifier = nn.Sequential(
        #   nn.Linear(4096, 2),
        #   nn.Softmax()
        #   )

        # if init_weights:
        self._initialize_weights()

        if opts['use_image_features']:
            # kwargs_ = {'num_classes': opts['num_classes']}
            __model = models.__dict__['resnet34'](pretrained=True)
            _feature_extraction_model = nn.Sequential(*list(__model.children())[:-1])
            for param in _feature_extraction_model.parameters():
                param.requires_grad = False
            _feature_extraction_model.cuda()
            self.image_feature_extractor = _feature_extraction_model
        
        

    def forward(self, arg):
        # x1 = self.features(im1)
        # x2 = self.features(im2)
        
        # x1_ = x1.view(x1.size(0), -1)
        # x2_ = x2.view(x2.size(0), -1)

        # if self.opts['all_features']:
        #   x = torch.cat((x1_, x2_), 1)

        R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
            num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
            pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
            vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
            sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
            labels, img1, img2 = arg

        x = torch.cat(( \
            R11s.type(torch.cuda.FloatTensor).view(R11s.size(0), -1), \
            R12s.type(torch.cuda.FloatTensor).view(R12s.size(0), -1), \
            R13s.type(torch.cuda.FloatTensor).view(R13s.size(0), -1), \
            R21s.type(torch.cuda.FloatTensor).view(R21s.size(0), -1), \
            R22s.type(torch.cuda.FloatTensor).view(R22s.size(0), -1), \
            R23s.type(torch.cuda.FloatTensor).view(R23s.size(0), -1), \
            R31s.type(torch.cuda.FloatTensor).view(R31s.size(0), -1), \
            R32s.type(torch.cuda.FloatTensor).view(R32s.size(0), -1), \
            R33s.type(torch.cuda.FloatTensor).view(R33s.size(0), -1), \

            num_rmatches.type(torch.cuda.FloatTensor).view(num_rmatches.size(0), -1), \
            num_matches.type(torch.cuda.FloatTensor).view(num_matches.size(0), -1), \

            spatial_entropy_1_8x8.type(torch.cuda.FloatTensor).view(spatial_entropy_1_8x8.size(0), -1), \
            spatial_entropy_2_8x8.type(torch.cuda.FloatTensor).view(spatial_entropy_2_8x8.size(0), -1), \
            spatial_entropy_1_16x16.type(torch.cuda.FloatTensor).view(spatial_entropy_1_16x16.size(0), -1), \
            spatial_entropy_2_16x16.type(torch.cuda.FloatTensor).view(spatial_entropy_2_16x16.size(0), -1), \
            
            pe_histogram.type(torch.cuda.FloatTensor).view(pe_histogram.size(0), -1), \
            pe_polygon_area_percentage.type(torch.cuda.FloatTensor).view(pe_polygon_area_percentage.size(0), -1), \

            nbvs_im1.type(torch.cuda.FloatTensor).view(nbvs_im1.size(0), -1), \
            nbvs_im2.type(torch.cuda.FloatTensor).view(nbvs_im2.size(0), -1), \

            te_histogram.type(torch.cuda.FloatTensor).view(te_histogram.size(0), -1), \
            
            ch_im1.type(torch.cuda.FloatTensor).view(ch_im1.size(0), -1), \
            ch_im2.type(torch.cuda.FloatTensor).view(ch_im2.size(0), -1), \

            vt_rank_percentage_im1_im2.type(torch.cuda.FloatTensor).view(vt_rank_percentage_im1_im2.size(0), -1), \
            vt_rank_percentage_im2_im1.type(torch.cuda.FloatTensor).view(vt_rank_percentage_im2_im1.size(0), -1), \

            sq_rank_scores_mean.type(torch.cuda.FloatTensor).view(sq_rank_scores_mean.size(0), -1), \
            sq_rank_scores_min.type(torch.cuda.FloatTensor).view(sq_rank_scores_min.size(0), -1), \
            sq_rank_scores_max.type(torch.cuda.FloatTensor).view(sq_rank_scores_max.size(0), -1), \
            sq_distance_scores.type(torch.cuda.FloatTensor).view(sq_distance_scores.size(0), -1), \
            ), 1)

        if self.opts['use_image_features']:
            
            i1 = self.image_feature_extractor(img1)
            i2 = self.image_feature_extractor(img2)
            # print '#'*100
            # for c in self.image_feature_extractor.modules():
            #     print c
            # print i1.size()

            # print i1.view(i1.size(0), -1).size()
            # print i2.view(-1, i2.size(0)).size()
            # i_dot = torch.bmm(i1.view(i1.size(0), -1), i2.view(-1, i2.size(0)))

            # print i1.view(i1.size(0), -1).size()
            # print i2.view(i2.size(0), -1).size()
            # print torch.t(i2.view(i2.size(0), -1)).size()
            i_dot = torch.diag(torch.matmul(i1.view(i1.size(0), -1), torch.t(i2.view(i2.size(0), -1)))).view(-1, 1)
            i_diff = i1.view(i1.size(0), -1) - i2.view(i2.size(0), -1)


            # print ('='*100)
            # print i_dot.size()
            # print i_dot.view(i2.size(0), 1).size()
            # i_diff = Variable(torch.zeros(i1.size(0), i1.size(1)).type(torch.cuda.FloatTensor))
            # print i_diff
            # print i_diff.size()
            # print '#'*100
            # print i_diff.size()
            # print i_dot.size()
            # print torch.dot(i1, i2, dims=0).size()
            # print '-'*100
            # y = torch.cat((i_diff, x), 1)
            # y = torch.cat((i1.view(i1.size(0), -1), i2.view(i2.size(0), -1), x), 1)
            y = torch.cat((i_diff, i_dot, x), 1)
            # print y.size()
            result = self.mlp(y)
        else:
            result = self.mlp(x)
            # print (x.data.cpu().numpy().tolist())
            # import sys; sys.exit(1)
            # x = num_rmatches.type(torch.cuda.FloatTensor).view(num_rmatches.size(0), -1)
                

              # z = torch.cat((x,y), 1)
              # result = self.image_match_classifier(z)
            # else:
            #   if self.opts['pretrained; dot-product']:
            #     # print '#'*100
            #     # print x1_
            #     # print '-'*100
            #     # print x2_

            #     # result = torch.dot(x1_, x2_)
            #     result = torch.diag(torch.matmul(x1_,torch.t(x2_)))
            #     # print '='*100
            #     # print result
            #     # sys.exit(1)
            #   else:
            #     z = torch.cat((x1_, x2_), 1)
            #     result = self.image_match_classifier(z)
                    
            # if False:
            #   result = Variable(torch.from_numpy(np.random.rand( len(rmatches) )).type(torch.cuda.FloatTensor))
            
        return result

    def _initialize_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if self.opts['use_small_weights']:
                    m.weight.data.normal_(0, 0.00000000001)
                else:
                    m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# def make_layers(cfg, batch_norm=False):
#   layers = []
#   in_channels = 3
#   for v in cfg:
#     if v == 'M':
#       layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#     else:
#       conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#       if batch_norm:
#         layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#       else:
#         layers += [conv2d, nn.ReLU(inplace=True)]
#       in_channels = v
#   return nn.Sequential(*layers)

class ImageMatchingDataset(data.Dataset):

    # def pil_loader(self, path):
    #     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    #     with open(path, 'rb') as f:
    #         img = Image.open(f)
    #         return img.convert('RGB')

    def __init__(self, arg, opts, transform=None, loader=tv.datasets.folder.default_loader):
        self.dsets, self.fns, self.R11s, self.R12s, self.R13s, self.R21s, self.R22s, self.R23s, self.R31s, self.R32s, self.R33s, \
            self.num_rmatches, self.num_matches, self.spatial_entropy_1_8x8, self.spatial_entropy_2_8x8, self.spatial_entropy_1_16x16, self.spatial_entropy_2_16x16, \
            self.pe_histogram, self.pe_polygon_area_percentage, self.nbvs_im1, self.nbvs_im2, self.te_histogram, self.ch_im1, self.ch_im2, \
            self.vt_rank_percentage_im1_im2, self.vt_rank_percentage_im2_im1, \
            self.sq_rank_scores_mean, self.sq_rank_scores_min, self.sq_rank_scores_max, self.sq_distance_scores, \
            self.labels, self.train, self.model, self.options = arg

        self.transform = transform
        self.loader = loader
        self.unique_fns_dsets = np.array([])
        self.unique_fns = np.array([])
        self.unique_imgs = {}

        # if self.train and self.options['triplet-sampling-strategy'] == 'random':
        #     self.sample_hierarchy = {}
        #     im_entry = {'positive_samples': [], 'negative_samples': []}

        #     # Separate out datasets and files into positive and negative samples
        #     for dset in list(set(self.dsets.tolist())):
        #         self.sample_hierarchy[dset] = {}
        #         ri = np.where(self.dsets == dset)[0]
                
        #         for i, (im1, im2) in enumerate(self.fns[ri,:]):
        #             for im in [im1, im2]:
        #                 if im not in self.sample_hierarchy[dset]:
        #                     self.sample_hierarchy[dset][im] = im_entry

        #                 if self.labels[ri[i]]:
        #                     self.sample_hierarchy[dset][im]['positive_samples'].append(ri[i])
        #                 else:
        #                     self.sample_hierarchy[dset][im]['negative_samples'].append(ri[i])

        #     self.positive_sample_indices = np.where(self.labels >= 1)[0]
        #     self.negative_sample_indices = np.where(self.labels <= 0)[0]

        # elif self.train and self.options['triplet-sampling-strategy'] == 'uniform-files':
        if self.train and self.options['triplet-sampling-strategy'] == 'random':
            self.positive_sample_indices = np.where(self.labels >= 1)[0]
            self.negative_sample_indices = np.where(self.labels <= 0)[0]
        elif self.train and self.options['triplet-sampling-strategy'] == 'uniform-files':

            self.sample_hierarchy = {}
            self.im_counter = {}
            
            # Separate out datasets and files into positive and negative samples
            for d, dset in enumerate(list(set(self.dsets.tolist()))):
                self.sample_hierarchy[dset] = {}
                ri = np.where(self.dsets == dset)[0].astype(np.int)
                _unique_fns = np.array(list(set(np.concatenate((self.fns[ri,0], self.fns[ri,1])).tolist())))
                excluded_fns = []
                for i,u in enumerate(_unique_fns):
                    # Make sure the file name has a positive and a negative sample
                    sanity_positive = len(np.where((self.fns[ri,0] == u) & (self.labels[ri] >= 1) | (self.fns[ri,1] == u) & (self.labels[ri] >= 1))[0])
                    sanity_negative = len(np.where((self.fns[ri,0] == u) & (self.labels[ri] <= 0) | (self.fns[ri,1] == u) & (self.labels[ri] <= 0))[0])
                    # if u == '000388.jpg':
                    #     print ('{} / {}'.format(sanity_negative, sanity_positive))
                    #     import sys; sys.exit(1)
                    if sanity_negative == 0 or sanity_positive == 0:
                        excluded_fns.append(i)
                # print _unique_fns
                _unique_fns = np.delete(_unique_fns, excluded_fns)
                # print _unique_fns
                _unique_fns_dsets = np.tile(dset, (len(_unique_fns),))

                if d == 0:
                    self.unique_fns = _unique_fns
                    self.unique_fns_dsets = _unique_fns_dsets
                else:
                    self.unique_fns = np.concatenate((self.unique_fns, _unique_fns))
                    self.unique_fns_dsets = np.concatenate((self.unique_fns_dsets, _unique_fns_dsets))

                # for im in _unique_fns:
                for i, (im1, im2) in enumerate(self.fns[ri,:]):

                    for im in [im1, im2]:
                        if im not in _unique_fns:
                            continue

                        im_key = '{}--{}'.format(dset, im)
                        if im_key not in self.im_counter:
                            self.im_counter[im_key] = 0
                        self.im_counter[im_key] = self.im_counter[im_key] + 1

                        if im not in self.sample_hierarchy[dset]:
                            im_entry = {'positive_samples': [], 'negative_samples': []}
                            self.sample_hierarchy[dset][im] = im_entry

                        if self.labels[ri[i]] >= 1:
                            self.sample_hierarchy[dset][im]['positive_samples'].append(ri[i])
                        else:
                            self.sample_hierarchy[dset][im]['negative_samples'].append(ri[i])

                        # if im == '000388.jpg':
                        #     print 'HERE'
                        #     print ('{} / {}'.format(im1, im2))
                        #     print self.sample_hierarchy[dset][im]['negative_samples']
                        #     # import sys; sys.exit(1)
        


        # if self.options['use_image_features']:
        #     for i,d in enumerate(self.dsets):
        #         for im in self.fns[i,:]:
        #             if d not in self.unique_imgs:
        #                 self.unique_imgs[d] = {}
        #             if im not in self.unique_imgs[d]:
        #                 img_fn = os.path.join(d, 'images', im)
        #                 self.unique_imgs[d][im] = self.transform(self.loader(img_fn))
        


        
            # print '#'*100
            # for dset in self.sample_hierarchy:
            #     for im in self.sample_hierarchy[dset]:
            #         if len(self.sample_hierarchy[dset][im]['negative_samples']) == 0:
            #             print (im)
            # print '#'*100
            # print excluded_fns
            # import sys; sys.exit(1)

            # ims_w_no_neg_samples = []
            # valid_p_samples = []
            # valid_n_samples = []
            # for dset in self.sample_hierarchy:
            #     for im in self.sample_hierarchy[dset]:
            #         if len(self.sample_hierarchy[dset][im]['negative_samples']) == 0:
            #             ims_w_no_neg_samples.append('{}--{}'.format(dset,im))
            #         else:
            #             self.sample_hierarchy[dset][im]['positive_samples'] = list(set(self.sample_hierarchy[dset][im]['positive_samples']))
            #             self.sample_hierarchy[dset][im]['negative_samples'] = list(set(self.sample_hierarchy[dset][im]['negative_samples']))
            #             valid_p_samples.extend(self.sample_hierarchy[dset][im]['positive_samples'])
            #             valid_n_samples.extend(self.sample_hierarchy[dset][im]['negative_samples'])
                    


            # # print ('Count: {} Images with no negative samples: {}'.format(len(ims_no_neg_samples), ims_no_neg_samples))
            # # sys.exit(1)
            # # self.positive_sample_indices = np.where(self.labels >= 1)[0]
            # # self.negative_sample_indices = np.where(self.labels <= 0)[0]
            # self.positive_sample_indices = np.array(valid_p_samples)
            # self.negative_sample_indices = np.array(valid_n_samples)

            # # for dset in self.sample_hierarchy:
            # #     for im in self.sample_hierarchy[dset]:
            # for x in ims_w_no_neg_samples:
            #     dset, im = x.split('--')
            #     self.sample_hierarchy[dset].pop(im)
            # for p in self.positive_sample_indices:
            #     if fns[p,0] in ims_no_neg_samples or fns[p,1] in ims_no_neg_samples:



            # import sys; sys.exit(1)
            # print (self.sample_hierarchy[dset]['000036.jpg']['positive_samples'])
            





            # self.sample_hierarchy = {}
            # im_entry = {'positive_samples': [], 'negative_samples': []}

            # # Separate out datasets and files into positive and negative samples
            # for dset in list(set(self.dsets.tolist())):
            #     self.sample_hierarchy[dset] = {}
            #     ri = np.where(self.dsets == dset)[0]
                
            #     for i, (im1, im2) in enumerate(self.fns[ri,:]):
            #         for im in [im1, im2]:
            #             if im not in self.sample_hierarchy[dset]:
            #                 self.sample_hierarchy[dset][im] = im_entry

            #             if self.labels[ri[i]]:
            #                 self.sample_hierarchy[dset][im]['positive_samples'].append(ri[i])
            #             else:
            #                 self.sample_hierarchy[dset][im]['negative_samples'].append(ri[i])

            # self.positive_sample_indices = np.where(self.labels >= 1)[0]
            # self.negative_sample_indices = np.where(self.labels <= 0)[0]
        else:
            self.positive_sample_indices = None
            self.negative_sample_indices = None

        # print ('Finished initializing datasets')


    def __getitem__(self, index):
        indices = []
        if self.train and self.options['triplet-sampling-strategy'] == 'random':
            # In train mode, index references positive samples only
            p_index = self.positive_sample_indices[index]
            n_index = self.negative_sample_indices[random.randint(0, len(self.negative_sample_indices) - 1)]
            indices = [p_index, n_index]

        # if self.train and self.options['triplet-sampling-strategy'] == 'random':
        #     # In train mode, index references positive samples only
        #     p_index = self.positive_sample_indices[index]
            
        #     im1_neg_samples = np.array(self.sample_hierarchy[self.dsets[p_index]][self.fns[p_index,0]]['negative_samples']).astype(np.int)
        #     # im2_neg_samples = np.array(self.sample_hierarchy[self.dsets[p_index]][self.fns[p_index,1]]['negative_samples']).astype(np.int)
        #     # relevant_negative_samples = np.concatenate((im1_neg_samples, im2_neg_samples)).tolist()
        #     relevant_negative_samples = im1_neg_samples.tolist()
        #     # print ('im1_neg_samples: {}'.format(im1_neg_samples))
        #     # print ('im2_neg_samples: {}'.format(im2_neg_samples))
        #     # print ('relevant_negative_samples: {}'.format(relevant_negative_samples))
        #     # if len(relevant_negative_samples) > 0:
        #     n_index = relevant_negative_samples[random.randint(0, len(relevant_negative_samples) - 1)]
        #     indices = [p_index, n_index]
        #     # else:
        #     #     # No valid negative sample
        #     #     data = [[], []]
        #     #     return data
        elif self.train and self.options['triplet-sampling-strategy'] == 'uniform-files':
            # In train mode, index references positive samples only
            _im = self.unique_fns[index]
            _dset = self.unique_fns_dsets[index]

            # im_pos_samples = np.array(self.sample_hierarchy[_dset][_im]['positive_samples'])
            # im_neg_samples = np.array(self.sample_hierarchy[_dset][_im]['negative_samples'])
            # # print ('#'*100)
            # # print ('#'*100)
            # # print ('** {} **'.format(_im))

            # # print (im_pos_samples)
            # for idx in im_pos_samples:
            #     im1, im2 = self.fns[idx,:]
            #     if _im == im1:
            #         _im2 = im2
            #     else:
            #         _im2 = im1

            #     im_key = '{}--{}'.format(_dset, _im2)
            #     pos_im_count = self.im_counter[im_key]

            # #     print ('\t{}  **  {}'.format(im1, im2))
            # #     print ('\t{}  --  {}: {} / {} - {}'.format(_dset, idx, _im, _im2, pos_im_count))
            # # import sys; sys.exit(1)
            # # print (self.sample_hierarchy[_dset][_fn])

            # # print ('{} / {}'.format(_dset, _fn))
            # # print (json.dumps(self.sample_hierarchy[_dset][_fn], sort_keys=True, indent=4, separators=(',', ': ')))
            # # import sys; sys.exit(1)



            
            # im1_neg_samples = np.array(self.sample_hierarchy[self.dsets[p_index]][self.fns[p_index,0]]['negative_samples'])
            # im2_neg_samples = np.array(self.sample_hierarchy[self.dsets[p_index]][self.fns[p_index,1]]['negative_samples'])
            # relevant_negative_samples = np.concatenate((im1_neg_samples, im2_neg_samples))
            # n_index = relevant_negative_samples[random.randint(0, len(relevant_negative_samples) - 1)]

            im_pos_samples = np.array(self.sample_hierarchy[_dset][_im]['positive_samples']).copy()
            im_neg_samples = np.array(self.sample_hierarchy[_dset][_im]['negative_samples']).copy()

            p_index = im_pos_samples[random.randint(0, len(im_pos_samples) - 1)]
            n_index = im_neg_samples[random.randint(0, len(im_neg_samples) - 1)]

            if self.options['sample-inclass'] and random.random() >= 0.5 and len(im_pos_samples) >= 2:
                # sample another positive
                # delete first sampled positive index
                im_pos_samples = np.delete(im_pos_samples, np.argwhere(im_pos_samples==p_index))
                p_index__ = im_pos_samples[random.randint(0, len(im_pos_samples) - 1)]
                if 15 + self.num_rmatches[p_index] >= self.num_rmatches[p_index__]:
                    # print ('Using positive samples')
                    indices = [p_index, p_index__]
                elif 15 + self.num_rmatches[p_index__] >= self.num_rmatches[p_index]:
                    # print ('Using positive samples')
                    indices = [p_index__, p_index]
                else:
                    indices = [p_index, n_index]

                # print ('#'*100)
                # print p_index
                # print im_pos_samples
                # print self.num_rmatches[im_pos_samples]
                # print self.labels[im_pos_samples]
                # print ('-'*100)
                # import sys; sys.exit(1)
            else:
                indices = [p_index, n_index]
        else:
            # In test mode, index references entire dataset
            indices = [index, index]


        data = []
        # print ('p_index: {} n_index: {} indices: {}'.format(p_index, n_index, indices))
        for i in indices:
            if self.options['use_image_features']:
                img1_fn = os.path.join(self.dsets[i], 'images', self.fns[i,0])
                img2_fn = os.path.join(self.dsets[i], 'images', self.fns[i,1])

                if self.dsets[i] not in self.unique_imgs:
                    self.unique_imgs[self.dsets[i]] = {}

                if self.fns[i,0] not in self.unique_imgs[self.dsets[i]]:
                    self.unique_imgs[self.dsets[i]][self.fns[i,0]] = self.transform(self.loader(img1_fn))

                if self.fns[i,1] not in self.unique_imgs[self.dsets[i]]:
                    self.unique_imgs[self.dsets[i]][self.fns[i,1]] = self.transform(self.loader(img2_fn))


                img1 = self.unique_imgs[self.dsets[i]][self.fns[i,0]]
                img2 = self.unique_imgs[self.dsets[i]][self.fns[i,1]]

                # img1 = self.transform(self.loader(img1_fn))
                # img2 = self.transform(self.loader(img2_fn))

                
                # self.unique_imgs[d][img2_fn] = self.transform(self.loader(img2_fn))

                # img1 = self.unique_imgs[self.dsets[i]][self.fns[i,0]]
                # img2 = self.unique_imgs[self.dsets[i]][self.fns[i,1]]
            else:
                img1 = np.zeros((3,256,256))
                img2 = np.zeros((3,256,256))

            data.append([self.dsets[i].tolist(), self.fns[i,0].tolist(), self.fns[i,1].tolist(), self.R11s[i], self.R12s[i], self.R13s[i], \
                self.R21s[i], self.R22s[i], self.R23s[i], self.R31s[i], self.R32s[i], self.R33s[i], \
                self.num_rmatches[i], self.num_matches[i], self.spatial_entropy_1_8x8[i], \
                self.spatial_entropy_2_8x8[i], self.spatial_entropy_1_16x16[i], self.spatial_entropy_2_16x16[i], \
                self.pe_histogram[i].reshape((-1,1)), self.pe_polygon_area_percentage[i], self.nbvs_im1[i], self.nbvs_im2[i], \
                self.te_histogram[i].reshape((-1,1)), self.ch_im1[i].reshape((-1,1)), self.ch_im2[i].reshape((-1,1)), \
                self.vt_rank_percentage_im1_im2[i], self.vt_rank_percentage_im2_im1[i], \
                self.sq_rank_scores_mean[i], self.sq_rank_scores_min[i], self.sq_rank_scores_max[i], self.sq_distance_scores[i], \
                self.labels[i], img1, img2])
            # else:
            #     data.append([self.dsets[i].tolist(), self.fns[i,0].tolist(), self.fns[i,1].tolist(), self.R11s[i], self.R12s[i], self.R13s[i], \
            #         self.R21s[i], self.R22s[i], self.R23s[i], self.R31s[i], self.R32s[i], self.R33s[i], \
            #         self.num_rmatches[i], self.num_matches[i], self.spatial_entropy_1_8x8[i], \
            #         self.spatial_entropy_2_8x8[i], self.spatial_entropy_1_16x16[i], self.spatial_entropy_2_16x16[i], \
            #         self.pe_histogram[i].reshape((-1,1)), self.pe_polygon_area_percentage[i], self.nbvs_im1[i], self.nbvs_im2[i], \
            #         self.te_histogram[i].reshape((-1,1)), self.ch_im1[i].reshape((-1,1)), self.ch_im2[i].reshape((-1,1)), \
            #         self.vt_rank_percentage_im1_im2[i], self.vt_rank_percentage_im2_im1[i], self.labels[i], np.zeros((3,256,256)), np.zeros((3,256,256))]) # last two entries are a placeholder

        return data


    # dataset_name = self.datasets[index][0].split('/')[-1]
    # im1_fn = os.path.join(self.opts['sfm_root'], dataset_name, 'images', self.file_names[index,0])
    # im2_fn = os.path.join(self.opts['sfm_root'], dataset_name, 'images', self.file_names[index,1])
    # if False:
    #   print '\t\tLoading images: \"{}\", \"{}\" - {}'.format(self.file_names[index,0], self.file_names[index,1], int(self.labels[index]))
    # 
    # img1 = self.loader(im1_fn)
    # img2 = self.loader(im2_fn)

    # print self.imgs.keys()
    # print key1
    # print key2

    # dataset_name = self.datasets[index].split('/')[-1]
    # key1 = dataset_name + '-' + self.file_names[index,0]
    # key2 = dataset_name + '-' + self.file_names[index,1]
    # img1 = self.imgs[key1]
    # img2 = self.imgs[key2]



    # if self.transform is not None:
    #   img1 = self.transform(img1)
    #   img2 = self.transform(img2)

    # return self.file_names[index,0], self.file_names[index,1], img1, img2, \
    #   float(self.rmatches[index]), \
    #   float(self.matches[index]), \
    #   float(self.spatial_entropy_1_8x8[index]), \
    #   float(self.spatial_entropy_2_8x8[index]), \
    #   float(self.spatial_entropy_1_16x16[index]), \
    #   float(self.spatial_entropy_2_16x16[index]), \
    #   float(self.median_reproj_error[index]), \
    #   float(self.median_angle_error[index]), \
    #   float(self.R33s[index]), \
    #   float(self.error_l2[index]), \
    #   float(self.error_l1[index]), \
    #   float(self.error_l2_hsv[index]), \
    #   float(self.error_l1_hsv[index]), \
    #   float(self.error_l2_lab[index]), \
    #   float(self.error_l1_lab[index]), \
    #   int(self.labels[index])

    def __len__(self):
        if self.train and self.options['triplet-sampling-strategy'] == 'random':
            return len(self.positive_sample_indices)
        elif self.train and self.options['triplet-sampling-strategy'] == 'uniform-files':
            return len(self.unique_fns)
        else:
            return len(self.fns)

def get_correct_counts(target, y_pred, verbose):
    try:
        # Score prediction when y_pred and target are on gpu
        result = np.sum(np.equal(target.data.cpu().numpy(), np.argmax(y_pred.data.cpu().numpy(),axis=1)))
    except:
        try:
            # Score prediction when y_pred and target are on cpu
            result = np.sum(np.equal(target, np.argmax(y_pred,axis=1)))
        except:
            # Baseline comparisons where prediction is only a 1d array
            result = np.sum(np.equal(target, y_pred))
    return result

def nn_accuracy(y, y_gt, all_im1s, all_im2s, color, ls, epoch, thresholds):
    np.save('results-{}.npy'.format(epoch), np.concatenate((y.reshape(-1,1),y_gt.reshape(-1,1)),axis=1) )
    np.savetxt('results-files-{}.out'.format(epoch), np.concatenate((all_im1s.reshape(-1,1),all_im2s.reshape(-1,1)),axis=1), delimiter=" ", fmt="%s")

    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_gt, y)

    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
    auc = sklearn.metrics.auc(recall, precision)
    return auc

def inference(data_loader, model, epoch, run_dir, logger, opts, mode=None, optimizer=None):
    # switch to train/eval mode
    print ('#'*100)
    if mode == 'train':
        model.train()
    else:
        model.eval()

    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Inlier/Outlier Precision-Recall Curve\n', fontsize=18)

    correct_counts = 0.0
    cum_loss = 0.0
    arrays_initialized = False

    for batch_idx, (p_sample, n_sample) in enumerate(data_loader):
        if len(p_sample) == 0 or len(n_sample) == 0:
            # not a valid triplet (no negative sample for anchor image)
            continue
        if mode == 'train':
            optimizer.zero_grad()

        for j, sample in enumerate([p_sample, n_sample]):
            dsets, im1s, im2s, R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
                num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
                pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
                vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
                sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
                labels, img1, img2 = sample

            _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, \
                _num_rmatches, _num_matches, _spatial_entropy_1_8x8, _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, \
                _pe_histogram, _pe_polygon_area_percentage, _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, \
                _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
                _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
                _labels, _img1, _img2 = \
                R11s.cuda(), R12s.cuda(), R13s.cuda(), R21s.cuda(), R22s.cuda(), R23s.cuda(), R31s.cuda(), R32s.cuda(), R33s.cuda(), \
                num_rmatches.cuda(), num_matches.cuda(), spatial_entropy_1_8x8.cuda(), spatial_entropy_2_8x8.cuda(), spatial_entropy_1_16x16.cuda(), spatial_entropy_2_16x16.cuda(), \
                pe_histogram.cuda(), pe_polygon_area_percentage.cuda(), nbvs_im1.cuda(), nbvs_im2.cuda(), te_histogram.cuda(), ch_im1.cuda(), ch_im2.cuda(), \
                vt_rank_percentage_im1_im2.cuda(), vt_rank_percentage_im2_im1.cuda(), \
                sq_rank_scores_mean.cuda(), sq_rank_scores_min.cuda(), sq_rank_scores_max.cuda(), sq_distance_scores.cuda(), \
                labels.cuda(), img1.cuda(), img2.cuda()

            arg = [Variable(_R11s), Variable(_R12s), Variable(_R13s), Variable(_R21s), Variable(_R22s), Variable(_R23s), Variable(_R31s), Variable(_R32s), Variable(_R33s), \
                Variable(_num_rmatches), Variable(_num_matches), Variable(_spatial_entropy_1_8x8), Variable(_spatial_entropy_2_8x8), Variable(_spatial_entropy_1_16x16), Variable(_spatial_entropy_2_16x16), \
                Variable(_pe_histogram), Variable(_pe_polygon_area_percentage), Variable(_nbvs_im1), Variable(_nbvs_im2), Variable(_te_histogram), Variable(_ch_im1), Variable(_ch_im2), \
                Variable(_vt_rank_percentage_im1_im2), Variable(_vt_rank_percentage_im2_im1), \
                Variable(_sq_rank_scores_mean), Variable(_sq_rank_scores_min), Variable(_sq_rank_scores_max), Variable(_sq_distance_scores), \
                Variable(_labels), Variable(_img1), Variable(_img2)]
            

            y_prediction = model(arg)
            target = Variable(_labels.type(torch.cuda.LongTensor))

            if j == 0:
                y_predictions = y_prediction
                positive_predictions = y_prediction
                positive_targets = target
                targets = target
                num_rmatches_b = Variable(_num_rmatches)
                fns_b = np.concatenate((np.array(im1s).reshape((-1,1)),np.array(im2s).reshape((-1,1))), axis=1)
                dsets_b = np.array(dsets)
            else:
                y_predictions = torch.cat((y_predictions, y_prediction))
                negative_predictions = y_prediction
                negative_targets = target
                targets = torch.cat((targets, target))
                num_rmatches_b = torch.cat((num_rmatches_b, Variable(_num_rmatches)))
                fns_b = np.concatenate((fns_b, np.concatenate((np.array(im1s).reshape((-1,1)),np.array(im2s).reshape((-1,1))), axis=1)), axis=0)
                dsets_b = np.concatenate((dsets_b, np.array(dsets)), axis=0)

        if mode == 'train':
            reformatted_targets = targets.type(torch.cuda.FloatTensor)
            reformatted_targets[reformatted_targets <= 0] = -1
            if opts['loss'] == 'cross-entropy':
                loss = cross_entropy_loss(y_predictions, targets)
            elif opts['loss'] == 'triplet':
                loss = margin_ranking_loss(y_predictions[:,1], y_predictions[:,0], reformatted_targets)
            elif opts['loss'] == 'cross-entropy+triplet':
                loss = 0.5*cross_entropy_loss(y_predictions, targets) + \
                    0.5*margin_ranking_loss(y_predictions[:,1], y_predictions[:,0], reformatted_targets)
            

            loss.backward()
            optimizer.step()

    # if opts['pretrained; dot-product']:
    #   # ignore accuracies
    #   correct_counts = 0
    # else:
        correct_counts = correct_counts + get_correct_counts(targets, y_predictions, False)

        if not arrays_initialized:
            all_dsets = dsets_b
            all_fns = fns_b
            all_targets = targets.data.cpu().numpy()
            # print y_pred
            # sys.exit(1)
            all_predictions = y_predictions.data.cpu().numpy()
            all_num_rmatches = num_rmatches_b.data.cpu().numpy()

            arrays_initialized = True
            # all_im1s = im1_fn
            # all_im2s = im2_fn
        else:
            all_dsets = np.concatenate((all_dsets, dsets_b), axis=0)
            # print (all_fns.shape)
            # print (fns.shape)
            all_fns = np.concatenate((all_fns, fns_b), axis=0)

            all_targets = np.concatenate((all_targets, targets.data.cpu().numpy()), axis=0)
            all_predictions = np.concatenate((all_predictions, y_predictions.data.cpu().numpy()), axis=0)
            all_num_rmatches = np.concatenate((all_num_rmatches, num_rmatches_b.data.cpu().numpy()), axis=0)
            # all_im1s = np.concatenate((all_im1s, im1_fn), axis=0)
            # all_im2s = np.concatenate((all_im2s, im2_fn), axis=0)

        if (batch_idx + 1) % opts['log_interval'] == 0:
            num_tests = (batch_idx + 1) * 2*opts['batch_size'] # positive and negative samples
            accuracy = correct_counts*100.0/num_tests
            if mode == 'train':
                print(
                    '{} Epoch: {} Accuracy: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        mode.upper(), epoch, accuracy, (batch_idx + 1) * opts['batch_size'], len(data_loader.dataset),
                        100.0 * (batch_idx + 1) * opts['batch_size'] / data_loader.dataset.__len__(),
                        loss.data[0]))
            else:
                print(
                    '{} Epoch: {} Accuracy: {} [{}/{} ({:.0f}%)]'.format(
                        mode.upper(), epoch, accuracy, (batch_idx + 1) * opts['batch_size'], len(data_loader.dataset),
                        100.0 * (batch_idx + 1) * opts['batch_size'] / data_loader.dataset.__len__()))

        if mode == 'train':
            cum_loss = cum_loss + loss.data[0]

  # num_tests = data_loader.dataset.__len__()
  # accuracy = correct_counts*100.0/num_tests
    if mode == 'train':
        # num_tests = data_loader.dataset.__len__()
        # correct_counts = get_correct_counts(all_targets, all_predictions, False)
        # accuracy = correct_counts*100.0/num_tests

        adjust_learning_rate(optimizer, opts)
        cum_loss = cum_loss/(num_tests/(1.0*opts['batch_size']))
        logger.log_value('TRAIN-LR', optimizer.param_groups[0]['lr'])
        logger.log_value('TRAIN-ACCURACY', accuracy)
        logger.log_value('TRAIN-LOSS', cum_loss)
        print ('{} Epoch: {}  Correct: {}  Accuracy: {}  Loss: {}'.format(mode.upper(), epoch, correct_counts, \
            round(accuracy, 2), round(cum_loss,2)))
        # do checkpointing
        if (epoch + 1) % 3 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               '{}/checkpoint_{}.pth'.format(run_dir, epoch))
    else:
        logger.log_value('TEST-ACCURACY', accuracy)
        print ('{} Epoch: {}  Correct: {}  Accuracy: {}\n'.format(mode.upper(), epoch, correct_counts, \
            round(accuracy, 2)))

    print ('-'*100)
    score_thresholds = np.linspace(0, 1.0, 21)
    rmatches_thresholds = np.linspace(18, 38, 21)
    for i,t in enumerate(score_thresholds):
        thresholded_scores_predictions = np.copy(all_predictions)
        thresholded_scores_predictions[all_predictions[:,1] >= t, 1] = 1
        thresholded_scores_predictions[all_predictions[:,1] >= t, 0] = 0
        thresholded_scores_correct_counts = get_correct_counts(all_targets, thresholded_scores_predictions, False)

        # print (all_num_rmatches)
        thresholded_rmatches_predictions = np.copy(all_num_rmatches)
        thresholded_rmatches_predictions[all_num_rmatches < rmatches_thresholds[i]] = 0
        thresholded_rmatches_predictions[all_num_rmatches >= rmatches_thresholds[i]] = 1
        thresholded_rmatches_correct_counts = get_correct_counts(all_targets, thresholded_rmatches_predictions, False)

        num_tests = 2 * len(data_loader.dataset) # positive and negative samples
        thresholded_scores_accuracy = thresholded_scores_correct_counts*100.0/num_tests
        thresholded_rmatches_accuracy = thresholded_rmatches_correct_counts*100.0/num_tests

        print ('\t{} Epoch: {}     Classifier - Correct: {}  Accuracy: {}  Threshold: {}     Baseline - Correct: {}  Accuracy: {}  Threshold: {}'.format(\
            mode.upper(), epoch, \
            thresholded_scores_correct_counts, round(thresholded_scores_accuracy, 2), t, \
            thresholded_rmatches_correct_counts, round(thresholded_rmatches_accuracy, 2), rmatches_thresholds[i])
        )


    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])

    # plt.clf()
    auc = matching_classifiers.calculate_dataset_auc(all_predictions[:,1], all_targets, color='green', ls='solid')
    _, _, _, _, auc_per_image_mean = matching_classifiers.calculate_per_image_mean_auc(all_dsets, all_fns, all_predictions[:,1], all_targets)
    _, _, _, _, mean_precision_per_image = matching_classifiers.calculate_per_image_precision_top_k(all_dsets, all_fns, all_predictions[:,1], all_targets)
    # plt.clf()
    auc_baseline = matching_classifiers.calculate_dataset_auc(all_num_rmatches, all_targets, color='green', ls='solid')
    _, _, _, _, auc_per_image_mean_baseline = matching_classifiers.calculate_per_image_mean_auc(all_dsets, all_fns, all_num_rmatches, all_targets)
    _, _, _, _, mean_precision_per_image_baseline = matching_classifiers.calculate_per_image_precision_top_k(all_dsets, all_fns, all_num_rmatches, all_targets)


    print ('\t{} Epoch: {}     Experiment: {} AUC: {} / {} / {}'.format(mode.upper(), epoch, opts['experiment'], \
        round(auc,3), round(auc_per_image_mean, 3), round(mean_precision_per_image, 3) \
        ))
    print ('\t{} Epoch: {}     Baseline: {} AUC: {} / {} / {}'.format(mode.upper(), epoch, 'Baseline', \
        round(auc_baseline, 3), round(auc_per_image_mean_baseline, 3), round(mean_precision_per_image_baseline, 3) \
        ))
    print ('='*100)
    if mode == 'train':
        logger.log_value('TRAIN-AUC-BASELINE', auc_baseline)
        logger.log_value('TRAIN-AUCPI-BASELINE', auc_per_image_mean_baseline)
        logger.log_value('TRAIN-PPI-BASELINE', mean_precision_per_image_baseline)
        logger.log_value('TRAIN-AUC-EXP', auc)
        logger.log_value('TRAIN-AUCPI-EXP', auc_per_image_mean)
        logger.log_value('TRAIN-PPI-EXP', mean_precision_per_image)
    else:
        logger.log_value('TEST-AUC-BASELINE', auc_baseline)
        logger.log_value('TEST-AUCPI-BASELINE', auc_per_image_mean_baseline)
        logger.log_value('TEST-PPI-BASELINE', mean_precision_per_image_baseline)
        logger.log_value('TEST-AUC-EXP', auc)
        logger.log_value('TEST-AUCPI-EXP', auc_per_image_mean)
        logger.log_value('TEST-PPI-EXP', mean_precision_per_image)
    # plt.legend(['{} : {} : {} / {}'.format(mode, opts['experiment'], auc, auc_per_image_mean)],  loc='lower left',  shadow=True, fontsize=20)
    # fig = plt.gcf()
    # fig.set_size_inches(18.5, 10.5)
    # # plt.savefig('nn-result-pr-{}-{}.png'.format(mode, epoch))
    # plt.clf()
    # plt.close()


# triplet_loss = nn.TripletMarginLoss(reduction='sum')
triplet_loss = nn.TripletMarginLoss()
margin_ranking_loss = nn.MarginRankingLoss(margin=0.15)
cross_entropy_loss = nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, opts):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = opts['lr'] / (1 + group['step'] * opts['lr_decay'])

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data, gain=math.sqrt(2.0))
        nn.init.constant(m.bias.data, 0.1)

def create_optimizer(model, opts):
    # setup optimizer
    if opts['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.mlp.parameters(), lr=opts['lr'],
                            momentum=0.9, dampening=0.9,
                            weight_decay=opts['wd'])
    elif opts['optimizer'] == 'adam':
        optimizer = optim.Adam(model.mlp.parameters(), lr=opts['lr'])
    
    else:
        raise Exception('Not supported optimizer: {0}'.format(opts['optimizer']))
    return optimizer

def classify_nn_image_match_inference(arg):
    model = arg[-2]
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg, opts, transform=None),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    run_dir = os.path.join(opts['nn_log_dir'], \
        'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-image-feats-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}'.format(\
            opts['optimizer'], \
            opts['batch_size'], \
            opts['lr'], \
            opts['experiment'], \
            opts['loss'], \
            opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            opts['image_match_classifier_min_match'], \
            opts['image_match_classifier_max_match'], \
            opts['use_all_training_data'], \
            opts['use_small_weights'], \
            model.name
        )
    )
    logger = Logger(run_dir)
    epoch = 0

    scores = inference(test_loader, model, epoch, run_dir, logger, opts, mode='test', optimizer=None)

    return None, None, None, scores, None

def classify_nn_image_match_initialization(train_loader, test_loader, run_dir, opts):
    # instantiate model and initialize weights
    kwargs = {}
    model = NN(opts, **kwargs)

    model.cuda()
    optimizer = create_optimizer(model, opts)

    # optionally resume from a checkpoint
    if opts['resume']:
        checkpoint_files = sorted(glob.glob(run_dir + '/*.pth'),key=os.path.getmtime)
        if len(checkpoint_files) > 0 and os.path.isfile(checkpoint_files[-1]):
              print('=> loading checkpoint {}'.format(checkpoint_files[-1]))
              checkpoint = torch.load(checkpoint_files[-1])
              opts['start_epoch'] = checkpoint['epoch']
              model.load_state_dict(checkpoint['state_dict'])
              optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('=> no checkpoint found')

    start = opts['start_epoch']
    end = start + opts['epochs']
  
    # create logger
    run_dir = os.path.join(opts['nn_log_dir'], \
        'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-image-feats-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}'.format(\
            opts['optimizer'], \
            opts['batch_size'], \
            opts['lr'], \
            opts['experiment'], \
            opts['loss'], \
            opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            opts['image_match_classifier_min_match'], \
            opts['image_match_classifier_max_match'], \
            opts['use_all_training_data'], \
            opts['use_small_weights'], \
            model.name
        )
    )
    
    # logger_train = Logger('{}-{}'.format(run_dir, 'train'))
    # logger_test = Logger('{}-{}'.format(run_dir, 'test'))

    logger = Logger(run_dir)
    
    for epoch in range(start, end):
        training_scores = inference(train_loader, model, epoch, run_dir, logger, opts, mode='train', optimizer=optimizer)
        if (epoch - start + 1) % 5 == 0:
            _ = inference(test_loader, model, epoch, run_dir, logger, opts, mode='test', optimizer=None)
        logger.step()

    # scores = inference(train_loader, model, epoch, run_dir, logger, opts, mode='train', optimizer=optimizer)
    return model, training_scores

def classify_nn_image_match_training(arg, arg_te):
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    run_dir = os.path.join(opts['nn_log_dir'], \
        'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-image-feats-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}'.format(\
            opts['optimizer'], \
            opts['batch_size'], \
            opts['lr'], \
            opts['experiment'], \
            opts['loss'], \
            opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            opts['image_match_classifier_min_match'], \
            opts['image_match_classifier_max_match'], \
            opts['use_all_training_data'], \
            opts['use_small_weights'], \
            'MLP'
        )
    )
    matching_classifiers.mkdir_p(run_dir)

    transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg, opts, transform=transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    print ('#'*50 + ' Training Data ' + '#'*50)
    print ('\tInliers: {}'.format(len(np.where(train_loader.dataset.labels == 1)[0])))
    print ('\tOutliers: {}'.format(len(np.where(train_loader.dataset.labels == 0)[0])))
    print ('\tTotal: {}'.format(train_loader.dataset.__len__()))
    print ('#'*100)

    opts = arg[-1]
    test_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg_te, opts, transform=transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    print ('#'*50 + ' Testing Data ' + '#'*50)
    print ('\tInliers: {}'.format(len(np.where(test_loader.dataset.labels == 1)[0])))
    print ('\tOutliers: {}'.format(len(np.where(test_loader.dataset.labels == 0)[0])))
    print ('\tTotal: {}'.format(test_loader.dataset.__len__()))
    print ('#'*110)


    model, training_scores = classify_nn_image_match_initialization(train_loader, test_loader, run_dir, opts)
    return None, None, model, training_scores, None
