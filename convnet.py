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
import shutil
import socket
import sklearn
# from gflags import flagvalues
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from eval_metrics import ErrorRateAt95Recall
from tensorboard_logger import configure, log_value, log_images
import matching_classifiers
import resnet

class Logger(object):
    def __init__(self, run_dir):
        # configure the project
        configure(run_dir, flush_secs=2)
        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)
        return self

    def log_images(self, name, images):
        log_images(name, images, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
          os.remove(path)  # remove the file
        elif os.path.isdir(path):
          shutil.rmtree(path)  # remove dir and all contains

class ConvNet(nn.Module):

    def __init__(self, opts):#, num_classes=1000, init_weights=True):
        super(ConvNet, self).__init__()
        self.name = 'CONVNET'
        self.opts = opts

        if opts['model'] == 'resnet18':
            mlp_input_size = 512
            __model = resnet.resnet18(pretrained=False, opts=opts)
        elif opts['model'] == 'resnet34':
            mlp_input_size = 512
            __model = resnet.resnet34(pretrained=False, opts=opts)
        elif opts['model'] == 'resnet50':
            mlp_input_size = 4096
            __model = resnet.resnet50(pretrained=False, opts=opts)
        elif opts['model'] == 'resnet101':
            mlp_input_size = 4096
            __model = resnet.resnet101(pretrained=False, opts=opts)
        elif opts['model'] == 'resnet152':
            mlp_input_size = 4096
            __model = resnet.resnet152(pretrained=False, opts=opts)
        else:
            __model = resnet.resnet50(pretrained=False, opts=opts)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 2)
        )

        _feature_extraction_model = nn.Sequential(*list(__model.children())[:-1])
        _feature_extraction_model.cuda()
        self.image_feature_extractor = _feature_extraction_model
        self._initialize_weights()
        
    def forward(self, arg):
        R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
            num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
            pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
            vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
            sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
            shortest_path_length, \
            mds_rank_percentage_im1_im2, mds_rank_percentage_im2_im1, \
            distance_rank_percentage_im1_im2_gt, distance_rank_percentage_im2_im1_gt, \
            labels, img1, img2, se_rmatches_img1, se_rmatches_img2, se_matches_img1, se_matches_img2, pe_img1, pe_img2, \
            pe_mask_img1, pe_mask_img2, pe_warped_img1, pe_warped_img2, se_fm1, se_fm2, tm_fm1, tm_fm2, se_non_rmatches_img1, se_non_rmatches_img2, \
            se_rmatches_secondary_motion_img1, se_rmatches_secondary_motion_img2 = arg

        input1 = None
        input2 = None
        
        if self.opts['convnet_use_matches_map']:
            input1 = torch.cat((input1, se_matches_img1), 1) if input1 is not None else se_matches_img1
            input2 = torch.cat((input2, se_matches_img2), 1) if input2 is not None else se_matches_img2

        if self.opts['convnet_use_rmatches_map']:
            input1 = torch.cat((input1, se_rmatches_img1), 1) if input1 is not None else se_rmatches_img1
            input2 = torch.cat((input2, se_rmatches_img2), 1) if input2 is not None else se_rmatches_img2

        if self.opts['convnet_use_photometric_error_maps']:
            input1 = torch.cat((input1, pe_img1, pe_mask_img1), 1) if input1 is not None else torch.cat((pe_img1, pe_mask_img1), 1)
            input2 = torch.cat((input2, pe_img2, pe_mask_img2), 1) if input2 is not None else torch.cat((pe_img2, pe_mask_img2), 1)

        if self.opts['convnet_use_non_rmatches_map']:
            input1 = torch.cat((input1, se_non_rmatches_img1), 1) if input1 is not None else se_non_rmatches_img1
            input2 = torch.cat((input2, se_non_rmatches_img2), 1) if input2 is not None else se_non_rmatches_img2

        if self.opts['convnet_use_rmatches_secondary_motion_map']:
            input1 = torch.cat((input1, se_rmatches_secondary_motion_img1), 1) if input1 is not None else se_rmatches_secondary_motion_img1
            input2 = torch.cat((input2, se_rmatches_secondary_motion_img2), 1) if input2 is not None else se_rmatches_secondary_motion_img2

        if self.opts['convnet_use_images']:
            input1 = torch.cat((input1, img1), 1) if input1 is not None else img1
            input2 = torch.cat((input2, img2), 1) if input2 is not None else img2

        if self.opts['convnet_use_feature_match_map']:
            input1 = torch.cat((input1, se_fm1), 1) if input1 is not None else se_fm1
            input2 = torch.cat((input2, se_fm2), 1) if input2 is not None else se_fm2

        if self.opts['convnet_use_track_map']:
            input1 = torch.cat((input1, tm_fm1), 1) if input1 is not None else tm_fm1
            input2 = torch.cat((input2, tm_fm2), 1) if input2 is not None else tm_fm2

        i1 = self.image_feature_extractor(input1)
        i2 = self.image_feature_extractor(input2)



        # i1 = se_rmatches_img1
        # i2 = se_rmatches_img2
        # i_dot = torch.diag(torch.matmul(i1.view(i1.size(0), -1), torch.t(i2.view(i2.size(0), -1)))).view(-1, 1)
        # i_diff = i1.view(i1.size(0), -1) - i2.view(i2.size(0), -1)

        # y = torch.cat((i_diff, i_dot, x), 1)
        # y = torch.cat((i_diff, i_dot), 1)
        # y = i_diff


        y1 = i1.view(i1.size(0), -1)
        y2 = i2.view(i2.size(0), -1)
        # y = torch.cat((i1.view(i2.size(0), -1), i2.view(i2.size(0), -1)), 1)

        # if 'TE' in self.opts['features']:
        #     y = torch.cat((y.view(y.size(0), -1), te_histogram.view(te_histogram.size(0), -1)), 1)
        # if 'NBVS' in self.opts['features']:
        #     y = torch.cat((y.view(y.size(0), -1), nbvs_im1.view(nbvs_im1.size(0), -1), nbvs_im2.view(nbvs_im2.size(0), -1)), 1)

        # print '='*100
        # # print img1.size()
        # # print se_rmatches_img1.size()
        # # print se_matches_img1.size()
        # # print pe_img1.size()
        # # print pe_mask_img1.size()
        # print '@'*100
        # print input1.size()
        # print input2.size()
        # print '='*100
        # print y.size()
        # import sys; sys.exit(1)

        # print 'y: {} - {}'.format(y1.size(), y2.size())
        result1 = self.mlp(y1)

        result2 = self.mlp(y2)
        # print 'result2: {} - {}'.format(result1.size(), result2.size())
        result = torch.div(torch.add(result1, result2),2.0)
        # print 'result: {}'.format(result.size())
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        for m in self.image_feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ImageMatchingDataset(data.Dataset):
    def __init__(self, arg, opts, transform=None, loader=tv.datasets.folder.default_loader):
        self.dsets, self.fns, self.R11s, self.R12s, self.R13s, self.R21s, self.R22s, self.R23s, self.R31s, self.R32s, self.R33s, \
            self.num_rmatches, self.num_matches, self.spatial_entropy_1_8x8, self.spatial_entropy_2_8x8, self.spatial_entropy_1_16x16, self.spatial_entropy_2_16x16, \
            self.pe_histogram, self.pe_polygon_area_percentage, self.nbvs_im1, self.nbvs_im2, self.te_histogram, self.ch_im1, self.ch_im2, \
            self.vt_rank_percentage_im1_im2, self.vt_rank_percentage_im2_im1, \
            self.sq_rank_scores_mean, self.sq_rank_scores_min, self.sq_rank_scores_max, self.sq_distance_scores, \
            self.lcc_im1_15, self.lcc_im2_15, self.min_lcc_15, self.max_lcc_15, \
            self.lcc_im1_20, self.lcc_im2_20, self.min_lcc_20, self.max_lcc_20, \
            self.lcc_im1_25, self.lcc_im2_25, self.min_lcc_25, self.max_lcc_25, \
            self.lcc_im1_30, self.lcc_im2_30, self.min_lcc_30, self.max_lcc_30, \
            self.lcc_im1_35, self.lcc_im2_35, self.min_lcc_35, self.max_lcc_35, \
            self.lcc_im1_40, self.lcc_im2_40, self.min_lcc_40, self.max_lcc_40, \
            self.shortest_path_length, \
            self.mds_rank_percentage_im1_im2, self.mds_rank_percentage_im2_im1, \
            self.distance_rank_percentage_im1_im2_gt, self.distance_rank_percentage_im2_im1_gt, \
            self.labels, self.weights, self.train, self.model, self.options = arg

        ri = np.where((self.num_rmatches >= self.options['range_min']) & (self.num_rmatches <= self.options['range_max']))[0]
        self.dsets = self.dsets[ri]
        self.fns = self.fns[ri]
        self.R11s = self.R11s[ri]
        self.R12s = self.R12s[ri]
        self.R13s = self.R13s[ri]
        self.R21s = self.R21s[ri]
        self.R22s = self.R22s[ri]
        self.R23s = self.R23s[ri]
        self.R31s = self.R31s[ri]
        self.R32s = self.R32s[ri]
        self.R33s = self.R33s[ri]
        self.num_rmatches = self.num_rmatches[ri]
        self.num_matches = self.num_matches[ri]
        self.spatial_entropy_1_8x8 = self.spatial_entropy_1_8x8[ri]
        self.spatial_entropy_2_8x8 = self.spatial_entropy_2_8x8[ri]
        self.spatial_entropy_1_16x16 = self.spatial_entropy_1_16x16[ri]
        self.spatial_entropy_2_16x16 = self.spatial_entropy_2_16x16[ri]
        self.pe_histogram = self.pe_histogram[ri]
        self.pe_polygon_area_percentage = self.pe_polygon_area_percentage[ri]
        self.nbvs_im1 = self.nbvs_im1[ri]
        self.nbvs_im2 = self.nbvs_im2[ri]
        self.te_histogram = self.te_histogram[ri]
        self.ch_im1 = self.ch_im1[ri]
        self.ch_im2 = self.ch_im2[ri]
        self.vt_rank_percentage_im1_im2 = self.vt_rank_percentage_im1_im2[ri]
        self.vt_rank_percentage_im2_im1 = self.vt_rank_percentage_im2_im1[ri]
        self.sq_rank_scores_mean = self.sq_rank_scores_mean[ri]
        self.sq_rank_scores_min = self.sq_rank_scores_min[ri]
        self.sq_rank_scores_max = self.sq_rank_scores_max[ri]
        self.sq_distance_scores = self.sq_distance_scores[ri]
        self.lcc_im1_15 = self.lcc_im1_15[ri]
        self.lcc_im2_15 = self.lcc_im2_15[ri]
        self.min_lcc_15 = self.min_lcc_15[ri]
        self.max_lcc_15 = self.max_lcc_15[ri]
        self.lcc_im1_20 = self.lcc_im1_20[ri]
        self.lcc_im2_20 = self.lcc_im2_20[ri]
        self.min_lcc_20 = self.min_lcc_20[ri]
        self.max_lcc_20 = self.max_lcc_20[ri]
        self.lcc_im1_25 = self.lcc_im1_25[ri]
        self.lcc_im2_25 = self.lcc_im2_25[ri]
        self.min_lcc_25 = self.min_lcc_25[ri]
        self.max_lcc_25 = self.max_lcc_25[ri]
        self.lcc_im1_30 = self.lcc_im1_30[ri]
        self.lcc_im2_30 = self.lcc_im2_30[ri]
        self.min_lcc_30 = self.min_lcc_30[ri]
        self.max_lcc_30 = self.max_lcc_30[ri]
        self.lcc_im1_35 = self.lcc_im1_35[ri]
        self.lcc_im2_35 = self.lcc_im2_35[ri]
        self.min_lcc_35 = self.min_lcc_35[ri]
        self.max_lcc_35 = self.max_lcc_35[ri]
        self.lcc_im1_40 = self.lcc_im1_40[ri]
        self.lcc_im2_40 = self.lcc_im2_40[ri]
        self.min_lcc_40 = self.min_lcc_40[ri]
        self.max_lcc_40 = self.max_lcc_40[ri]
        self.shortest_path_length = self.shortest_path_length[ri]
        self.mds_rank_percentage_im1_im2 = self.mds_rank_percentage_im1_im2[ri]
        self.mds_rank_percentage_im2_im1 = self.mds_rank_percentage_im2_im1[ri]
        self.distance_rank_percentage_im1_im2_gt = self.distance_rank_percentage_im1_im2_gt[ri]
        self.distance_rank_percentage_im2_im1_gt = self.distance_rank_percentage_im2_im1_gt[ri]
        self.labels = self.labels[ri]
        self.weights = self.weights[ri]



        self.transform = transform
        self.loader = loader
        self.unique_fns_dsets = np.array([])
        self.unique_fns = np.array([])
        self.unique_imgs = {}

        # if self.options['convnet_load_dataset_in_memory']:
        #     self.f_img1 = []
        #     self.f_img2 = []
        #     self.f_se_rmatches_img1 = []
        #     self.f_se_matches_img1 = []
        #     self.f_pe_img1 = []
        #     self.f_pe_mask_img1 = []
        #     self.f_pe_warped_img1 = []
        #     self.f_se_rmatches_img2 = []
        #     self.f_se_matches_img2 = []
        #     self.f_pe_img2 = []
        #     self.f_pe_mask_img2 = []
        #     self.f_pe_warped_img2 = []
        #     self.f_se_fm1 = []
        #     self.f_se_fm2 = []
        #     self.f_tm_fm1 = []
        #     self.f_tm_fm2 = []
        #     self.f_se_non_rmatches_img1 = []
        #     self.f_se_non_rmatches_img2 = []
        #     self.f_se_rmatches_secondary_motion_img1 = []
        #     self.f_se_rmatches_secondary_motion_img2 = []
        #     for i, _ in enumerate(self.fns):
        #         img1_fn = os.path.join(self.dsets[i], 'images-resized', self.fns[i,0])
        #         img2_fn = os.path.join(self.dsets[i], 'images-resized', self.fns[i,1])
        #         se_rmatches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
        #         se_rmatches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
        #         se_matches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'matches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
        #         se_matches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'matches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
        #         # pe_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-a-em-filtered.png'.format(self.fns[i,0], self.fns[i,1]))
        #         # pe_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-a-em-filtered.png'.format(self.fns[i,1], self.fns[i,0]))
        #         # pe_mask_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-a-m-filtered.png'.format(self.fns[i,0], self.fns[i,1]))
        #         # pe_mask_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-a-m-filtered.png'.format(self.fns[i,1], self.fns[i,0]))
        #         # pe_warped_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-a-wi-filtered.png'.format(self.fns[i,0], self.fns[i,1]))
        #         # pe_warped_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-a-wi-filtered.png'.format(self.fns[i,1], self.fns[i,0]))
        #         pe_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-em-filtered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
        #         pe_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-em-filtered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
        #         pe_mask_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-m-filtered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
        #         pe_mask_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-m-filtered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
        #         pe_warped_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-wi-filtered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
        #         pe_warped_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-wi-filtered-ga.png'.format(self.fns[i,1], self.fns[i,0]))

        #         se_fm1_fn = os.path.join(self.dsets[i], 'classifier_features', 'feature_maps', 'feature---{}.png'.format(self.fns[i,0]))
        #         se_fm2_fn = os.path.join(self.dsets[i], 'classifier_features', 'feature_maps', 'feature---{}.png'.format(self.fns[i,1]))
        #         tm_fm1_fn = os.path.join(self.dsets[i], 'classifier_features', 'track_maps', '{}.png'.format(self.fns[i,0]))
        #         tm_fm2_fn = os.path.join(self.dsets[i], 'classifier_features', 'track_maps', '{}.png'.format(self.fns[i,1]))
        #         se_non_rmatches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'non_rmatches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
        #         se_non_rmatches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'non_rmatches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
        #         se_rmatches_secondary_motion_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches_secondary_motion---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
        #         se_rmatches_secondary_motion_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches_secondary_motion---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))

                
        #         if self.options['convnet_use_rmatches_map']:
        #             self.f_se_rmatches_img1.append(self.loader(se_rmatches_img1_fn).convert('L'))
        #             self.f_se_rmatches_img2.append(self.loader(se_rmatches_img2_fn).convert('L'))
        #         else:
        #             self.f_se_rmatches_img1.append(np.zeros((1,1,1)))
        #             self.f_se_rmatches_img2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_matches_map']:
        #             self.f_se_matches_img1.append(self.loader(se_matches_img1_fn).convert('L'))
        #             self.f_se_matches_img2.append(self.loader(se_matches_img2_fn).convert('L'))
        #         else:
        #             self.f_se_matches_img1.append(np.zeros((1,1,1)))
        #             self.f_se_matches_img2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_photometric_error_maps']:
        #             self.f_pe_img1.append(self.loader(pe_img1_fn).convert('L'))
        #             self.f_pe_mask_img1.append(self.loader(pe_mask_img1_fn).convert('L'))
        #             self.f_pe_img2.append(self.loader(pe_img2_fn).convert('L'))
        #             self.f_pe_mask_img2.append(self.loader(pe_mask_img2_fn).convert('L'))
        #         else:
        #             self.f_pe_img1.append(np.zeros((1,1,1)))
        #             self.f_pe_mask_img1.append(np.zeros((1,1,1)))
        #             self.f_pe_img2.append(np.zeros((1,1,1)))
        #             self.f_pe_mask_img2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_images']:
        #             self.f_img1.append(self.loader(img1_fn))
        #             self.f_img2.append(self.loader(img2_fn))
        #         else:
        #             self.f_img1.append(np.zeros((1,1,1)))
        #             self.f_img2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_warped_images']:
        #             self.f_pe_warped_img1.append(self.loader(pe_warped_img1_fn))    
        #             self.f_pe_warped_img2.append(self.loader(pe_warped_img2_fn))
        #         else:
        #             self.f_pe_warped_img1.append(np.zeros((1,1,1)))
        #             self.f_pe_warped_img2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_feature_match_map']:
        #             self.f_se_fm1.append(self.loader(se_fm1_fn).convert('L'))
        #             self.f_se_fm2.append(self.loader(se_fm2_fn).convert('L'))
        #         else:
        #             self.f_se_fm1.append(np.zeros((1,1,1)))
        #             self.f_se_fm2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_track_map']:
        #             self.f_tm_fm1.append(self.loader(tm_fm1_fn).convert('L'))
        #             self.f_tm_fm2.append(self.loader(tm_fm2_fn).convert('L'))
        #         else:
        #             self.f_tm_fm1.append(np.zeros((1,1,1)))
        #             self.f_tm_fm2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_non_rmatches_map']:
        #             self.f_se_non_rmatches_img1.append(self.loader(se_non_rmatches_img1_fn).convert('L'))
        #             self.f_se_non_rmatches_img2.append(self.loader(se_non_rmatches_img2_fn).convert('L'))
        #         else:
        #             self.f_se_non_rmatches_img1.append(np.zeros((1,1,1)))
        #             self.f_se_non_rmatches_img2.append(np.zeros((1,1,1)))

        #         if self.options['convnet_use_rmatches_secondary_motion_map']:
        #             self.f_se_rmatches_secondary_motion_img1.append(self.loader(se_rmatches_secondary_motion_img1_fn).convert('L'))
        #             self.f_se_rmatches_secondary_motion_img2.append(self.loader(se_rmatches_secondary_motion_img2_fn).convert('L'))
        #         else:
        #             self.f_se_rmatches_secondary_motion_img1.append(np.zeros((1,1,1)))
        #             self.f_se_rmatches_secondary_motion_img2.append(np.zeros((1,1,1)))

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
        else:
            self.positive_sample_indices = None
            self.negative_sample_indices = None

    def __getitem__(self, index):
        indices = []
        if self.train and self.options['triplet-sampling-strategy'] == 'random':
            # In train mode, index references positive samples only
            p_index = self.positive_sample_indices[index]
            n_index = self.negative_sample_indices[random.randint(0, len(self.negative_sample_indices) - 1)]
            

            if self.options['sample-inclass'] and random.random() >= 0.5:
                p_index__ = self.positive_sample_indices[random.randint(0, len(self.positive_sample_indices) - 1)]
                if 2.0*self.num_rmatches[p_index] >= self.num_rmatches[p_index__] and p_index__ != p_index:
                    indices = [p_index, p_index__]
                elif 2.0*self.num_rmatches[p_index__] >= self.num_rmatches[p_index] and p_index__ != p_index:
                    indices = [p_index__, p_index]
                else:
                    indices = [p_index, n_index]
            else:
                if random.random() >= 0.5:
                    indices = [p_index, n_index]
                else:
                    indices = [n_index, p_index]
                # indices = [p_index, n_index]

        elif self.train and self.options['triplet-sampling-strategy'] == 'uniform-files':
            # In train mode, index references positive samples only
            _im = self.unique_fns[index]
            _dset = self.unique_fns_dsets[index]

            im_pos_samples = np.array(self.sample_hierarchy[_dset][_im]['positive_samples']).copy()
            im_neg_samples = np.array(self.sample_hierarchy[_dset][_im]['negative_samples']).copy()

            p_index = im_pos_samples[random.randint(0, len(im_pos_samples) - 1)]
            n_index = im_neg_samples[random.randint(0, len(im_neg_samples) - 1)]

            if self.options['sample-inclass'] and random.random() >= 0.5 and len(im_pos_samples) >= 2:
                # sample another positive
                # delete first sampled positive index
                im_pos_samples = np.delete(im_pos_samples, np.argwhere(im_pos_samples==p_index))
                p_index__ = im_pos_samples[random.randint(0, len(im_pos_samples) - 1)]
                if 2*self.num_rmatches[p_index] >= self.num_rmatches[p_index__]:
                    indices = [p_index, p_index__]
                elif 2*self.num_rmatches[p_index__] >= self.num_rmatches[p_index]:
                    indices = [p_index__, p_index]
                else:
                    indices = [p_index, n_index]
            else:
                indices = [p_index, n_index]
        else:
            # In test mode/regular training mode, index references entire dataset
            indices = [index, index]


        data = []
        for i in indices:
            # if self.options['convnet_load_dataset_in_memory']:
            #     img1 = self.transform(self.f_img1[i])
            #     img2 = self.transform(self.f_img2[i])
            #     # No need to cache these images since they're only used once per epoch (they're unique for each pair of images)
            #     se_rmatches_img1 = self.transform(self.f_se_rmatches_img1[i])
            #     se_matches_img1 = self.transform(self.f_se_matches_img1[i])
            #     pe_img1 = self.transform(self.f_pe_img1[i])
            #     pe_mask_img1 = self.transform(self.f_pe_mask_img1[i])
            #     pe_warped_img1 = self.transform(self.f_pe_warped_img1[i])
            #     se_rmatches_img2 = self.transform(self.f_se_rmatches_img2[i])
            #     se_matches_img2 = self.transform(self.f_se_matches_img2[i])
            #     pe_img2 = self.transform(self.f_pe_img2[i])
            #     pe_mask_img2 = self.transform(self.f_pe_mask_img2[i])
            #     pe_warped_img2 = self.transform(self.f_pe_warped_img2[i])
            #     se_fm1 = self.transform(self.f_se_fm1[i])
            #     se_fm2 = self.transform(self.f_se_fm2[i])
            #     tm_fm1 = self.transform(self.f_tm_fm1[i])
            #     tm_fm2 = self.transform(self.f_tm_fm2[i])
            #     se_non_rmatches_img1 = self.transform(self.f_se_non_rmatches_img1[i])
            #     se_non_rmatches_img2 = self.transform(self.f_se_non_rmatches_img2[i])
            #     se_rmatches_secondary_motion_img1 = self.transform(self.f_se_rmatches_secondary_motion_img1[i])
            #     se_rmatches_secondary_motion_img2 = self.transform(self.f_se_rmatches_secondary_motion_img2[i])
            # else:
            img1_fn = os.path.join(self.dsets[i], 'images-resized', self.fns[i,0])
            img2_fn = os.path.join(self.dsets[i], 'images-resized', self.fns[i,1])
            
            # Example: rmatches---DSC_0322.JPG-DSC_0308.JPG.png
            se_rmatches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
            se_rmatches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
            se_matches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'matches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
            se_matches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'matches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))

            if 'pe_filtered' in self.options['experiment']:
                pe_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-em-filtered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
                pe_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-em-filtered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
                pe_mask_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-m-filtered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
                pe_mask_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-m-filtered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
                pe_warped_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-wi-filtered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
                pe_warped_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-wi-filtered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
            else:
                pe_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-em-unfiltered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
                pe_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-em-unfiltered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
                pe_mask_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-m-unfiltered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
                pe_mask_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-m-unfiltered-ga.png'.format(self.fns[i,1], self.fns[i,0]))
                pe_warped_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-wi-unfiltered-ga.png'.format(self.fns[i,0], self.fns[i,1]))
                pe_warped_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps', '{}-{}-a-wi-unfiltered-ga.png'.format(self.fns[i,1], self.fns[i,0]))

            se_fm1_fn = os.path.join(self.dsets[i], 'classifier_features', 'feature_maps', 'feature---{}.png'.format(self.fns[i,0]))
            se_fm2_fn = os.path.join(self.dsets[i], 'classifier_features', 'feature_maps', 'feature---{}.png'.format(self.fns[i,1]))
            tm_fm1_fn = os.path.join(self.dsets[i], 'classifier_features', 'track_maps', '{}.png'.format(self.fns[i,0]))
            tm_fm2_fn = os.path.join(self.dsets[i], 'classifier_features', 'track_maps', '{}.png'.format(self.fns[i,1]))
            se_non_rmatches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'non_rmatches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
            se_non_rmatches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'non_rmatches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
            se_rmatches_secondary_motion_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches_secondary_motion---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
            se_rmatches_secondary_motion_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches_secondary_motion---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))

            if self.options['convnet_use_rmatches_map']:
                se_rmatches_img1 = self.transform(self.loader(se_rmatches_img1_fn).convert('L'))
                se_rmatches_img2 = self.transform(self.loader(se_rmatches_img2_fn).convert('L'))
            else:
                se_rmatches_img1 = np.zeros((1,1,1))
                se_rmatches_img2 = np.zeros((1,1,1))

            if self.options['convnet_use_matches_map']:
                se_matches_img1 = self.transform(self.loader(se_matches_img1_fn).convert('L'))
                se_matches_img2 = self.transform(self.loader(se_matches_img2_fn).convert('L'))
            else:
                se_matches_img1 = np.zeros((1,1,1))
                se_matches_img2 = np.zeros((1,1,1))

            if self.options['convnet_use_photometric_error_maps']:
                if not os.path.isfile(pe_img1_fn):
                    pe_img1 = self.transform(Image.new('L', (self.options['convnet_input_size'], self.options['convnet_input_size'])))
                    pe_mask_img1 = self.transform(Image.new('L', (self.options['convnet_input_size'], self.options['convnet_input_size'])))
                else:
                    pe_img1 = self.transform(self.loader(pe_img1_fn).convert('L'))
                    pe_mask_img1 = self.transform(self.loader(pe_mask_img1_fn).convert('L'))

                if not os.path.isfile(pe_img2_fn):
                    pe_img2 = self.transform(Image.new('L', (self.options['convnet_input_size'], self.options['convnet_input_size'])))
                    pe_mask_img2 = self.transform(Image.new('L', (self.options['convnet_input_size'], self.options['convnet_input_size'])))
                else:
                    pe_img2 = self.transform(self.loader(pe_img2_fn).convert('L'))
                    pe_mask_img2 = self.transform(self.loader(pe_mask_img2_fn).convert('L'))
            else:
                pe_img1 = np.zeros((1,1,1))
                pe_mask_img1 = np.zeros((1,1,1))
                pe_img2 = np.zeros((1,1,1))
                pe_mask_img2 = np.zeros((1,1,1))
            
            if self.options['convnet_use_images']:
                img1 = self.transform(self.loader(img1_fn))
                img2 = self.transform(self.loader(img2_fn))
            else:
                img1 = np.zeros((1,1,1))
                img2 = np.zeros((1,1,1))

            if self.options['convnet_use_warped_images']:
                pe_warped_img1 = self.transform(self.loader(pe_warped_img1_fn))
                pe_warped_img2 = self.transform(self.loader(pe_warped_img2_fn))
            else:
                pe_warped_img1 = np.zeros((1,1,1))
                pe_warped_img2 = np.zeros((1,1,1))

            if self.options['convnet_use_feature_match_map']:
                se_fm1 = self.transform(self.loader(se_fm1_fn).convert('L'))
                se_fm2 = self.transform(self.loader(se_fm2_fn).convert('L'))
            else:
                se_fm1 = np.zeros((1,1,1))
                se_fm2 = np.zeros((1,1,1))

            if self.options['convnet_use_track_map']:
                tm_fm1 = self.transform(self.loader(tm_fm1_fn).convert('L'))
                tm_fm2 = self.transform(self.loader(tm_fm2_fn).convert('L'))
            else:
                tm_fm1 = np.zeros((1,1,1))
                tm_fm2 = np.zeros((1,1,1))
        
            if self.options['convnet_use_non_rmatches_map']:
                se_non_rmatches_img1 = self.transform(self.loader(se_non_rmatches_img1_fn).convert('L'))
                se_non_rmatches_img2 = self.transform(self.loader(se_non_rmatches_img2_fn).convert('L'))
            else:
                se_non_rmatches_img1 = np.zeros((1,1,1))
                se_non_rmatches_img2 = np.zeros((1,1,1))

            if self.options['convnet_use_rmatches_secondary_motion_map']:
                se_rmatches_secondary_motion_img1 = self.transform(self.loader(se_rmatches_secondary_motion_img1_fn).convert('L'))
                se_rmatches_secondary_motion_img2 = self.transform(self.loader(se_rmatches_secondary_motion_img2_fn).convert('L'))
            else:
                se_rmatches_secondary_motion_img1 = np.zeros((1,1,1))
                se_rmatches_secondary_motion_img2 = np.zeros((1,1,1))


            data.append([self.dsets[i].tolist(), self.fns[i,0].tolist(), self.fns[i,1].tolist(), self.R11s[i], self.R12s[i], self.R13s[i], \
                self.R21s[i], self.R22s[i], self.R23s[i], self.R31s[i], self.R32s[i], self.R33s[i], \
                self.num_rmatches[i], self.num_matches[i], self.spatial_entropy_1_8x8[i], \
                self.spatial_entropy_2_8x8[i], self.spatial_entropy_1_16x16[i], self.spatial_entropy_2_16x16[i], \
                self.pe_histogram[i].reshape((-1,1)), self.pe_polygon_area_percentage[i], self.nbvs_im1[i], self.nbvs_im2[i], \
                self.te_histogram[i].reshape((-1,1)), self.ch_im1[i].reshape((-1,1)), self.ch_im2[i].reshape((-1,1)), \
                self.vt_rank_percentage_im1_im2[i], self.vt_rank_percentage_im2_im1[i], \
                self.sq_rank_scores_mean[i], self.sq_rank_scores_min[i], self.sq_rank_scores_max[i], self.sq_distance_scores[i], \
                self.shortest_path_length[i], \
                self.mds_rank_percentage_im1_im2[i], self.mds_rank_percentage_im2_im1[i], \
                self.distance_rank_percentage_im1_im2_gt[i], self.distance_rank_percentage_im2_im1_gt[i], \
                self.labels[i], img1, img2, se_rmatches_img1, se_rmatches_img2, se_matches_img1, se_matches_img2, pe_img1, pe_img2, pe_mask_img1, pe_mask_img2, \
                pe_warped_img1, pe_warped_img2, se_fm1, se_fm2, tm_fm1, tm_fm2, se_non_rmatches_img1, se_non_rmatches_img2, se_rmatches_secondary_motion_img1, se_rmatches_secondary_motion_img2
                ])

        return data

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

def convnet_accuracy(y, y_gt, all_im1s, all_im2s, color, ls, epoch, thresholds):
    np.save('results-{}.npy'.format(epoch), np.concatenate((y.reshape(-1,1),y_gt.reshape(-1,1)),axis=1) )
    np.savetxt('results-files-{}.out'.format(epoch), np.concatenate((all_im1s.reshape(-1,1),all_im2s.reshape(-1,1)),axis=1), delimiter=" ", fmt="%s")

    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_gt, y)

    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')
    auc = sklearn.metrics.auc(recall, precision)
    return auc

def inference(data_loader, model, epoch, run_dir, logger, opts, range_min, range_max, mode=None, optimizer=None):
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
    all_fns = []
    all_num_rmatches = []
    all_shortest_path_lengths = []
    all_predictions = np.empty((1,2))
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
                shortest_path_length, \
                mds_rank_percentage_im1_im2, mds_rank_percentage_im2_im1, \
                distance_rank_percentage_im1_im2_gt, distance_rank_percentage_im2_im1_gt, \
                labels, img1, img2, se_rmatches_img1, se_rmatches_img2, se_matches_img1, se_matches_img2, pe_img1, pe_img2, pe_mask_img1, pe_mask_img2, \
                pe_warped_img1, pe_warped_img2, se_fm1, se_fm2, tm_fm1, tm_fm2, se_non_rmatches_img1, se_non_rmatches_img2, se_rmatches_secondary_motion_img1, se_rmatches_secondary_motion_img2 = sample

            _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, \
                _num_rmatches, _num_matches, _spatial_entropy_1_8x8, _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, \
                _pe_histogram, _pe_polygon_area_percentage, _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, \
                _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
                _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
                _shortest_path_length, \
                _mds_rank_percentage_im1_im2, _mds_rank_percentage_im2_im1, \
                _distance_rank_percentage_im1_im2_gt, _distance_rank_percentage_im2_im1_gt, \
                _labels, _img1, _img2, \
                _se_rmatches_img1, _se_rmatches_img2, _se_matches_img1, _se_matches_img2, _pe_img1, _pe_img2, _pe_mask_img1, _pe_mask_img2, \
                _pe_warped_img1, _pe_warped_img2, _se_fm1, _se_fm2, _tm_fm1, _tm_fm2, _se_non_rmatches_img1, _se_non_rmatches_img2, _se_rmatches_secondary_motion_img1, _se_rmatches_secondary_motion_img2 = \
                R11s.cuda(), R12s.cuda(), R13s.cuda(), R21s.cuda(), R22s.cuda(), R23s.cuda(), R31s.cuda(), R32s.cuda(), R33s.cuda(), \
                num_rmatches.cuda(), num_matches.cuda(), spatial_entropy_1_8x8.cuda(), spatial_entropy_2_8x8.cuda(), spatial_entropy_1_16x16.cuda(), spatial_entropy_2_16x16.cuda(), \
                pe_histogram.cuda(), pe_polygon_area_percentage.cuda(), nbvs_im1.cuda().type(torch.cuda.FloatTensor), nbvs_im2.cuda().type(torch.cuda.FloatTensor), te_histogram.cuda().type(torch.cuda.FloatTensor), ch_im1.cuda(), ch_im2.cuda(), \
                vt_rank_percentage_im1_im2.cuda(), vt_rank_percentage_im2_im1.cuda(), \
                sq_rank_scores_mean.cuda(), sq_rank_scores_min.cuda(), sq_rank_scores_max.cuda(), sq_distance_scores.cuda(), \
                shortest_path_length.cuda(), \
                mds_rank_percentage_im1_im2.cuda(), mds_rank_percentage_im2_im1.cuda(), \
                distance_rank_percentage_im1_im2_gt.cuda(), distance_rank_percentage_im2_im1_gt.cuda(), \
                labels.cuda(), img1.cuda(), img2.cuda(), \
                se_rmatches_img1.cuda(), se_rmatches_img2.cuda(), se_matches_img1.cuda(), se_matches_img2.cuda(), pe_img1.cuda(), pe_img2.cuda(), \
                pe_mask_img1.cuda(), pe_mask_img2.cuda(), pe_warped_img1.cuda(), pe_warped_img2.cuda(), se_fm1.cuda(), se_fm2.cuda(), tm_fm1.cuda(), tm_fm2.cuda(), \
                se_non_rmatches_img1.cuda(), se_non_rmatches_img2.cuda(), se_rmatches_secondary_motion_img1.cuda(), se_rmatches_secondary_motion_img2.cuda()

            arg = [Variable(_R11s), Variable(_R12s), Variable(_R13s), Variable(_R21s), Variable(_R22s), Variable(_R23s), Variable(_R31s), Variable(_R32s), Variable(_R33s), \
                Variable(_num_rmatches), Variable(_num_matches), Variable(_spatial_entropy_1_8x8), Variable(_spatial_entropy_2_8x8), Variable(_spatial_entropy_1_16x16), Variable(_spatial_entropy_2_16x16), \
                Variable(_pe_histogram), Variable(_pe_polygon_area_percentage), Variable(_nbvs_im1), Variable(_nbvs_im2), Variable(_te_histogram), Variable(_ch_im1), Variable(_ch_im2), \
                Variable(_vt_rank_percentage_im1_im2), Variable(_vt_rank_percentage_im2_im1), \
                Variable(_sq_rank_scores_mean), Variable(_sq_rank_scores_min), Variable(_sq_rank_scores_max), Variable(_sq_distance_scores), \
                Variable(_shortest_path_length),
                Variable(_mds_rank_percentage_im1_im2), Variable(_mds_rank_percentage_im2_im1), \
                Variable(_distance_rank_percentage_im1_im2_gt), Variable(_distance_rank_percentage_im2_im1_gt), \
                Variable(_labels), Variable(_img1), Variable(_img2), \
                Variable(_se_rmatches_img1), Variable(_se_rmatches_img2), Variable(_se_matches_img1), Variable(_se_matches_img2), Variable(_pe_img1), \
                Variable(_pe_img2), Variable(_pe_mask_img1), Variable(_pe_mask_img2), Variable(_pe_warped_img1), Variable(_pe_warped_img2), \
                Variable(_se_fm1), Variable(_se_fm2), Variable(_tm_fm1), Variable(_tm_fm2), \
                Variable(_se_non_rmatches_img1), Variable(_se_non_rmatches_img2), \
                Variable(_se_rmatches_secondary_motion_img1), Variable(_se_rmatches_secondary_motion_img2)
                ]
            

            y_prediction = model(arg)
            target = Variable(_labels.type(torch.cuda.LongTensor))

            if j == 0:
                y_predictions_softmax = nn.functional.softmax(y_prediction)
                y_predictions_logits = y_prediction
                positive_predictions = nn.functional.softmax(y_prediction)#y_prediction
                positive_targets = target
                targets = target
                num_rmatches_b = Variable(_num_rmatches)
                shortest_path_lengths_b = Variable(_shortest_path_length)
                fns_b = np.concatenate((np.array(im1s).reshape((-1,1)),np.array(im2s).reshape((-1,1))), axis=1)
                dsets_b = np.array(dsets)
            else:
                y_predictions_softmax = torch.cat((y_predictions_softmax, nn.functional.softmax(y_prediction)))
                y_predictions_logits = torch.cat((y_predictions_logits, y_prediction), dim=0)
                negative_predictions = nn.functional.softmax(y_prediction)#y_prediction
                negative_targets = target
                targets = torch.cat((targets, target), dim=0)
                num_rmatches_b = torch.cat((num_rmatches_b, Variable(_num_rmatches)), dim=0)
                shortest_path_lengths_b = torch.cat((shortest_path_lengths_b, Variable(_shortest_path_length)))
                fns_b = np.concatenate((fns_b, np.concatenate((np.array(im1s).reshape((-1,1)),np.array(im2s).reshape((-1,1))), axis=1)), axis=0)
                dsets_b = np.concatenate((dsets_b, np.array(dsets)), axis=0)

            # The samples are the same in the normal strategy, so we only need to do it once
            if opts['triplet-sampling-strategy'] == 'normal' or mode == 'test':
                break


        if opts['loss'] == 'cross-entropy' or mode == 'test':
            loss = cross_entropy_loss(y_predictions_logits, targets)
        elif opts['loss'] == 'triplet':
            # loss = margin_ranking_loss(y_predictions_logits[:,1], y_predictions_logits[:,0], reformatted_targets)
            # ones_label = Variable(torch.ones(positive_predictions.size()[0]).cuda().type(torch.cuda.FloatTensor))
            # loss = margin_ranking_loss(positive_predictions[:,1], negative_predictions[:,1], ones_label)
            # loss = 0.5*margin_ranking_loss(positive_predictions[:,1], negative_predictions[:,1], ones_label) + 0.5*cross_entropy_loss(y_predictions_logits, targets)
            
            # mloss = margin_ranking_loss(positive_predictions[:,1], negative_predictions[:,1], ones_label)
            loss = cross_entropy_loss(y_predictions_logits, targets)

            # loss = mloss + closs
            # loss = closs
            # print '#'*100
            # print y_predictions_logits
            # print positive_predictions
            # print negative_predictions
            # print '$'*100
            # print num_rmatches_b
            # print '-'*100
            # print ones_label
            # print targets
            # print '!'*100
            # print mloss
            # print closs
            # print loss
            # import sys; sys.exit(1);

        if mode == 'train':
            loss.backward()
            optimizer.step()

        correct_counts = correct_counts + get_correct_counts(targets, y_predictions_softmax, False)
        if not arrays_initialized:
            all_dsets = dsets_b
            all_fns = fns_b
            all_targets = targets.data.cpu().numpy()
            all_predictions = y_predictions_softmax.data.cpu().numpy()
            all_num_rmatches = num_rmatches_b.data.cpu().numpy()
            all_shortest_path_lengths = shortest_path_lengths_b.data.cpu().numpy()
            arrays_initialized = True
        else:
            all_dsets = np.concatenate((all_dsets, dsets_b), axis=0)
            all_fns = np.concatenate((all_fns, fns_b), axis=0)
            all_targets = np.concatenate((all_targets, targets.data.cpu().numpy()), axis=0)
            all_predictions = np.concatenate((all_predictions, y_predictions_softmax.data.cpu().numpy()), axis=0)
            all_num_rmatches = np.concatenate((all_num_rmatches, num_rmatches_b.data.cpu().numpy()), axis=0)
            all_shortest_path_lengths = np.concatenate((all_shortest_path_lengths, shortest_path_lengths_b.data.cpu().numpy()), axis=0)

        if (batch_idx + 1) % opts['log_interval'] == 0:
            if opts['triplet-sampling-strategy'] == 'normal' or mode == 'test':
                num_tests_so_far = (batch_idx + 1) * opts['batch_size'] # only one sample
                dataset_size = data_loader.dataset.__len__()
            else:
                num_tests_so_far = (batch_idx + 1) * 2 * opts['batch_size'] # positive and negative samples
                dataset_size = 2 * data_loader.dataset.__len__() # iterate over all positive examples * 2 (since negatives are repeated)

            accuracy = correct_counts*100.0/num_tests_so_far
            if mode == 'train':
                if logger is not None:
                    print(
                        '{} Epoch: {} Accuracy: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            mode.upper(), epoch, accuracy, (batch_idx + 1) * opts['batch_size'], len(data_loader.dataset),
                            100.0 * num_tests_so_far / dataset_size,
                            loss.data[0]))
            else:
                if logger is not None:
                    print(
                        '{} Epoch: {} Accuracy: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            mode.upper(), epoch, accuracy, (batch_idx + 1) * opts['batch_size'], len(data_loader.dataset),
                            100.0 * num_tests_so_far / dataset_size,
                            loss.data[0]))

        # if mode == 'train':
        cum_loss = cum_loss + loss.data[0]

  # num_tests = data_loader.dataset.__len__()
  # accuracy = correct_counts*100.0/num_tests
    if mode == 'train':
        # num_tests = data_loader.dataset.__len__()
        # correct_counts = get_correct_counts(all_targets, all_predictions, False)
        # accuracy = correct_counts*100.0/num_tests

        adjust_learning_rate(optimizer, opts)
        cum_loss = cum_loss/(num_tests_so_far/(1.0*opts['batch_size']))
        if logger is not None:
            logger.log_value('TRAIN-LR', optimizer.param_groups[0]['lr'])
            logger.log_value('TRAIN-ACCURACY', accuracy)
            logger.log_value('TRAIN-LOSS', cum_loss)
            print ('{} Epoch: {}  Correct: {}  Accuracy: {}  Loss: {}'.format(mode.upper(), epoch, correct_counts, \
                round(accuracy, 2), round(cum_loss,2)))
        # do checkpointing
        if (epoch + 1) % 1 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               '{}/checkpoint_{}.pth'.format(run_dir, epoch))
    else:
        cum_loss = cum_loss/(num_tests_so_far/(1.0*opts['batch_size']))
        if logger is not None:
            logger.log_value('TEST-ACCURACY-RANGE-{}-{}'.format(range_min, range_max), accuracy)
            logger.log_value('TEST-LOSS-RANGE-{}-{}'.format(range_min, range_max), cum_loss)
            print ('{} Epoch: {}  Correct: {}  Accuracy: {}  Loss: {}\n'.format(mode.upper(), epoch, correct_counts, \
                round(accuracy, 2), round(cum_loss,2)))

    if logger is not None:
        markers_exp = [0.3, 0.4, 0.5]
        auc_pr, auc_roc, pr, roc = matching_classifiers.calculate_dataset_auc(all_predictions[:,1], all_targets)#, color='red', ls='solid', markers=markers_exp)
        _, _, _, _, _, auc_per_image_mean, _ = matching_classifiers.calculate_per_image_mean_auc(all_dsets, all_fns, all_predictions[:,1], all_targets)
        _, _, _, _, mean_precision_per_image = matching_classifiers.calculate_per_image_precision_top_k(all_dsets, all_fns, all_predictions[:,1], all_targets)

        markers_baseline = [15, 16, 20]
        auc_pr_baseline, auc_roc_baseline, pr_baseline, roc_baseline = matching_classifiers.calculate_dataset_auc(all_num_rmatches, all_targets)#, color='green', ls='dashed', markers=markers_baseline)
        _, _, _, _, _, auc_per_image_mean_baseline, _ = matching_classifiers.calculate_per_image_mean_auc(all_dsets, all_fns, all_num_rmatches, all_targets)
        _, _, _, _, mean_precision_per_image_baseline = matching_classifiers.calculate_per_image_precision_top_k(all_dsets, all_fns, all_num_rmatches, all_targets)


        fig_rocs = matching_classifiers.plot_rocs(roc, roc_baseline, auc_roc, auc_roc_baseline, markers=markers_exp, markers_baseline=markers_baseline)
        fig_prs = matching_classifiers.plot_prs(pr, pr_baseline, auc_pr, auc_pr_baseline, markers=markers_exp, markers_baseline=markers_baseline)
        fig_rates_baseline = matching_classifiers.plot_rates(roc_baseline, metric='rmatches', markers=markers_baseline)
        fig_rates = matching_classifiers.plot_rates(roc, metric='classifier scores', markers=markers_exp)

        print ('\t{} Epoch: {}     Experiment: {} AUC: {} / {} / {}'.format(mode.upper(), epoch, opts['experiment'], \
            round(auc_pr,3), round(auc_per_image_mean, 3), round(mean_precision_per_image, 3) \
            ))
        print ('\t{} Epoch: {}     Baseline: {} AUC: {} / {} / {}'.format(mode.upper(), epoch, 'Baseline', \
            round(auc_pr_baseline, 3), round(auc_per_image_mean_baseline, 3), round(mean_precision_per_image_baseline, 3) \
            ))
        print ('='*100)

    if mode == 'train':
        if logger is not None:
            logger.log_images('TRAIN-PR-PLOT-RANGE-{}-{}'.format(range_min, range_max), [fig_prs])
            logger.log_images('TRAIN-ROC-PLOT-RANGE-{}-{}'.format(range_min, range_max), [fig_rocs, fig_rates, fig_rates_baseline])
            logger.log_value('TRAIN-AUC-BASELINE', auc_pr_baseline)
            logger.log_value('TRAIN-AUCPI-BASELINE', auc_per_image_mean_baseline)
            logger.log_value('TRAIN-PPI-BASELINE', mean_precision_per_image_baseline)
            logger.log_value('TRAIN-AUC-EXP', auc_pr)
            logger.log_value('TRAIN-AUCPI-EXP', auc_per_image_mean)
            logger.log_value('TRAIN-PPI-EXP', mean_precision_per_image)
    else:
        if logger is not None:
            logger.log_images('TEST-PR-PLOT-RANGE-{}-{}'.format(range_min, range_max), [fig_prs])
            logger.log_images('TEST-ROC-PLOT-RANGE-{}-{}'.format(range_min, range_max), [fig_rocs, fig_rates, fig_rates_baseline])
            logger.log_value('TEST-AUC-BASELINE-RANGE-{}-{}'.format(range_min, range_max), auc_pr_baseline)
            logger.log_value('TEST-AUCPI-BASELINE-RANGE-{}-{}'.format(range_min, range_max), auc_per_image_mean_baseline)
            logger.log_value('TEST-PPI-BASELINE-RANGE-{}-{}'.format(range_min, range_max), mean_precision_per_image_baseline)
            logger.log_value('TEST-AUC-EXP-RANGE-{}-{}'.format(range_min, range_max), auc_pr)
            logger.log_value('TEST-AUCPI-EXP-RANGE-{}-{}'.format(range_min, range_max), auc_per_image_mean)
            logger.log_value('TEST-PPI-EXP-RANGE-{}-{}'.format(range_min, range_max), mean_precision_per_image)
    return all_fns, all_num_rmatches, all_predictions[:,1], all_shortest_path_lengths


triplet_loss = nn.TripletMarginLoss()
margin_ranking_loss = nn.MarginRankingLoss()
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
        optimizer = optim.SGD(model.parameters(), lr=opts['lr'],
                            momentum=0.9, dampening=0.9,
                            weight_decay=opts['wd'])
    elif opts['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts['lr'])
    
    else:
        raise Exception('Not supported optimizer: {0}'.format(opts['optimizer']))
    return optimizer

def classify_convnet_image_match_inference(arg):
    model = arg[-2]
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize((opts['convnet_input_size'], opts['convnet_input_size'])),
        tv.transforms.ToTensor(),
    ])

    test_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg, opts, transform=test_transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    # run_dir = os.path.join(opts['convnet_log_dir'], \
    #     'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}-is-{}-mlp-layer-size-{}-use-images-{}'.format(\
    #         opts['optimizer'], \
    #         opts['batch_size'], \
    #         opts['lr'], \
    #         opts['experiment'], \
    #         opts['loss'], \
    #         opts['triplet-sampling-strategy'], \
    #         opts['sample-inclass'], \
    #         opts['image_match_classifier_min_match'], \
    #         opts['image_match_classifier_max_match'], \
    #         opts['use_all_training_data'], \
    #         opts['use_small_weights'], \
    #         opts['model'], \
    #         opts['convnet_input_size'], \
    #         opts['mlp-layer-size'], \
    #         opts['convnet_use_images']
    #     )
    # )
    # logger = Logger(run_dir)
    epoch = 0

    fns, rmatches, scores, spl = inference(test_loader, model, epoch, run_dir=None, logger=None, opts=opts, range_min=opts['range_min'], range_max=opts['range_max'], mode='test', optimizer=None)

    return fns, rmatches, None, scores, spl, None

def classify_convnet_image_match_initialization(train_loader, test_loaders, run_dir, opts):
    # instantiate model and initialize weights
    kwargs = {}
    model = ConvNet(opts, **kwargs)

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
    run_dir = os.path.join(opts['convnet_log_dir'], \
        # 'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-model-{}-is-{}-mlp-layer-size-{}-use-images-{}-use_fmm-{}-use_tm-{}-tov-{}-use_rmm-{}-use_mm-{}-use_pems-{}'.format(\
        'run-ss-{}-sample-inclass-{}-model-{}-lr-{}-is-{}-mlp-layer-size-{}-use-images-{}-use_fmm-{}-use_tm-{}-tov-{}-use_rmm-{}-use_mm-{}-use_nrmm-{}-use-rmsmm-{}-use_pems-{}-exp-{}'.format(\
            # opts['optimizer'], \
            # opts['batch_size'], \
            
            # opts['experiment'], \
            # opts['loss'], \
            # # opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            # opts['image_match_classifier_min_match'], \
            # opts['image_match_classifier_max_match'], \
            # opts['use_all_training_data'], \
            opts['model'], \
            opts['lr'], \
            opts['convnet_input_size'], \
            opts['mlp-layer-size'], \
            opts['convnet_use_images'], \
            opts['convnet_use_feature_match_map'], \
            opts['convnet_use_track_map'], \
            opts['train_on_val'], \
            opts['convnet_use_rmatches_map'], \
            opts['convnet_use_matches_map'], \
            opts['convnet_use_non_rmatches_map'], \
            opts['convnet_use_rmatches_secondary_motion_map'], \
            opts['convnet_use_photometric_error_maps'], \
            opts['experiment']
        )
    )
    
    # logger_train = Logger('{}-{}'.format(run_dir, 'train'))
    # logger_test = Logger('{}-{}'.format(run_dir, 'test'))

    logger = Logger(run_dir)
    logger.global_step = start
    
    for epoch in range(start, end):
        _, _, training_scores, _ = inference(train_loader, model, epoch, run_dir, logger, opts, range_min=None, range_max=None,  mode='train', optimizer=optimizer)
        if (epoch + 1) % 1 == 0:
            for t, test_loader in enumerate(test_loaders):
                _, _, testing_scores, _ = inference(test_loader, model, epoch, run_dir, logger, opts, range_min=opts['ranges'][t][0], range_max=opts['ranges'][t][1], mode='test', optimizer=None)

        logger.step()

    # scores = inference(train_loader, model, epoch, run_dir, logger, opts, mode='train', optimizer=optimizer)
    return model, training_scores

def classify_convnet_image_match_training(arg, arg_te):
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    run_dir = os.path.join(opts['convnet_log_dir'], \
        # 'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-model-{}-is-{}-mlp-layer-size-{}-use-images-{}-use_fmm-{}-use_tm-{}-tov-{}-use_rmm-{}-use_mm-{}-use_pems-{}'.format(\
        'run-ss-{}-sample-inclass-{}-model-{}-lr-{}-is-{}-mlp-layer-size-{}-use-images-{}-use_fmm-{}-use_tm-{}-tov-{}-use_rmm-{}-use_mm-{}-use_nrmm-{}-use-rmsmm-{}-use_pems-{}-exp-{}'.format(\
            # opts['optimizer'], \
            # opts['batch_size'], \
            
            # opts['experiment'], \
            # opts['loss'], \
            # # opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            # opts['image_match_classifier_min_match'], \
            # opts['image_match_classifier_max_match'], \
            # opts['use_all_training_data'], \
            opts['model'], \
            opts['lr'], \
            opts['convnet_input_size'], \
            opts['mlp-layer-size'], \
            opts['convnet_use_images'], \
            opts['convnet_use_feature_match_map'], \
            opts['convnet_use_track_map'], \
            opts['train_on_val'], \
            opts['convnet_use_rmatches_map'], \
            opts['convnet_use_matches_map'], \
            opts['convnet_use_non_rmatches_map'], \
            opts['convnet_use_rmatches_secondary_motion_map'], \
            opts['convnet_use_photometric_error_maps'], \
            opts['experiment']
        )
    )
    matching_classifiers.mkdir_p(run_dir)

    train_transform = tv.transforms.Compose([
        tv.transforms.Resize((opts['convnet_input_size'], opts['convnet_input_size'])),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.ToTensor(),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize((opts['convnet_input_size'], opts['convnet_input_size'])),
        tv.transforms.ToTensor(),
    ])

    opts['range_min'] = 15
    opts['range_max'] = 5000
    train_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg, opts, transform=train_transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    print ('#'*50 + ' Training Data ' + '#'*50)
    print ('\tInliers: {}'.format(len(np.where(train_loader.dataset.labels == 1)[0])))
    print ('\tOutliers: {}'.format(len(np.where(train_loader.dataset.labels == 0)[0])))
    print ('\tTotal: {}'.format(train_loader.dataset.__len__()))
    print ('#'*100)

    opts = arg[-1]

    test_loaders = []

    opts['ranges'] = []
    for range_min, range_max in [[15,50], [50, 5000], [15, 5000]]:
        opts['ranges'].append([range_min, range_max])
        opts['range_min'] = range_min
        opts['range_max'] = range_max

        test_loader = torch.utils.data.DataLoader(
            ImageMatchingDataset(arg_te, opts, transform=test_transform),
            batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
            )
        test_loaders.append(test_loader)

        print ('#'*50 + ' Testing Data ' + '#'*50)
        print ('\tInliers: {}'.format(len(np.where(test_loader.dataset.labels == 1)[0])))
        print ('\tOutliers: {}'.format(len(np.where(test_loader.dataset.labels == 0)[0])))
        print ('\tTotal: {}'.format(test_loader.dataset.__len__()))
        print ('#'*110)


    model, training_scores = classify_convnet_image_match_initialization(train_loader, test_loaders, run_dir, opts)
    return None, None, model, training_scores, None
