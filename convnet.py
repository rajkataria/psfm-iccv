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
import matching_classifiers
import resnet

class Logger(object):
  def __init__(self, run_dir):
    # # clean previous logged data under the same directory name
    # self._remove(run_dir)

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


class ConvNet(nn.Module):

    def __init__(self, opts):#, num_classes=1000, init_weights=True):
        super(ConvNet, self).__init__()
        self.name = 'CONVNET'
        # self.features = features
        self.opts = opts

        # if opts['use_image_features']:
        if opts['model'] == 'resnet18':
            mlp_input_size = 1024
            __model = resnet.resnet18(pretrained=False, opts=opts)
        elif opts['model'] == 'resnet34':
            mlp_input_size = 1024
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

        
        if 'TE' in opts['features']:
            mlp_input_size += 80
        if 'NBVS' in opts['features']:
            mlp_input_size += 2

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, opts['mlp-layer-size']),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(opts['mlp-layer-size'], opts['mlp-layer-size']),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(opts['mlp-layer-size'], 2)
        )
        # else:
        #     self.mlp = nn.Sequential(
        #         # nn.Linear(919, 2),
        #         nn.Linear(923, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(),
        #         nn.Dropout(),
        #         nn.Linear(1024, 2),
        #         nn.Softmax()
        #     )

        # if opts['use_image_features']:
        #     # kwargs_ = {'num_classes': opts['num_classes']}
        #     # __model = models.__dict__['resnet34'](pretrained=True)
        #     if opts['model'] == 'resnet18':
        #         __model = resnet.resnet18(pretrained=False)
        #     elif opts['model'] == 'resnet34':
        #         __model = resnet.resnet34(pretrained=False)
        #     elif opts['model'] == 'resnet50':
        #         __model = resnet.resnet50(pretrained=False)
        #     elif opts['model'] == 'resnet101':
        #         __model = resnet.resnet101(pretrained=False)
        #     elif opts['model'] == 'resnet152':
        #         __model = resnet.resnet152(pretrained=False)
        #     else:
        #         __model = resnet.resnet50(pretrained=False)

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
            labels, img1, img2, se_rmatches_img1, se_rmatches_img2, se_matches_img1, se_matches_img2, pe_img1, pe_img2, \
            pe_mask_img1, pe_mask_img2, pe_warped_img1, pe_warped_img2 = arg

        # x = torch.cat(( \
        #     R11s.type(torch.cuda.FloatTensor).view(R11s.size(0), -1), \
        #     R12s.type(torch.cuda.FloatTensor).view(R12s.size(0), -1), \
        #     R13s.type(torch.cuda.FloatTensor).view(R13s.size(0), -1), \
        #     R21s.type(torch.cuda.FloatTensor).view(R21s.size(0), -1), \
        #     R22s.type(torch.cuda.FloatTensor).view(R22s.size(0), -1), \
        #     R23s.type(torch.cuda.FloatTensor).view(R23s.size(0), -1), \
        #     R31s.type(torch.cuda.FloatTensor).view(R31s.size(0), -1), \
        #     R32s.type(torch.cuda.FloatTensor).view(R32s.size(0), -1), \
        #     R33s.type(torch.cuda.FloatTensor).view(R33s.size(0), -1), \

        #     num_rmatches.type(torch.cuda.FloatTensor).view(num_rmatches.size(0), -1), \
        #     num_matches.type(torch.cuda.FloatTensor).view(num_matches.size(0), -1), \

        #     spatial_entropy_1_8x8.type(torch.cuda.FloatTensor).view(spatial_entropy_1_8x8.size(0), -1), \
        #     spatial_entropy_2_8x8.type(torch.cuda.FloatTensor).view(spatial_entropy_2_8x8.size(0), -1), \
        #     spatial_entropy_1_16x16.type(torch.cuda.FloatTensor).view(spatial_entropy_1_16x16.size(0), -1), \
        #     spatial_entropy_2_16x16.type(torch.cuda.FloatTensor).view(spatial_entropy_2_16x16.size(0), -1), \
            
        #     pe_histogram.type(torch.cuda.FloatTensor).view(pe_histogram.size(0), -1), \
        #     pe_polygon_area_percentage.type(torch.cuda.FloatTensor).view(pe_polygon_area_percentage.size(0), -1), \

        #     nbvs_im1.type(torch.cuda.FloatTensor).view(nbvs_im1.size(0), -1), \
        #     nbvs_im2.type(torch.cuda.FloatTensor).view(nbvs_im2.size(0), -1), \

        #     te_histogram.type(torch.cuda.FloatTensor).view(te_histogram.size(0), -1), \
            
        #     ch_im1.type(torch.cuda.FloatTensor).view(ch_im1.size(0), -1), \
        #     ch_im2.type(torch.cuda.FloatTensor).view(ch_im2.size(0), -1), \

        #     vt_rank_percentage_im1_im2.type(torch.cuda.FloatTensor).view(vt_rank_percentage_im1_im2.size(0), -1), \
        #     vt_rank_percentage_im2_im1.type(torch.cuda.FloatTensor).view(vt_rank_percentage_im2_im1.size(0), -1), \

        #     sq_rank_scores_mean.type(torch.cuda.FloatTensor).view(sq_rank_scores_mean.size(0), -1), \
        #     sq_rank_scores_min.type(torch.cuda.FloatTensor).view(sq_rank_scores_min.size(0), -1), \
        #     sq_rank_scores_max.type(torch.cuda.FloatTensor).view(sq_rank_scores_max.size(0), -1), \
        #     sq_distance_scores.type(torch.cuda.FloatTensor).view(sq_distance_scores.size(0), -1), \
        #     ), 1)

        # if self.opts['use_image_features']:
        # input1 = torch.cat((img1, se_rmatches_img1, se_matches_img1, pe_img1, pe_mask_img1, pe_warped_img1), 1)
        # input2 = torch.cat((img2, se_rmatches_img2, se_matches_img2, pe_img2, pe_mask_img2, pe_warped_img2), 1)
        # input1 = torch.cat((img1, se_rmatches_img1, se_matches_img1, pe_img1, pe_mask_img1), 1)
        # input2 = torch.cat((img2, se_rmatches_img2, se_matches_img2, pe_img2, pe_mask_img2), 1)
        if self.opts['convnet_use_images']:
            input1 = torch.cat((img1, se_rmatches_img1, se_matches_img1, pe_img1, pe_mask_img1), 1)
            input2 = torch.cat((img2, se_rmatches_img2, se_matches_img2, pe_img2, pe_mask_img2), 1)
        else:    
            input1 = torch.cat((se_rmatches_img1, se_matches_img1, pe_img1, pe_mask_img1), 1)
            input2 = torch.cat((se_rmatches_img2, se_matches_img2, pe_img2, pe_mask_img2), 1)
        # input1 = torch.cat((img1, se_rmatches_img1, se_matches_img1, pe_img1), 1)
        # input2 = torch.cat((img2, se_rmatches_img2, se_matches_img2, pe_img2), 1)
        # input1 = se_rmatches_img1
        # input2 = se_rmatches_img2

        

        i1 = self.image_feature_extractor(input1)
        i2 = self.image_feature_extractor(input2)
        # i1 = se_rmatches_img1
        # i2 = se_rmatches_img2
        # i_dot = torch.diag(torch.matmul(i1.view(i1.size(0), -1), torch.t(i2.view(i2.size(0), -1)))).view(-1, 1)
        # i_diff = i1.view(i1.size(0), -1) - i2.view(i2.size(0), -1)

        # y = torch.cat((i_diff, i_dot, x), 1)
        # y = torch.cat((i_diff, i_dot), 1)
        # y = i_diff
        y = torch.cat((i1.view(i2.size(0), -1), i2.view(i2.size(0), -1)), 1)

        if 'TE' in self.opts['features']:
            y = torch.cat((y.view(y.size(0), -1), te_histogram.view(te_histogram.size(0), -1)), 1)
        if 'NBVS' in self.opts['features']:
            y = torch.cat((y.view(y.size(0), -1), nbvs_im1.view(nbvs_im1.size(0), -1), nbvs_im2.view(nbvs_im2.size(0), -1)), 1)

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
        result = self.mlp(y)

            
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
            self.labels, self.weights, self.train, self.model, self.options = arg

        self.transform = transform
        self.loader = loader
        self.unique_fns_dsets = np.array([])
        self.unique_fns = np.array([])
        self.unique_imgs = {}

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
            indices = [p_index, n_index]
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
                if 15 + self.num_rmatches[p_index] >= self.num_rmatches[p_index__]:
                    indices = [p_index, p_index__]
                elif 15 + self.num_rmatches[p_index__] >= self.num_rmatches[p_index]:
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
            # if self.options['use_image_features']:
            img1_fn = os.path.join(self.dsets[i], 'images', self.fns[i,0])
            img2_fn = os.path.join(self.dsets[i], 'images', self.fns[i,1])
            # Example: rmatches---DSC_0322.JPG-DSC_0308.JPG.png
            se_rmatches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
            se_rmatches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'rmatches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
            se_matches_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'matches---{}-{}.png'.format(self.fns[i,0], self.fns[i,1]))
            se_matches_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'match_maps', 'matches---{}-{}.png'.format(self.fns[i,1], self.fns[i,0]))
            pe_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-em.png'.format(self.fns[i,0], self.fns[i,1]))
            pe_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-em.png'.format(self.fns[i,1], self.fns[i,0]))
            pe_mask_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-m.png'.format(self.fns[i,0], self.fns[i,1]))
            pe_mask_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-m.png'.format(self.fns[i,1], self.fns[i,0]))
            pe_warped_img1_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-wi.png'.format(self.fns[i,0], self.fns[i,1]))
            pe_warped_img2_fn = os.path.join(self.dsets[i], 'classifier_features', 'pe_maps-10-iterations-size-224', '{}-{}-wi.png'.format(self.fns[i,1], self.fns[i,0]))

            # if self.dsets[i] not in self.unique_imgs:
            #     self.unique_imgs[self.dsets[i]] = {}
            # if self.fns[i,0] not in self.unique_imgs[self.dsets[i]]:
            #     self.unique_imgs[self.dsets[i]][self.fns[i,0]] = self.transform(self.loader(img1_fn))
            # if self.fns[i,1] not in self.unique_imgs[self.dsets[i]]:
            #     self.unique_imgs[self.dsets[i]][self.fns[i,1]] = self.transform(self.loader(img2_fn))

            # img1 = self.unique_imgs[self.dsets[i]][self.fns[i,0]]
            # img2 = self.unique_imgs[self.dsets[i]][self.fns[i,1]]

            img1 = self.transform(self.loader(img1_fn))
            img2 = self.transform(self.loader(img2_fn))
            # No need to cache these images since they're only used once per epoch (they're unique for each pair of images)
            se_rmatches_img1 = self.transform(self.loader(se_rmatches_img1_fn).convert('L'))
            se_matches_img1 = self.transform(self.loader(se_matches_img1_fn).convert('L'))
            pe_img1 = self.transform(self.loader(pe_img1_fn).convert('L'))
            pe_mask_img1 = self.transform(self.loader(pe_mask_img1_fn).convert('L'))
            pe_warped_img1 = self.transform(self.loader(pe_warped_img1_fn))
            se_rmatches_img2 = self.transform(self.loader(se_rmatches_img2_fn).convert('L'))
            se_matches_img2 = self.transform(self.loader(se_matches_img2_fn).convert('L'))
            pe_img2 = self.transform(self.loader(pe_img2_fn).convert('L'))
            pe_mask_img2 = self.transform(self.loader(pe_mask_img2_fn).convert('L'))
            pe_warped_img2 = self.transform(self.loader(pe_warped_img2_fn))


            data.append([self.dsets[i].tolist(), self.fns[i,0].tolist(), self.fns[i,1].tolist(), self.R11s[i], self.R12s[i], self.R13s[i], \
                self.R21s[i], self.R22s[i], self.R23s[i], self.R31s[i], self.R32s[i], self.R33s[i], \
                self.num_rmatches[i], self.num_matches[i], self.spatial_entropy_1_8x8[i], \
                self.spatial_entropy_2_8x8[i], self.spatial_entropy_1_16x16[i], self.spatial_entropy_2_16x16[i], \
                self.pe_histogram[i].reshape((-1,1)), self.pe_polygon_area_percentage[i], self.nbvs_im1[i], self.nbvs_im2[i], \
                self.te_histogram[i].reshape((-1,1)), self.ch_im1[i].reshape((-1,1)), self.ch_im2[i].reshape((-1,1)), \
                self.vt_rank_percentage_im1_im2[i], self.vt_rank_percentage_im2_im1[i], \
                self.sq_rank_scores_mean[i], self.sq_rank_scores_min[i], self.sq_rank_scores_max[i], self.sq_distance_scores[i], \
                self.labels[i], img1, img2, se_rmatches_img1, se_rmatches_img2, se_matches_img1, se_matches_img2, pe_img1, pe_img2, pe_mask_img1, pe_mask_img2, \
                pe_warped_img1, pe_warped_img2
                ])
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
                labels, img1, img2, se_rmatches_img1, se_rmatches_img2, se_matches_img1, se_matches_img2, pe_img1, pe_img2, pe_mask_img1, pe_mask_img2, \
                pe_warped_img1, pe_warped_img2 = sample

            _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, \
                _num_rmatches, _num_matches, _spatial_entropy_1_8x8, _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, \
                _pe_histogram, _pe_polygon_area_percentage, _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, \
                _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
                _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
                _labels, _img1, _img2, \
                _se_rmatches_img1, _se_rmatches_img2, _se_matches_img1, _se_matches_img2, _pe_img1, _pe_img2, _pe_mask_img1, _pe_mask_img2, \
                _pe_warped_img1, _pe_warped_img2 = \
                R11s.cuda(), R12s.cuda(), R13s.cuda(), R21s.cuda(), R22s.cuda(), R23s.cuda(), R31s.cuda(), R32s.cuda(), R33s.cuda(), \
                num_rmatches.cuda(), num_matches.cuda(), spatial_entropy_1_8x8.cuda(), spatial_entropy_2_8x8.cuda(), spatial_entropy_1_16x16.cuda(), spatial_entropy_2_16x16.cuda(), \
                pe_histogram.cuda(), pe_polygon_area_percentage.cuda(), nbvs_im1.cuda(), nbvs_im2.cuda(), te_histogram.cuda(), ch_im1.cuda(), ch_im2.cuda(), \
                vt_rank_percentage_im1_im2.cuda(), vt_rank_percentage_im2_im1.cuda(), \
                sq_rank_scores_mean.cuda(), sq_rank_scores_min.cuda(), sq_rank_scores_max.cuda(), sq_distance_scores.cuda(), \
                labels.cuda(), img1.cuda(), img2.cuda(), \
                se_rmatches_img1.cuda(), se_rmatches_img2.cuda(), se_matches_img1.cuda(), se_matches_img2.cuda(), pe_img1.cuda(), pe_img2.cuda(), \
                pe_mask_img1.cuda(), pe_mask_img2.cuda(), pe_warped_img1.cuda(), pe_warped_img2.cuda()

            arg = [Variable(_R11s), Variable(_R12s), Variable(_R13s), Variable(_R21s), Variable(_R22s), Variable(_R23s), Variable(_R31s), Variable(_R32s), Variable(_R33s), \
                Variable(_num_rmatches), Variable(_num_matches), Variable(_spatial_entropy_1_8x8), Variable(_spatial_entropy_2_8x8), Variable(_spatial_entropy_1_16x16), Variable(_spatial_entropy_2_16x16), \
                Variable(_pe_histogram), Variable(_pe_polygon_area_percentage), Variable(_nbvs_im1), Variable(_nbvs_im2), Variable(_te_histogram), Variable(_ch_im1), Variable(_ch_im2), \
                Variable(_vt_rank_percentage_im1_im2), Variable(_vt_rank_percentage_im2_im1), \
                Variable(_sq_rank_scores_mean), Variable(_sq_rank_scores_min), Variable(_sq_rank_scores_max), Variable(_sq_distance_scores), \
                Variable(_labels), Variable(_img1), Variable(_img2), \
                Variable(_se_rmatches_img1), Variable(_se_rmatches_img2), Variable(_se_matches_img1), Variable(_se_matches_img2), Variable(_pe_img1), \
                Variable(_pe_img2), Variable(_pe_mask_img1), Variable(_pe_mask_img2), Variable(_pe_warped_img1), Variable(_pe_warped_img2) \
                ]
            

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

            # The samples are the same in the normal strategy, so we only need to do it once
            if opts['triplet-sampling-strategy'] == 'normal' or mode == 'test':
                break

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

        correct_counts = correct_counts + get_correct_counts(targets, y_predictions, False)
        if not arrays_initialized:
            all_dsets = dsets_b
            all_fns = fns_b
            all_targets = targets.data.cpu().numpy()
            all_predictions = y_predictions.data.cpu().numpy()
            all_num_rmatches = num_rmatches_b.data.cpu().numpy()
            arrays_initialized = True
        else:
            all_dsets = np.concatenate((all_dsets, dsets_b), axis=0)
            all_fns = np.concatenate((all_fns, fns_b), axis=0)
            all_targets = np.concatenate((all_targets, targets.data.cpu().numpy()), axis=0)
            all_predictions = np.concatenate((all_predictions, y_predictions.data.cpu().numpy()), axis=0)
            all_num_rmatches = np.concatenate((all_num_rmatches, num_rmatches_b.data.cpu().numpy()), axis=0)

        if (batch_idx + 1) % opts['log_interval'] == 0:
            if opts['triplet-sampling-strategy'] == 'normal' or mode == 'test':
                num_tests = (batch_idx + 1) * opts['batch_size'] # only one sample
            else:
                num_tests = (batch_idx + 1) * 2 * opts['batch_size'] # positive and negative samples
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
        if (epoch + 1) % 1 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
               '{}/checkpoint_{}.pth'.format(run_dir, epoch))
    else:
        logger.log_value('TEST-ACCURACY', accuracy)
        print ('{} Epoch: {}  Correct: {}  Accuracy: {}\n'.format(mode.upper(), epoch, correct_counts, \
            round(accuracy, 2)))

    # print ('-'*100)
    # score_thresholds = np.linspace(0, 1.0, 21)
    # rmatches_thresholds = np.linspace(18, 38, 21)
    # for i,t in enumerate(score_thresholds):
    #     thresholded_scores_predictions = np.copy(all_predictions)
    #     thresholded_scores_predictions[all_predictions[:,1] >= t, 1] = 1
    #     thresholded_scores_predictions[all_predictions[:,1] >= t, 0] = 0
    #     thresholded_scores_correct_counts = get_correct_counts(all_targets, thresholded_scores_predictions, False)

    #     # print (all_num_rmatches)
    #     thresholded_rmatches_predictions = np.copy(all_num_rmatches)
    #     thresholded_rmatches_predictions[all_num_rmatches < rmatches_thresholds[i]] = 0
    #     thresholded_rmatches_predictions[all_num_rmatches >= rmatches_thresholds[i]] = 1
    #     thresholded_rmatches_correct_counts = get_correct_counts(all_targets, thresholded_rmatches_predictions, False)

    #     num_tests = 2 * len(data_loader.dataset) # positive and negative samples
    #     thresholded_scores_accuracy = thresholded_scores_correct_counts*100.0/num_tests
    #     thresholded_rmatches_accuracy = thresholded_rmatches_correct_counts*100.0/num_tests

    #     print ('\t{} Epoch: {}     Classifier - Correct: {}  Accuracy: {}  Threshold: {}     Baseline - Correct: {}  Accuracy: {}  Threshold: {}'.format(\
    #         mode.upper(), epoch, \
    #         thresholded_scores_correct_counts, round(thresholded_scores_accuracy, 2), t, \
    #         thresholded_rmatches_correct_counts, round(thresholded_rmatches_accuracy, 2), rmatches_thresholds[i])
    #     )


    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])

    # plt.clf()
    auc, _ = matching_classifiers.calculate_dataset_auc(all_predictions[:,1], all_targets, color='green', ls='solid')
    _, _, _, _, _, auc_per_image_mean, _ = matching_classifiers.calculate_per_image_mean_auc(all_dsets, all_fns, all_predictions[:,1], all_targets)
    _, _, _, _, mean_precision_per_image = matching_classifiers.calculate_per_image_precision_top_k(all_dsets, all_fns, all_predictions[:,1], all_targets)
    # plt.clf()
    auc_baseline, _ = matching_classifiers.calculate_dataset_auc(all_num_rmatches, all_targets, color='green', ls='solid')
    _, _, _, _, _, auc_per_image_mean_baseline, _ = matching_classifiers.calculate_per_image_mean_auc(all_dsets, all_fns, all_num_rmatches, all_targets)
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

    test_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg, opts, transform=None),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    run_dir = os.path.join(opts['convnet_log_dir'], \
        'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}-is-{}-mlp-layer-size-{}-use-images-{}'.format(\
            opts['optimizer'], \
            opts['batch_size'], \
            opts['lr'], \
            opts['experiment'], \
            opts['loss'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            opts['image_match_classifier_min_match'], \
            opts['image_match_classifier_max_match'], \
            opts['use_all_training_data'], \
            opts['use_small_weights'], \
            opts['model'], \
            opts['convnet_input_size'], \
            opts['mlp-layer-size'], \
            opts['convnet_use_images']
        )
    )
    logger = Logger(run_dir)
    epoch = 0

    scores = inference(test_loader, model, epoch, run_dir, logger, opts, mode='test', optimizer=None)

    return None, None, None, scores, None

def classify_convnet_image_match_initialization(train_loader, test_loader, run_dir, opts):
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
        'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}-is-{}-mlp-layer-size-{}-use-images-{}'.format(\
            opts['optimizer'], \
            opts['batch_size'], \
            opts['lr'], \
            opts['experiment'], \
            opts['loss'], \
            # opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            opts['image_match_classifier_min_match'], \
            opts['image_match_classifier_max_match'], \
            opts['use_all_training_data'], \
            opts['use_small_weights'], \
            opts['model'], \
            opts['convnet_input_size'], \
            opts['mlp-layer-size'], \
            opts['convnet_use_images']
        )
    )
    
    # logger_train = Logger('{}-{}'.format(run_dir, 'train'))
    # logger_test = Logger('{}-{}'.format(run_dir, 'test'))

    logger = Logger(run_dir)
    
    for epoch in range(start, end):
        training_scores = inference(train_loader, model, epoch, run_dir, logger, opts, mode='train', optimizer=optimizer)
        if (epoch + 1) % 1 == 0:
            _ = inference(test_loader, model, epoch, run_dir, logger, opts, mode='test', optimizer=None)
        logger.step()

    # scores = inference(train_loader, model, epoch, run_dir, logger, opts, mode='train', optimizer=optimizer)
    return model, training_scores

def classify_convnet_image_match_training(arg, arg_te):
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    run_dir = os.path.join(opts['convnet_log_dir'], \
        'run-opt-{}-bs-{}-lr-{}-exp-{}-loss-{}-triplet-sampling-{}-sample-inclass-{}-min-images-{}-max-images-{}-use-all-data-{}-use-small-weights-{}-model-{}-is-{}-mlp-layer-size-{}-use-images-{}'.format(\
            opts['optimizer'], \
            opts['batch_size'], \
            opts['lr'], \
            opts['experiment'], \
            opts['loss'], \
            # opts['use_image_features'], \
            opts['triplet-sampling-strategy'], \
            opts['sample-inclass'], \
            opts['image_match_classifier_min_match'], \
            opts['image_match_classifier_max_match'], \
            opts['use_all_training_data'], \
            opts['use_small_weights'], \
            opts['model'], \
            opts['convnet_input_size'], \
            opts['mlp-layer-size'], \
            opts['convnet_use_images']
        )
    )
    matching_classifiers.mkdir_p(run_dir)

    train_transform = tv.transforms.Compose([
        tv.transforms.Resize((opts['convnet_input_size'], opts['convnet_input_size'])),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Resize((opts['convnet_input_size'], opts['convnet_input_size'])),
        tv.transforms.ToTensor(),
    ])

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
    test_loader = torch.utils.data.DataLoader(
        ImageMatchingDataset(arg_te, opts, transform=test_transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    print ('#'*50 + ' Testing Data ' + '#'*50)
    print ('\tInliers: {}'.format(len(np.where(test_loader.dataset.labels == 1)[0])))
    print ('\tOutliers: {}'.format(len(np.where(test_loader.dataset.labels == 0)[0])))
    print ('\tTotal: {}'.format(test_loader.dataset.__len__()))
    print ('#'*110)


    model, training_scores = classify_convnet_image_match_initialization(train_loader, test_loader, run_dir, opts)
    return None, None, model, training_scores, None
