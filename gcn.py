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
import networkx as nx
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

from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from eval_metrics import ErrorRateAt95Recall
from tensorboard_logger import configure, log_value
from timeit import default_timer as timer
import torchvision.models as models

import matching_classifiers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


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


class GCN(nn.Module):

    def __init__(self, opts):
        super(GCN, self).__init__()
        self.name = 'GCN'
        self.opts = opts
        
        # self.gcn_ll = nn.Linear(923, 923)
        # self.gcn_layers = []
        # for i in range(0,4):
        #     gcn_layer = nn.Linear(923,923)
        #     self.gcn_layers.append(gcn_layer)

        __model = models.resnet18(pretrained=True)
        _feature_extraction_model = nn.Sequential(*list(__model.children())[:-1])
        _feature_extraction_model.cuda()
        self.image_feature_extractor = _feature_extraction_model

        d = 512
        self.gcn_layer_1 = nn.Sequential(
            nn.Linear(d, d)
        )
        self.gcn_layer_2 = nn.Sequential(
            nn.Linear(d, d)
        )
        self.gcn_layer_3 = nn.Sequential(
            nn.Linear(d, d)
        )
        self.gcn_layer_4 = nn.Sequential(
            nn.Linear(d, d)
        )
        self.gcn_layer_5 = nn.Sequential(
            nn.Linear(d, d)
        )
        self.relu = nn.Sequential(
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )
        
        
        self._initialize_weights()

    def forward(self, args):
        # features = []
        # R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
        #     num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
        #     pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
        #     vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
        #     sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores = args

        # # for i in range(0,len(num_rmatches)):
        #     # R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
        #     #     num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
        #     #     pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
        #     #     vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
        #     #     sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
        #     #     labels = args
        # print ('R11s: {}'.format(R11s.size()))

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

        # if a == 0:
        #     features = x
        # else:
        #     features = torch.cat((features, x), dim=0)
        
        
        # mean_features = torch.mean(features, dim=0)

        # print ('#'*100)
        # print (num_rmatches.size())
        # print (features.size())
        # print (mean_features.size())
        # print ('#'*100)
        # import sys; sys.exit(1)
        # print ('x: {}'.format(x.size()))

        x, images, node_indices, A, D_normalizer = args
        # print '='*100
        # print A.size()
        # # print A
        # print '#'*100
        # print x.size()
        # # print x
        # print '='*100
        # print images.size()
        # print '='*100
        # import sys; sys.exit(1)
        # print '='*100
        # for i,image in enumerate(images[0]):
        # print '='*100
        # print images.size()
        image_features = self.image_feature_extractor(images)
        # print image_features.size()
        # print '='*100
        feature_shape = image_features.size()
        # print '^'*100
        # print image_features
        node_features = image_features.view(1, feature_shape[0], feature_shape[1])

        # print node_features.size()
        # print '='*100
        # import sys; sys.exit(1)

        # result = self.gcn(x.type(torch.cuda.FloatTensor))
        # node_features = x.type(torch.cuda.FloatTensor)
        # node_features = x
        for i in range(0, 5):
            # self.gcn_layers[l](torch.bmm(A.type(torch.cuda.FloatTensor), node_features))
            # print '*'*100
            
            # print A
            # print '*'*100
            # print node_features.size()
            
            # print '*'*100
            
            # print relevant_node_features
            # print relevant_node_features.is_cuda
            
            # print '#'*100
            if i == 0:
                node_features_result = self.gcn_layer_1(node_features)
            elif i == 1:
                node_features_result = self.gcn_layer_2(node_features)
            elif i == 2:
                node_features_result = self.gcn_layer_3(node_features)
            elif i == 3:
                node_features_result = self.gcn_layer_4(node_features)
            elif i == 4:
                node_features_result = self.gcn_layer_5(node_features)

            node_features_result = torch.bmm(A, node_features_result)
            node_features_result = torch.bmm(D_normalizer, node_features_result)
            node_features_result = self.relu(node_features_result)
            node_features = node_features_result
        
       
        for n in node_indices[0].data.cpu().numpy():

            # pdb.set_trace()
            n1, n2 = n
            a = node_features[0][n1]
            b = node_features[0][n2]

            # x = toch

        result = self.mlp(node_features)
        # print product.size()
        # print '*'*100
        # result = self.gcn_layer_1(product)

        # print result.size()
        
        
        
        
        # print ('='*100)
        # print (result.size())
        # print ('='*100)
        # import sys; sys.exit(1)

           
        return result

    def _initialize_weights(self):
        for m in self.modules():
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

class MatchGraphDataset(data.Dataset):
    def filter_scores(self, fns, scores):
        filtered_scores = {}
        for img1, img2 in fns:
            if img1 not in filtered_scores:
                filtered_scores[img1] = {}
            # if img2 not in filtered_scores:
            #     filtered_scores[img2] = {}

            filtered_scores[img1][img2] = scores[img1][img2]
            # filtered_scores[img2][img1] = scores[img2][img1]
        return filtered_scores

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
            self.labels, self.train, self.model, self.options = arg

        self.transform = transform
        self.loader = loader
        self.adjacency_matrices_rm = {}
        self.adjacency_matrices_te = {}
        self.D_inv_rm = {}
        self.D_inv_te = {}
        # self.graphs = {}
        self.feature_indices = {}
        self.feature_indices_sorted = {}
        self.neighbor_nodes = {}
        self.node_mapping = {}
        self.unique_dsets = list(set(self.dsets.tolist()))
        self.unique_fns = {}
        self.unique_fns_dsets = {}
        self.image_fns = {}

        for d, dset in enumerate(self.unique_dsets):
            dset_name = dset.split('/')[-1]
            data = dataset.DataSet(dset)
            ri = np.where(self.dsets == dset)[0].astype(np.int)
            dset_fns = self.fns[ri,:]
            
            dset_rmatches = classifier.rmatches_adapter(data)
            dset_tes, _ = classifier.triplet_errors_adapter(data)

            filtered_dset_rmatches = self.filter_scores(dset_fns, dset_rmatches)
            filtered_dset_tes = self.filter_scores(dset_fns, dset_tes)

            dset_unique_fns = np.array(list(set(np.concatenate((self.fns[ri,0], self.fns[ri,1])).tolist())))
            dset_name = dset.split('/')[-1]
            dset_unique_fns_dsets = np.tile(dset, (len(dset_unique_fns),))

            # if d == 0:
            self.unique_fns[dset] = dset_unique_fns
            self.unique_fns_dsets[dset] = dset_unique_fns_dsets
            # else:
            #     self.unique_fns = np.concatenate((self.unique_fns, dset_unique_fns))
            #     self.unique_fns_dsets = np.concatenate((self.unique_fns_dsets, dset_unique_fns_dsets))
            
            arg_rm = [data, dset_unique_fns, filtered_dset_rmatches, 'rm']
            arg_te = [data, dset_unique_fns, filtered_dset_tes, 'te']

            run_dir = os.path.join(opts['gcn_log_dir'], \
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
                    'GCN'
                )
            )
            g_rm_fn = os.path.join(run_dir,"{}-graph-rm.gpickle".format(dset_name))
            g_te_fn = os.path.join(run_dir,"{}-graph-te.gpickle".format(dset_name))
            g_inv_fn = os.path.join(run_dir,"{}-graph-inverted.gpickle".format(dset_name))
            if os.path.exists(g_rm_fn) and os.path.exists(g_te_fn):
                G_rm = nx.read_gpickle(g_rm_fn)
                G_te = nx.read_gpickle(g_te_fn)
                G_inv = nx.read_gpickle(g_inv_fn)
            else:
                G_rm = formulate_graphs.formulate_graph(arg_rm)
                G_te = formulate_graphs.formulate_graph(arg_te)
                G_inv = formulate_graphs.invert_graph(G_rm)
                nx.write_gpickle(G_rm, g_rm_fn)
                nx.write_gpickle(G_te, g_te_fn)
                nx.write_gpickle(G_inv, g_inv_fn)

            A_rm = nx.adjacency_matrix(G_rm, nodelist=sorted(G_rm.nodes()), weight='weight').todense()
            A_te = nx.adjacency_matrix(G_te, nodelist=sorted(G_te.nodes()), weight='weight').todense()
            I = np.matrix(np.eye(A_rm.shape[0]))
            A_rm = A_rm + I
            A_te = A_te + I

            D_rm = np.array(np.sum(A_rm, axis=0))[0]
            D_rm = np.matrix(np.diag(D_rm))

            D_te = np.array(np.sum(A_te, axis=0))[0]
            D_te = np.matrix(np.diag(D_te))
            
            # self.feature_indices[dset] = {}
            self.feature_indices_sorted[dset] = []
            # self.neighbor_nodes[dset] = {}
            
            self.node_mapping[dset] = {}
            for i,n1 in enumerate(sorted(G_rm.nodes())):
                self.node_mapping[dset][n1] = i
                for j,n2 in enumerate(sorted(G_rm.nodes())):
                    if j <= i:
                        continue
                    f_ri = np.where(\
                        (dset_fns[:,0] == n1) & (dset_fns[:,1] == n2) | \
                        (dset_fns[:,1] == n1) & (dset_fns[:,0] == n2) \
                        )[0]
                    if len(f_ri) == 0:
                        continue
                    self.feature_indices_sorted[dset].append(ri[f_ri])

            self.image_fns[dset] = []
            for i,n1 in enumerate(sorted(G_rm.nodes())):
                img_fn = os.path.join(dset, 'images', n1)
                self.image_fns[dset].append(img_fn)

            # self.feature_indices_sorted[dset] = np.array(self.feature_indices_sorted[dset])
            # self.neighbor_nodes[dset][n] = []

            # A_triplet = np.matrix(np.eye(A.shape[0]))
            # nns = G_inv.neighbors(n)
            # n_index = self.node_mapping[dset][n]
            # for neighbor_n in nns:
            #     neighbor_n1, neighbor_n2 = neighbor_n.split('---')
            #     if n1 == neighbor_n1 or n1 == neighbor_n2:
            #         triplet_node = n2
            #         # neighbor_n1, neighbor_n2 = neighbor_n2, neighbor_n1
            #     elif n2 == neighbor_n1 or n2 == neighbor_n2:
            #         triplet_node = n1

            #     for second_neighbor_n in G_inv.neighbors(neighbor_n):
            #         if second_neighbor_n == n:
            #             continue
            #         second_neighbor_n1, second_neighbor_n2 = second_neighbor_n.split('---')
            #         if triplet_node == second_neighbor_n1 or triplet_node == second_neighbor_n2:
            #             self.neighbor_nodes[dset][n].append([n, neighbor_n, second_neighbor_n])

            #             A_triplet[n_index, self.node_mapping[dset][neighbor_n]] = 1.0
            #             A_triplet[n_index, self.node_mapping[dset][second_neighbor_n]] = 1.0


            # D_triplet = np.array(np.sum(A_triplet, axis=0))[0]
            # D_triplet = np.matrix(np.diag(D_triplet))

            self.adjacency_matrices_rm[dset] = A_rm
            self.D_inv_rm[dset] = D_rm**-1
            self.adjacency_matrices_te[dset] = A_te
            self.D_inv_te[dset] = D_te**-1

            # print '$'*100
            # print len(ri)
            # print len(self.feature_indices_sorted[dset])
            # print self.feature_indices_sorted[dset]
            # print '$'*100
        # import sys; sys.exit(1)
            # self.adjacency_matrices[dset] = A_triplet
            # self.D_inv[dset] = D_triplet**-1
            # self.adjacency_matrices[dset] = np.matrix(np.eye(A.shape[0]))
            # self.D_inv[dset] = np.matrix(np.eye(A.shape[0]))

    def __getitem__(self, index):
        dset = self.unique_dsets[index]
        for c, i in enumerate(self.feature_indices_sorted[dset]):
            datum = np.concatenate((
                self.R11s[i].flatten(), self.R12s[i].flatten(), self.R13s[i].flatten(), \
                self.R21s[i].flatten(), self.R22s[i].flatten(), self.R23s[i].flatten(), self.R31s[i].flatten(), self.R32s[i].flatten(), self.R33s[i].flatten(), \
                self.num_rmatches[i].flatten(), self.num_matches[i].flatten(), self.spatial_entropy_1_8x8[i].flatten(), \
                self.spatial_entropy_2_8x8[i].flatten(), self.spatial_entropy_1_16x16[i].flatten(), self.spatial_entropy_2_16x16[i].flatten(), \
                self.pe_histogram[i].flatten(), self.pe_polygon_area_percentage[i].flatten(), self.nbvs_im1[i].flatten(), self.nbvs_im2[i].flatten(), \
                self.te_histogram[i].flatten(), self.ch_im1[i].flatten(), self.ch_im2[i].flatten(), \
                self.vt_rank_percentage_im1_im2[i].flatten(), self.vt_rank_percentage_im2_im1[i].flatten(), \
                self.sq_rank_scores_mean[i].flatten(), self.sq_rank_scores_min[i].flatten(), self.sq_rank_scores_max[i].flatten(), self.sq_distance_scores[i].flatten()
                ))
            # print '('*100
            # print self.fns[i,:].reshape((-1,2))
            # print ')'*100
            if c == 0:
                data = datum.reshape((1,-1))
                fns = self.fns[i,:].reshape((-1,2))
                labels = self.labels[i].flatten().reshape((1,-1))
                num_rmatches = self.num_rmatches[i].flatten().reshape((1,-1))
                node_indices = np.concatenate((\
                    np.array(self.node_mapping[dset][fns[0][0]]).reshape((-1,1)), np.array(self.node_mapping[dset][fns[0][1]]).reshape((-1,1)) \
                    ), axis=1)
            else:
                current_fns = self.fns[i,:].reshape((-1,2))
                data = np.concatenate((data, datum.reshape((1,-1))), axis=0)
                fns = np.concatenate((fns, current_fns), axis=0)
                labels = np.concatenate((labels, self.labels[i].flatten().reshape((1,-1))), axis=0)
                num_rmatches = np.concatenate((num_rmatches, self.num_rmatches[i].flatten().reshape((1,-1))), axis=0)
                node_indices = np.concatenate((node_indices, np.concatenate((\
                    np.array(self.node_mapping[dset][current_fns[0][0]]).reshape((-1,1)), np.array(self.node_mapping[dset][current_fns[0][1]]).reshape((-1,1)) \
                    ), axis=1)), axis=0)

        # print '#'*100
        # print (self.adjacency_matrices[dset].tolist())
        # print '#'*100
        # metadata = {'fns1': , 'fns2':, 'dset': }
        images = []
        for img_fn in self.image_fns[dset]:
            images.append(self.transform(self.loader(img_fn)).tolist())

        # print '#'*100
        # print '#'*100
        # print node_indices
        # print '@'*100
        # print '@'*100
        # import sys; sys.exit(1)
        # print len(images)
        # print '-'*100
        # print images[0].shape
        # print '#'*100
        # print '#'*100
        # import sys; sys.exit(1)
        return [ \
            self.D_inv_rm[dset].tolist(), \
            self.adjacency_matrices_rm[dset].tolist(), \
            self.D_inv_te[dset].tolist(), \
            self.adjacency_matrices_te[dset].tolist(), \
            images, \
            node_indices, \
            data, \
            labels.flatten(), \
            num_rmatches.flatten(), \
            fns[:,0].flatten().tolist(), \
            fns[:,1].flatten().tolist(), \
            dset \
        ]


    def __len__(self):
        return len(self.unique_dsets)
    # def __getitem__(self, index):
    #     # indices = []
    #     im1,im2 = self.fns[index]
    #     dset = self.dsets[index]
    #     node_name = '{}---{}'.format(im1,im2)
    #     if node_name not in self.feature_indices[dset]:
    #         node_name = '{}---{}'.format(im2,im1)

    #     if len(self.neighbor_nodes[dset][node_name]) > 0:
    #         random_triplet = random.randint(0,len(self.neighbor_nodes[dset][node_name])-1)
    #         # neighbor_nodes = self.neighbor_nodes[dset][node_name][random_triplet]
    #         valid_triplet = True
    #         for n in self.neighbor_nodes[dset][node_name][random_triplet]:
    #             node_index = self.feature_indices[dset][n]
    #             if self.num_rmatches[node_index] < 30:
    #                 valid_triplet = False
    #         if valid_triplet:
    #             neighbor_nodes = self.neighbor_nodes[dset][node_name][random_triplet]
    #         else:
    #             neighbor_nodes = [node_name, node_name, node_name]    
    #     else:
    #         neighbor_nodes = [node_name, node_name, node_name]
        
    #     indices = []
    #     for nnn in neighbor_nodes:
    #         indices.append(self.feature_indices[dset][nnn])

    #     # print '$'*100
    #     # print 'indices: {}'.format(len(indices))
    #     # print '$'*100
    #     # data = []
    #     for c, i in enumerate(indices):
    #         # print ('self.ch_im2[i]: {}'.format(self.ch_im2[i].flatten().shape))
    #         datum = np.concatenate((
    #             self.R11s[i].flatten(), self.R12s[i].flatten(), self.R13s[i].flatten(), \
    #             self.R21s[i].flatten(), self.R22s[i].flatten(), self.R23s[i].flatten(), self.R31s[i].flatten(), self.R32s[i].flatten(), self.R33s[i].flatten(), \
    #             self.num_rmatches[i].flatten(), self.num_matches[i].flatten(), self.spatial_entropy_1_8x8[i].flatten(), \
    #             self.spatial_entropy_2_8x8[i].flatten(), self.spatial_entropy_1_16x16[i].flatten(), self.spatial_entropy_2_16x16[i].flatten(), \
    #             self.pe_histogram[i].flatten(), self.pe_polygon_area_percentage[i].flatten(), self.nbvs_im1[i].flatten(), self.nbvs_im2[i].flatten(), \
    #             self.te_histogram[i].flatten(), self.ch_im1[i].flatten(), self.ch_im2[i].flatten(), \
    #             self.vt_rank_percentage_im1_im2[i].flatten(), self.vt_rank_percentage_im2_im1[i].flatten(), \
    #             self.sq_rank_scores_mean[i].flatten(), self.sq_rank_scores_min[i].flatten(), self.sq_rank_scores_max[i].flatten(), self.sq_distance_scores[i].flatten()
    #             ))
    #         # fns = np.c
    #         # print 'datum: {}'.format(datum.shape)
    #         if c == 0:
    #             data = datum.reshape((1,-1))
    #             fns = self.fns[i,:]
    #             # labels = self.labels[i].reshape((1,-1))
    #             labels = self.labels[i].flatten()
    #             # num_rmatches = self.num_rmatches[i].reshape((1,-1))
    #             num_rmatches = self.num_rmatches[i].flatten()
    #         else:
    #             # print '{} / {}'.format(data.shape, datum.reshape((1,-1)).shape)
    #             data = np.concatenate((data, datum.reshape((1,-1))), axis=0)
    #             # fns = np.concatenate((fns, self.fns[i,:]), axis=0)
    #             # labels = np.concatenate((labels, self.labels[i].reshape((1,-1))), axis=0)

    #         # data.append([self.dsets[i].tolist(), self.fns[i,0].tolist(), self.fns[i,1].tolist(), self.R11s[i], self.R12s[i], self.R13s[i], \
    #         #     self.R21s[i], self.R22s[i], self.R23s[i], self.R31s[i], self.R32s[i], self.R33s[i], \
    #         #     self.num_rmatches[i], self.num_matches[i], self.spatial_entropy_1_8x8[i], \
    #         #     self.spatial_entropy_2_8x8[i], self.spatial_entropy_1_16x16[i], self.spatial_entropy_2_16x16[i], \
    #         #     self.pe_histogram[i].reshape((-1,1)), self.pe_polygon_area_percentage[i], self.nbvs_im1[i], self.nbvs_im2[i], \
    #         #     self.te_histogram[i].reshape((-1,1)), self.ch_im1[i].reshape((-1,1)), self.ch_im2[i].reshape((-1,1)), \
    #         #     self.vt_rank_percentage_im1_im2[i], self.vt_rank_percentage_im2_im1[i], \
    #         #     self.sq_rank_scores_mean[i], self.sq_rank_scores_min[i], self.sq_rank_scores_max[i], self.sq_distance_scores[i], \
    #         #     self.labels[i]])

    #     # mean_data = np.mean(data, axis=0)

    #     # print ('Node neighbors: {}'.format(len(neighbor_nodes)))
    #     # print ('Mean Data: {}'.format(mean_data.shape))
    #     # print ('fns: {} labels: {}'.format(fns.shape, labels.shape))

    #     # print '!'*100
    #     # print mean_data.shape
    #     # print data.shape
    #     # print '!'*100
    #     return [data, labels, num_rmatches, fns[:,0].flatten()[0], fns[:,1].flatten()[0], dset]

    # def __len__(self):
    #     return len(self.fns)

def get_correct_counts(target, y_pred, verbose):
    try:
        # Score prediction when y_pred and target are on gpu
        result = np.sum(np.equal(target.data.cpu().numpy().flatten(), np.argmax(y_pred.data.cpu().numpy(),axis=1)))
        # print ('target: {}  y_pred: {}  y_pred argmax: {} result: {}'.format(target.data.cpu().numpy().flatten(), y_pred.data.cpu().numpy(), \
        #     np.argmax(y_pred.data.cpu().numpy(),axis=1), \
        #     np.equal(target.data.cpu().numpy(), np.argmax(y_pred.data.cpu().numpy(),axis=1))))
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

    for batch_idx, batch in enumerate(data_loader):
        # if len(p_sample) == 0 or len(n_sample) == 0:
        #     # not a valid triplet (no negative sample for anchor image)
        #     continue
        D_inv_rm, adjacency_matrix_rm, D_inv_te, adjacency_matrix_te, imgs, node_indices, data, labels, num_rmatches, im1s, im2s, dset = batch
        # im1s, im2s, dsets = metadata['fns1'], metadata['fns2'], metadata['dset']
        # node_label = None
        if mode == 'train':
            optimizer.zero_grad()


        # R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
        #     num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
        #     pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
        #     vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
        #     sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores = data

        # _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, \
        #     _num_rmatches, _num_matches, _spatial_entropy_1_8x8, _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, \
        #     _pe_histogram, _pe_polygon_area_percentage, _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, \
        #     _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
        #     _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores = \
        #     R11s.cuda(), R12s.cuda(), R13s.cuda(), R21s.cuda(), R22s.cuda(), R23s.cuda(), R31s.cuda(), R32s.cuda(), R33s.cuda(), \
        #     num_rmatches.cuda(), num_matches.cuda(), spatial_entropy_1_8x8.cuda(), spatial_entropy_2_8x8.cuda(), spatial_entropy_1_16x16.cuda(), spatial_entropy_2_16x16.cuda(), \
        #     pe_histogram.cuda(), pe_polygon_area_percentage.cuda(), nbvs_im1.cuda(), nbvs_im2.cuda(), te_histogram.cuda(), ch_im1.cuda(), ch_im2.cuda(), \
        #     vt_rank_percentage_im1_im2.cuda(), vt_rank_percentage_im2_im1.cuda(), \
        #     sq_rank_scores_mean.cuda(), sq_rank_scores_min.cuda(), sq_rank_scores_max.cuda(), sq_distance_scores.cuda()

        # args = [Variable(_R11s), Variable(_R12s), Variable(_R13s), Variable(_R21s), Variable(_R22s), Variable(_R23s), Variable(_R31s), Variable(_R32s), Variable(_R33s), \
        #     Variable(_num_rmatches), Variable(_num_matches), Variable(_spatial_entropy_1_8x8), Variable(_spatial_entropy_2_8x8), Variable(_spatial_entropy_1_16x16), Variable(_spatial_entropy_2_16x16), \
        #     Variable(_pe_histogram), Variable(_pe_polygon_area_percentage), Variable(_nbvs_im1), Variable(_nbvs_im2), Variable(_te_histogram), Variable(_ch_im1), Variable(_ch_im2), \
        #     Variable(_vt_rank_percentage_im1_im2), Variable(_vt_rank_percentage_im2_im1), \
        #     Variable(_sq_rank_scores_mean), Variable(_sq_rank_scores_min), Variable(_sq_rank_scores_max), Variable(_sq_distance_scores)]
        
        # print '#'*100
        # A = torch.tensor(adjacency_matrix).float()
        # print '#'*100
        # print dset
        # print data.shape
        # print '-'*100
        # print np.array(adjacency_matrix_rm).shape
        # print '#'*100

        A = torch.from_numpy(np.array(adjacency_matrix_rm).reshape((-1, len(adjacency_matrix_rm), len(adjacency_matrix_rm)))).float()
        D_normalizer = torch.from_numpy(np.array(D_inv_rm).reshape((-1, len(adjacency_matrix_rm), len(adjacency_matrix_rm)))).float()

        # print '#'*100
        # print '#'*100
        # print imgs.size()
        # print '$'*100
        # print len(imgs)
        # print '-'*100
        # print imgs[0].shape
        # print '#'*100
        # print '#'*100
        # print imgs.shape
        
        # print np.array(imgs).shape

        # images = torch.from_numpy(np.array(imgs).reshape((-1, len(imgs), 3, 224, 224))).float()
        # images = imgs
        images = torch.from_numpy(np.array(imgs)).float()

        # print images.size()
        # import sys; sys.exit(1)
        # print A.size()
        # print D_normalizer.size()
        # print '#'*100
        # import sys; sys.exit(1)
        # print dset
        # print im1s[0:10]
        # print im2s[0:10]
        # print data[0,0:10,9]

        args = [ \
            Variable(data.cuda()).type(torch.cuda.FloatTensor), \
            Variable(images.cuda()).type(torch.cuda.FloatTensor), \
            Variable(node_indices.cuda()).type(torch.cuda.FloatTensor), \
            Variable(A.cuda()).type(torch.cuda.FloatTensor), \
            Variable(D_normalizer.cuda()).type(torch.cuda.FloatTensor) \
        ]


        # print args.size()
        # print '#'*100
        # import sys; sys.exit(1)

        # args = []
        # As = []
        # for j, datum in enumerate(data):
        #     print ('data shape: {}  datum shape: {} labels shape: {}'.format(data.shape, datum.shape, labels.shape))
        #     dsets, im1s, im2s, R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, \
        #         num_rmatches, num_matches, spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
        #         pe_histogram, pe_polygon_area_percentage, nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, \
        #         vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
        #         sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
        #         labels = datum

        #     _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, \
        #         _num_rmatches, _num_matches, _spatial_entropy_1_8x8, _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, \
        #         _pe_histogram, _pe_polygon_area_percentage, _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, \
        #         _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
        #         _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
        #         _labels = \
        #         R11s.cuda(), R12s.cuda(), R13s.cuda(), R21s.cuda(), R22s.cuda(), R23s.cuda(), R31s.cuda(), R32s.cuda(), R33s.cuda(), \
        #         num_rmatches.cuda(), num_matches.cuda(), spatial_entropy_1_8x8.cuda(), spatial_entropy_2_8x8.cuda(), spatial_entropy_1_16x16.cuda(), spatial_entropy_2_16x16.cuda(), \
        #         pe_histogram.cuda(), pe_polygon_area_percentage.cuda(), nbvs_im1.cuda(), nbvs_im2.cuda(), te_histogram.cuda(), ch_im1.cuda(), ch_im2.cuda(), \
        #         vt_rank_percentage_im1_im2.cuda(), vt_rank_percentage_im2_im1.cuda(), \
        #         sq_rank_scores_mean.cuda(), sq_rank_scores_min.cuda(), sq_rank_scores_max.cuda(), sq_distance_scores.cuda(), \
        #         labels.cuda()

        #     args.append([Variable(_R11s), Variable(_R12s), Variable(_R13s), Variable(_R21s), Variable(_R22s), Variable(_R23s), Variable(_R31s), Variable(_R32s), Variable(_R33s), \
        #         Variable(_num_rmatches), Variable(_num_matches), Variable(_spatial_entropy_1_8x8), Variable(_spatial_entropy_2_8x8), Variable(_spatial_entropy_1_16x16), Variable(_spatial_entropy_2_16x16), \
        #         Variable(_pe_histogram), Variable(_pe_polygon_area_percentage), Variable(_nbvs_im1), Variable(_nbvs_im2), Variable(_te_histogram), Variable(_ch_im1), Variable(_ch_im2), \
        #         Variable(_vt_rank_percentage_im1_im2), Variable(_vt_rank_percentage_im2_im1), \
        #         Variable(_sq_rank_scores_mean), Variable(_sq_rank_scores_min), Variable(_sq_rank_scores_max), Variable(_sq_distance_scores), \
        #         Variable(_labels)])
        #     # As.append(Variable(A.cuda()))
        #     As.append(A)
        #     print ('num_rmatches: {}'.format(len(num_rmatches)))
        #     if j == 0:
        #         node_label = _labels

        # print ('#'*100)
        # print ('#'*100)
        # print (len(data))
        # print ('Data iterations: {}'.format(j))
        y_prediction = model(args)
        
        # print (num_rmatches.shape)
        # print (node_label.shape)
        # print (y_prediction.shape)
        
        target = Variable(labels.type(torch.cuda.LongTensor))

        # print ('='*100)
        # print ('='*100)
        # print labels.size()
        # print y_prediction.size()
        # print ('='*100)
        # print ('='*100)
        # import sys; sys.exit(1)

        if not arrays_initialized:
            arrays_initialized = True
            all_predictions = y_prediction[0]
            # positive_predictions = y_prediction
            # positive_targets = target
            all_targets = target[0]
            all_num_rmatches = Variable(num_rmatches[0])
            # print ('im1s: {} im2s: {}'.format(np.array(im1s).reshape((-1,1)).shape, np.array(im2s).reshape((-1,1)).shape))
            all_fns = np.concatenate((np.array(im1s).reshape((-1,1)), np.array(im2s).reshape((-1,1))), axis=1)
            all_dsets = np.array(dset)
        else:
            all_predictions = torch.cat((all_predictions, y_prediction[0]))
            # negative_predictions = y_prediction
            # negative_targets = target
            all_targets = torch.cat((all_targets, target[0]), dim=0)
            all_num_rmatches = torch.cat((all_num_rmatches, Variable(num_rmatches[0])))

            # print 'fns: {} / {}  dset: {} / {}'.format(all_fns.shape, np.concatenate((np.array(im1s).reshape((-1,1)), np.array(im2s).reshape((-1,1))), axis=1).shape, all_dsets.shape, np.array(dsets).shape)
            # print 'targets: {} / {}'.format(all_targets.shape, target.shape)
            all_fns = np.concatenate((all_fns, np.concatenate((np.array(im1s).reshape((-1,1)), np.array(im2s).reshape((-1,1))), axis=1)), axis=0)
            all_dsets = np.concatenate((all_dsets, np.array(dset)), axis=0)
            # dsets_b = np.concatenate((dsets_b, np.array(dsets)), axis=0)

        if mode == 'train':
            reformatted_targets = all_targets.type(torch.cuda.FloatTensor).clone()
            reformatted_targets[reformatted_targets <= 0] = -1

            loss = cross_entropy_loss(y_prediction[0], target[0].view(-1,))

            loss.backward()
            optimizer.step()

        correct_counts = correct_counts + get_correct_counts(target[0], y_prediction[0], False)

        # if not arrays_initialized:
        #     all_dsets = dsets_b
        #     all_fns = fns_b
        #     all_targets = targets.data.cpu().numpy()
        #     all_predictions = y_predictions.data.cpu().numpy()
        #     all_num_rmatches = num_rmatches_b.data.cpu().numpy()
        #     arrays_initialized = True
        #     # all_im1s = im1_fn
        #     # all_im2s = im2_fn
        # else:
        #     all_dsets = np.concatenate((all_dsets, dsets_b), axis=0)
        #     all_fns = np.concatenate((all_fns, fns_b), axis=0)
        #     all_targets = np.concatenate((all_targets, targets.data.cpu().numpy()), axis=0)
        #     all_predictions = np.concatenate((all_predictions, y_predictions.data.cpu().numpy()), axis=0)
        #     all_num_rmatches = np.concatenate((all_num_rmatches, num_rmatches_b.data.cpu().numpy()), axis=0)
        #     # all_im1s = np.concatenate((all_im1s, im1_fn), axis=0)
        #     # all_im2s = np.concatenate((all_im2s, im2_fn), axis=0)
        # print (correct_counts)
        # print (target.shape)
        # print (y_prediction.shape)
        # print ('$'*100)
        if (batch_idx + 1) % opts['log_interval'] == 0:
            num_tests = data_loader.dataset.__len__()
            accuracy = correct_counts*100.0/((batch_idx + 1) * opts['batch_size'])
            # print (batch_idx)
            # print (correct_counts)
            # print ((batch_idx + 1) * opts['batch_size'])
            # import sys; sys.exit(1)
            if mode == 'train':
                print(
                    '{} Epoch: {} Accuracy: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        mode.upper(), epoch, accuracy, (batch_idx + 1) * opts['batch_size'], data_loader.dataset.__len__(),
                        100.0 * (batch_idx + 1) * opts['batch_size'] / data_loader.dataset.__len__(),
                        loss.data[0]))
            else:
                print(
                    '{} Epoch: {} Accuracy: {} [{}/{} ({:.0f}%)]'.format(
                        mode.upper(), epoch, accuracy, (batch_idx + 1) * opts['batch_size'], data_loader.dataset.__len__(),
                        100.0 * (batch_idx + 1) * opts['batch_size'] / data_loader.dataset.__len__()))

        if mode == 'train':
            cum_loss = cum_loss + loss.data[0]

        # print all_dsets.reshape((-1,))
        # print all_dsets.reshape((-1,)).shape

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
    # score_thresholds = np.linspace(0, 1.0, 21)
    # rmatches_thresholds = np.linspace(18, 38, 21)
    # for i,t in enumerate(score_thresholds):
    #     thresholded_scores_predictions = np.copy(all_predictions.data.cpu().numpy())
    #     thresholded_scores_predictions[all_predictions[:,1].data.cpu().numpy() >= t, 1] = 1
    #     thresholded_scores_predictions[all_predictions[:,1].data.cpu().numpy() >= t, 0] = 0
    #     thresholded_scores_correct_counts = get_correct_counts(all_targets.data.cpu().numpy().flatten(), thresholded_scores_predictions, False)

    #     # print (all_num_rmatches)
    #     thresholded_rmatches_predictions = np.copy(all_num_rmatches.data.cpu().numpy())
    #     thresholded_rmatches_predictions[all_num_rmatches.data.cpu().numpy() < rmatches_thresholds[i]] = 0
    #     thresholded_rmatches_predictions[all_num_rmatches.data.cpu().numpy() >= rmatches_thresholds[i]] = 1
    #     thresholded_rmatches_correct_counts = get_correct_counts(all_targets.data.cpu().numpy().flatten(), thresholded_rmatches_predictions, False)

    #     num_tests = data_loader.dataset.__len__()
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
    # print ('all_predictions: {}  all_targets: {}'.format(all_predictions.data.cpu().numpy().shape, all_targets.data.cpu().numpy().shape))
    conversion_t_s = timer()
    all_predictions_cpu = all_predictions[:,1].data.cpu().numpy()
    all_targets_cpu = all_targets.data.cpu().numpy()
    all_num_rmatches_cpu = all_num_rmatches.data.cpu().numpy()
    conversion_t_e = timer()
    # print ('\t\tConversion time: {}'.format(round(conversion_t_e - conversion_t_s,3)))

    auc_s_t = timer()
    auc = matching_classifiers.calculate_dataset_auc(all_predictions_cpu, all_targets_cpu, color='green', ls='solid')
    auc_e_t = timer()
    aucpi_s_t = timer()
    _, _, _, _, auc_per_image_mean = matching_classifiers.calculate_per_image_mean_auc(all_dsets.reshape((-1,)), all_fns, all_predictions_cpu, all_targets_cpu)
    aucpi_e_t = timer()
    ppi_s_t = timer()
    _, _, _, _, mean_precision_per_image = matching_classifiers.calculate_per_image_precision_top_k(all_dsets.reshape((-1,)), all_fns, all_predictions_cpu, all_targets_cpu)
    ppi_e_t = timer()
    # plt.clf()
    auc_s_t_b = timer()
    auc_baseline = matching_classifiers.calculate_dataset_auc(all_num_rmatches_cpu, all_targets_cpu, color='green', ls='solid')
    auc_e_t_b = timer()
    aucpi_s_t_b = timer()
    _, _, _, _, auc_per_image_mean_baseline = matching_classifiers.calculate_per_image_mean_auc(all_dsets.reshape((-1,)), all_fns, all_num_rmatches_cpu, all_targets_cpu)
    aucpi_e_t_b = timer()
    ppi_s_t_b = timer()
    _, _, _, _, mean_precision_per_image_baseline = matching_classifiers.calculate_per_image_precision_top_k(all_dsets.reshape((-1,)), all_fns, all_num_rmatches_cpu, all_targets_cpu)
    ppi_e_t_b = timer()


    print ('\t{} Epoch: {}     Experiment: {} AUC: {} / {} / {}'.format(mode.upper(), epoch, opts['experiment'], \
        round(auc,3), round(auc_per_image_mean, 3), round(mean_precision_per_image, 3) \
        # round(auc_e_t - auc_s_t,3), round(aucpi_e_t - aucpi_s_t,3), round(ppi_e_t - ppi_s_t,3) \
        ))
    print ('\t{} Epoch: {}     Baseline: {} AUC: {} / {} / {}'.format(mode.upper(), epoch, 'Baseline', \
        round(auc_baseline, 3), round(auc_per_image_mean_baseline, 3), round(mean_precision_per_image_baseline, 3) \
        # round(auc_e_t_b - auc_s_t_b,3), round(aucpi_e_t_b - aucpi_s_t_b,3), round(ppi_e_t_b - ppi_s_t_b,3) \
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

def classify_gcn_image_match_inference(arg):
    model = arg[-2]
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    test_loader = torch.utils.data.DataLoader(
        MatchGraphDataset(arg, opts, transform=None),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    # run_dir = os.path.join(opts['gcn_log_dir'], \
    #     'run-optimizer-{}-batch_size-{}-lr-{}-model-{}-test'.format(opts['optimizer'], opts['batch_size'], opts['lr'], model.name))
    run_dir = os.path.join(opts['gcn_log_dir'], \
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

def classify_gcn_image_match_initialization(train_loader, test_loader, run_dir, opts):
    # instantiate model and initialize weights
    kwargs = {}
    model = GCN(opts, **kwargs)

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
    run_dir = os.path.join(opts['gcn_log_dir'], \
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

def classify_gcn_image_match_training(arg, arg_te):
    opts = arg[-1]
    kwargs = {'num_workers': opts['num_workers'], 'pin_memory': True}

    if not opts['opensfm_path'] in sys.path:
        sys.path.insert(1, opts['opensfm_path'])
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    from opensfm.commands import formulate_graphs
    global matching, classifier, dataset, formulate_graphs

    run_dir = os.path.join(opts['gcn_log_dir'], \
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
            'GCN'
        )
    )
    matching_classifiers.mkdir_p(run_dir)

    transform = tv.transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        MatchGraphDataset(arg, opts, transform=transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    print ('#'*50 + ' Training Data ' + '#'*50)
    print ('\tInliers: {}'.format(len(np.where(train_loader.dataset.labels == 1)[0])))
    print ('\tOutliers: {}'.format(len(np.where(train_loader.dataset.labels == 0)[0])))
    print ('\tTotal: {}'.format(train_loader.dataset.__len__()))
    print ('#'*100)

    opts = arg[-1]
    test_loader = torch.utils.data.DataLoader(
        MatchGraphDataset(arg_te, opts, transform=transform),
        batch_size=opts['batch_size'], shuffle=opts['shuffle'], **kwargs
        )

    print ('#'*50 + ' Testing Data ' + '#'*50)
    print ('\tInliers: {}'.format(len(np.where(test_loader.dataset.labels == 1)[0])))
    print ('\tOutliers: {}'.format(len(np.where(test_loader.dataset.labels == 0)[0])))
    print ('\tTotal: {}'.format(test_loader.dataset.__len__()))
    print ('#'*110)


    model, training_scores = classify_gcn_image_match_initialization(train_loader, test_loader, run_dir, opts)
    # return None, None, None, None, None
    return None, None, model, training_scores, None
