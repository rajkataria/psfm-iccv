#import gflags
import datetime
import os
import datetime
import glob
import gzip
import itertools
import time
import logging
import json
import math
import networkx as nx
import numpy as np
import pdb
import pickle
import sys
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import cv2, random
import sklearn
import scipy
import pprint
import socket

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectFromModel
from timeit import default_timer as timer
# from nn import classify_nn_image_match_training, classify_nn_image_match_inference
import convnet
import nn
import gcn

from argparse import ArgumentParser

def save_classifier(regr, name):
    name_ = name + '.pkl'
    joblib.dump(regr, name_)

def load_classifier(name):
    if name.split('.')[-1] != 'pkl':
        name_ = name + '.pkl'
    else:
        name_ = name
    return joblib.load(name_)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def calculate_dataset_auc(y, y_gt, color, ls):
    if ls == 'solid':
        width = 50
    else:
        width = 10

    precision, recall, threshs = sklearn.metrics.precision_recall_curve(y_gt, y)
    auc = sklearn.metrics.average_precision_score(y_gt, y)
    auc_roc = sklearn.metrics.roc_auc_score(y_gt, y)
    plt.step(recall, precision, color=color, alpha=0.2 * width,
        where='post')
    return auc, auc_roc

def calculate_per_image_mean_auc(dsets, fns, y, y_gt):

    # precision, recall, threshs = sklearn.metrics.precision_recall_curve(y_gt, y)
    # auc = sklearn.metrics.average_precision_score(y_gt, y)
    # plt.step(recall, precision, color=color, alpha=0.2 * width,
    #     where='post')
    # return auc
    a_dsets, a_fns, a_precision_recall_auc = [], [], []
    for dset in sorted(list(set(dsets))):
        ri = np.where(dsets == dset)[0]
        # print ('{}  /  {}'.format(dsets.shape, fns.shape))
        # print (ri)
        dset_fns = fns[ri]
        dset_y = y[ri]
        dset_y_gt = y_gt[ri]

        unique_dset_fns = sorted(list(set(np.concatenate((dset_fns[:,0], dset_fns[:,1])).tolist())))
        for f in unique_dset_fns:
            ri_ = np.where((dset_fns[:,0] == f) | (dset_fns[:,1] == f))[0]
            f_y = dset_y[ri_]
            f_y_gt = dset_y_gt[ri_]

            f_precision, f_recall, f_threshs = sklearn.metrics.precision_recall_curve(f_y_gt, f_y)
            f_auc = sklearn.metrics.average_precision_score(f_y_gt, f_y)
            
            # if len(f_y_gt) == np.sum(f_y_gt) or np.sum(f_y_gt) == 0:
            #     continue
            
            # f_auc_roc = sklearn.metrics.roc_auc_score(f_y_gt, f_y)
            f_auc_roc = 0.0
            if np.isnan(f_auc):
                continue
            
            a_dsets.append(dset)
            a_fns.append(f)
            
            
                # f_auc = 0.0
            a_precision_recall_auc.append([f_precision, f_recall, f_auc, f_auc_roc])
            # print (f_precision.shape)

            # if np.isnan(f_auc):#
            #     a_precision_recall_auc[-1][2] = 0.0

            # if np.isnan(f_precision).any() or np.isnan(f_recall).any():
            #     # print (a_precision_recall_auc)
            #     print f_y_gt
            #     print f_y
            #     print [f_precision, f_recall, f_auc]
            #     import sys; sys.exit(1)
    a_dsets = np.array(a_dsets)
    a_fns = np.array(a_fns)
    a_precision_recall_auc = np.array(a_precision_recall_auc)

    auc_dset_means = []
    auc_roc_dset_means = []
    auc_cum = 0.0
    for i, d in enumerate(list(set(a_dsets))):
        ri = np.where(a_dsets == d)[0]
        auc_dset_means.append([d, np.mean(a_precision_recall_auc[ri][:,2])])
        auc_roc_dset_means.append([d, np.mean(a_precision_recall_auc[ri][:,3])])

    auc_overall_mean = np.sum(a_precision_recall_auc[:,2]) / len(a_precision_recall_auc)
    auc_roc_overall_mean = np.sum(a_precision_recall_auc[:,3]) / len(a_precision_recall_auc)

    return a_dsets, a_fns, a_precision_recall_auc, auc_dset_means, auc_roc_dset_means, auc_overall_mean, auc_roc_overall_mean

def calculate_per_image_precision_top_k(dsets, fns, y, y_gt):
    a_dsets, a_fns, a_precision_scores = [], [], []
    for dset in sorted(list(set(dsets))):
        ri = np.where(dsets == dset)[0]
        dset_fns = fns[ri]
        dset_y = y[ri]
        dset_y_gt = y_gt[ri]

        unique_dset_fns = sorted(list(set(np.concatenate((dset_fns[:,0], dset_fns[:,1])).tolist())))
        for f in unique_dset_fns:
            ri_ = np.where((dset_fns[:,0] == f) | (dset_fns[:,1] == f))[0]
            ri_match = np.where(dset_y_gt[ri_] >= 1.0)[0]
            f_y = dset_y[ri_]
            f_y_gt = dset_y_gt[ri_]

            order = np.argsort(-f_y)
            k = len(ri_match)
            if k == 0:
                continue
            # print f_y_gt[order][:k].astype(np.int).astype(np.bool)
            # print f_y_gt[order][:k].astype(np.int)
            # print f_y[order][:k]
            # precision_score = sklearn.metrics.precision_score(f_y_gt[order][:k].astype(np.int), f_y[order][:k])
            precision_score = 1.0 * np.sum(f_y_gt[order][:k].astype(np.int)) / k
            # f_auc = sklearn.metrics.average_precision_score(f_y_gt, f_y)

            a_dsets.append(dset)
            a_fns.append(f)
            
            # if np.isnan(f_auc):
            #     f_auc = 0.0
            a_precision_scores.append(precision_score)
    a_dsets = np.array(a_dsets)
    a_fns = np.array(a_fns)
    a_precision_scores = np.array(a_precision_scores)

    dset_mean_precisions = []
    auc_cum = 0.0
    for i, d in enumerate(list(set(a_dsets))):
        ri = np.where(a_dsets == d)[0]
        dset_mean_precisions.append([d, np.mean(a_precision_scores[ri][:])])

    overall_mean_precision = np.sum(a_precision_scores) / len(a_precision_scores)

    return a_dsets, a_fns, a_precision_scores, dset_mean_precisions, overall_mean_precision

# def get_precision_recall(fns, labels, criteria, k, debug=False):
#     raw_results = []
#     aggregated_results = [0.0, 0.0]
#     unique_fns = sorted(list(set(np.concatenate((fns[:,0], fns[:,1])).tolist())))
#     for f in unique_fns:
#         if debug:
#             print ('\tPrecision/Recall for image "{}" at top {}'.format(f, k))

#         ri = np.where((fns[:,0] == f) | (fns[:,1] == f))[0]
#         criteria_ = criteria[ri]
#         labels_ = labels[ri]

#         order = criteria_.argsort()[::-1][:k]

#         precision = np.sum(labels_[order]) / len(labels_[order])
#         if np.sum(labels_) == 0:
#             recall = 1.0
#         else:
#             recall = np.sum(labels_[order]) / np.sum(labels_)

#         raw_results.append([f, precision, recall])
#         aggregated_results[0] += precision
#         aggregated_results[1] += recall
#         if debug:
#             print ('\t\tPrecision: {}  Recall: {}'.format(precision, recall))
#     aggregated_results[0] /= len(unique_fns)
#     aggregated_results[1] /= len(unique_fns)
#     return raw_results, aggregated_results

def get_postfix(datasets):
    root_datasets = []
    for d in datasets:
        root_datasets.append(os.path.abspath(d).split('/')[-2])

    root_datasets = list(set(root_datasets))
    return '+'.join(root_datasets) + '+'

def feature_matching_learned_classifier(options, training_datasets, testing_datasets):
    #################################################################################################################################
    #################################################################################################################################
    ############################################################ Training ###########################################################
    #################################################################################################################################
    #################################################################################################################################    
    for i,t in enumerate(training_datasets):
        data = dataset.DataSet(t)
        _fns, [_indices_1, _indices_2, _dists1, _dists2, _errors, _size1, _size2, _angle1, _angle2, _rerr1, \
            _rerr2, _labels] = data.load_feature_matching_dataset(lowes_threshold=0.95)
        if i == 0:
            fns, indices_1, indices_2, dists1, dists2, errors, size1, size2, angle1, angle2, rerr1, \
                rerr2, labels = _fns, _indices_1, _indices_2, _dists1, _dists2, _errors, _size1, _size2, _angle1, _angle2, _rerr1, \
                _rerr2, _labels
        else:
            fns = np.concatenate((fns, _fns), axis=0)
            indices_1 = np.concatenate((indices_1, _indices_1), axis=0)
            indices_2 = np.concatenate((indices_2, _indices_2), axis=0)
            dists1 = np.concatenate((dists1, _dists1), axis=0)
            dists2 = np.concatenate((dists2, _dists2), axis=0)
            errors = np.concatenate((errors, _errors), axis=0)
            size1 = np.concatenate((size1, _size1), axis=0)
            size2 = np.concatenate((size2, _size2), axis=0)
            angle1 = np.concatenate((angle1, _angle1), axis=0)
            angle2 = np.concatenate((angle2, _angle2), axis=0)
            rerr1 = np.concatenate((rerr1, _rerr1), axis=0)
            rerr2 = np.concatenate((rerr2, _rerr2), axis=0)
            labels = np.concatenate((labels, _labels), axis=0)
    labels[labels < 0] = 0
    max_distances = np.maximum(dists1, dists2)
    auc_baseline_train = calculate_dataset_auc(-1.0 * max_distances**2, labels, color='green', ls='dashed')
    _, _, _, _, regr_bdt, y = classifier.classify_boosted_dts_feature_match([fns, dists1, dists2, size1, size2, angle1, angle2, labels, True, None, options])
    auc_bdts_train = calculate_dataset_auc(y, labels,'r','dashed')
    training_postfix = get_postfix(training_datasets)
    fm_data_folder = options['feature_matching_data_folder']
    mkdir_p(fm_data_folder)
    save_classifier(regr_bdt, os.path.join(fm_data_folder, '{}{}-{}'.format(training_postfix, options['max_depth'], options['n_estimators'])))
    
    #################################################################################################################################
    #################################################################################################################################
    ############################################################ Testing ############################################################
    #################################################################################################################################
    #################################################################################################################################    

    for i,t in enumerate(testing_datasets):
        data = dataset.DataSet(t)
        _fns, [_indices_1, _indices_2, _dists1, _dists2, _errors, _size1, _size2, _angle1, _angle2, _rerr1, \
            _rerr2, _labels] = data.load_feature_matching_dataset(lowes_threshold=0.8)
        if i == 0:
            fns, indices_1, indices_2, dists1, dists2, errors, size1, size2, angle1, angle2, rerr1, \
                rerr2, labels = _fns, _indices_1, _indices_2, _dists1, _dists2, _errors, _size1, _size2, _angle1, _angle2, _rerr1, \
                _rerr2, _labels
        else:
            fns = np.concatenate((fns, _fns), axis=0)
            indices_1 = np.concatenate((indices_1, _indices_1), axis=0)
            indices_2 = np.concatenate((indices_2, _indices_2), axis=0)
            dists1 = np.concatenate((dists1, _dists1), axis=0)
            dists2 = np.concatenate((dists2, _dists2), axis=0)
            errors = np.concatenate((errors, _errors), axis=0)
            size1 = np.concatenate((size1, _size1), axis=0)
            size2 = np.concatenate((size2, _size2), axis=0)
            angle1 = np.concatenate((angle1, _angle1), axis=0)
            angle2 = np.concatenate((angle2, _angle2), axis=0)
            rerr1 = np.concatenate((rerr1, _rerr1), axis=0)
            rerr2 = np.concatenate((rerr2, _rerr2), axis=0)
            labels = np.concatenate((labels, _labels), axis=0)
    labels[labels < 0] = 0

    max_distances = np.maximum(dists1, dists2)
    auc_baseline_test = calculate_dataset_auc(-1.0 * max_distances**2, labels, color='blue', ls='dashed')
    _, _, _, _, regr_bdt, y = classifier.classify_boosted_dts_feature_match([fns, dists1, dists2, size1, size2, angle1, angle2, labels, False, regr_bdt, options])
    auc_bdts_test = calculate_dataset_auc(y, labels,'r','solid')
  
    plt.legend(
        ['Baseline (Train - ' + str([os.path.basename(t) for t in training_datasets]) + '), AUC=' + str(auc_baseline_train), 
        'Boosted DTs (Train - ' + str([os.path.basename(t) for t in training_datasets]) + '), AUC=' + str(auc_bdts_train),
        'Baseline (Test - ' + str([os.path.basename(t) for t in testing_datasets]) + '), AUC=' + str(auc_baseline_test),
        'Boosted DTs (Test - ' + str([os.path.basename(t) for t in testing_datasets]) + '), AUC=' + str(auc_bdts_test),
        ], 
        loc='lower right', 
        shadow=True,
        fontsize=10
        )

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(fm_data_folder, 'feature-matching-PR-{}{}-{}_{}.png'.format(\
        training_postfix, options['max_depth'], options['n_estimators'], str(datetime.datetime.now()) 
        ))
    )

def get_sample_weights(num_rmatches_tr, labels_tr):
    print ('Need to implement get_sample_weights')
    # import sys; sys.exit(1)
    return np.ones((len(labels_tr)))

def image_matching_learned_classifier(training_datasets, testing_datasets, options={}):
    epsilon = 0.00000001
    #################################################################################################################################
    #################################################################################################################################
    ############################################################ Training ###########################################################
    #################################################################################################################################
    #################################################################################################################################    
    for i,t in enumerate(training_datasets):
        print '\tDataset: {}'.format(t)
        data = dataset.DataSet(t)
        # _fns, [_R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
        #     _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
        #     _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
        #     _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, _num_gt_inliers, _labels] \
        #     = data.load_image_matching_dataset(robust_matches_threshold=15, rmatches_min_threshold=options['image_match_classifier_min_match'], \
        #         rmatches_max_threshold=options['image_match_classifier_max_match'])
        training_min_threshold = 0 if options['use_all_training_data'] else options['image_match_classifier_min_match']
        training_max_threshold = 10000 if options['use_all_training_data'] else options['image_match_classifier_max_match']
        _fns, [_R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
            _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
            _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
            _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
            _lcc_im1_15, _lcc_im2_15, _min_lcc_15, _max_lcc_15, \
            _lcc_im1_20, _lcc_im2_20, _min_lcc_20, _max_lcc_20, \
            _lcc_im1_25, _lcc_im2_25, _min_lcc_25, _max_lcc_25, \
            _lcc_im1_30, _lcc_im2_30, _min_lcc_30, _max_lcc_30, \
            _lcc_im1_35, _lcc_im2_35, _min_lcc_35, _max_lcc_35, \
            _lcc_im1_40, _lcc_im2_40, _min_lcc_40, _max_lcc_40, \
            _shortest_path_length, \
            _mds_rank_percentage_im1_im2, _mds_rank_percentage_im2_im1, \
            _distance_rank_percentage_im1_im2_gt, _distance_rank_percentage_im2_im1_gt, \
            _num_gt_inliers, _labels] \
            = data.load_image_matching_dataset(robust_matches_threshold=options['image_matching_gt_threshold'], rmatches_min_threshold=training_min_threshold, \
                rmatches_max_threshold=training_max_threshold, spl=options['shortest_path_length'])

        if i == 0:
            fns_tr, R11s_tr, R12s_tr, R13s_tr, R21s_tr, R22s_tr, R23s_tr, R31s_tr, R32s_tr, R33s_tr, num_rmatches_tr, num_matches_tr, spatial_entropy_1_8x8_tr, \
                spatial_entropy_2_8x8_tr, spatial_entropy_1_16x16_tr, spatial_entropy_2_16x16_tr, pe_histogram_tr, pe_polygon_area_percentage_tr, \
                nbvs_im1_tr, nbvs_im2_tr, te_histogram_tr, ch_im1_tr, ch_im2_tr, vt_rank_percentage_im1_im2_tr, vt_rank_percentage_im2_im1_tr, \
                sq_rank_scores_mean_tr, sq_rank_scores_min_tr, sq_rank_scores_max_tr, sq_distance_scores_tr, \
                lcc_im1_15_tr, lcc_im2_15_tr, min_lcc_15_tr, max_lcc_15_tr, \
                lcc_im1_20_tr, lcc_im2_20_tr, min_lcc_20_tr, max_lcc_20_tr, \
                lcc_im1_25_tr, lcc_im2_25_tr, min_lcc_25_tr, max_lcc_25_tr, \
                lcc_im1_30_tr, lcc_im2_30_tr, min_lcc_30_tr, max_lcc_30_tr, \
                lcc_im1_35_tr, lcc_im2_35_tr, min_lcc_35_tr, max_lcc_35_tr, \
                lcc_im1_40_tr, lcc_im2_40_tr, min_lcc_40_tr, max_lcc_40_tr, \
                shortest_path_length_tr, \
                mds_rank_percentage_im1_im2_tr, mds_rank_percentage_im2_im1_tr, \
                distance_rank_percentage_im1_im2_gt_tr, distance_rank_percentage_im2_im1_gt_tr, \
                num_gt_inliers_tr, labels_tr \
                = _fns, _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
                _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
                _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
                _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
                _lcc_im1_15, _lcc_im2_15, _min_lcc_15, _max_lcc_15, \
                _lcc_im1_20, _lcc_im2_20, _min_lcc_20, _max_lcc_20, \
                _lcc_im1_25, _lcc_im2_25, _min_lcc_25, _max_lcc_25, \
                _lcc_im1_30, _lcc_im2_30, _min_lcc_30, _max_lcc_30, \
                _lcc_im1_35, _lcc_im2_35, _min_lcc_35, _max_lcc_35, \
                _lcc_im1_40, _lcc_im2_40, _min_lcc_40, _max_lcc_40, \
                _shortest_path_length, \
                _mds_rank_percentage_im1_im2, _mds_rank_percentage_im2_im1, \
                _distance_rank_percentage_im1_im2_gt, _distance_rank_percentage_im2_im1_gt, \
                _num_gt_inliers, _labels
            dsets_tr = np.tile(t, (len(labels_tr),))
        else:
            fns_tr = np.concatenate((fns_tr, _fns), axis=0)
            R11s_tr = np.concatenate((R11s_tr, _R11s), axis=0)
            R12s_tr = np.concatenate((R12s_tr, _R12s), axis=0)
            R13s_tr = np.concatenate((R13s_tr, _R13s), axis=0)
            R21s_tr = np.concatenate((R21s_tr, _R21s), axis=0)
            R22s_tr = np.concatenate((R22s_tr, _R22s), axis=0)
            R23s_tr = np.concatenate((R23s_tr, _R23s), axis=0)
            R31s_tr = np.concatenate((R31s_tr, _R31s), axis=0)
            R32s_tr = np.concatenate((R32s_tr, _R32s), axis=0)
            R33s_tr = np.concatenate((R33s_tr, _R33s), axis=0)
            num_rmatches_tr = np.concatenate((num_rmatches_tr, _num_rmatches), axis=0)
            num_matches_tr = np.concatenate((num_matches_tr, _num_matches), axis=0)
            spatial_entropy_1_8x8_tr = np.concatenate((spatial_entropy_1_8x8_tr, _spatial_entropy_1_8x8), axis=0)
            spatial_entropy_2_8x8_tr = np.concatenate((spatial_entropy_2_8x8_tr, _spatial_entropy_2_8x8), axis=0)
            spatial_entropy_1_16x16_tr = np.concatenate((spatial_entropy_1_16x16_tr, _spatial_entropy_1_16x16), axis=0)
            spatial_entropy_2_16x16_tr = np.concatenate((spatial_entropy_2_16x16_tr, _spatial_entropy_2_16x16), axis=0)
            pe_histogram_tr = np.concatenate((pe_histogram_tr, _pe_histogram), axis=0)
            pe_polygon_area_percentage_tr = np.concatenate((pe_polygon_area_percentage_tr, _pe_polygon_area_percentage), axis=0)
            nbvs_im1_tr = np.concatenate((nbvs_im1_tr, _nbvs_im1), axis=0)
            nbvs_im2_tr = np.concatenate((nbvs_im2_tr, _nbvs_im2), axis=0)
            te_histogram_tr = np.concatenate((te_histogram_tr, _te_histogram), axis=0)
            ch_im1_tr = np.concatenate((ch_im1_tr, _ch_im1), axis=0)
            ch_im2_tr = np.concatenate((ch_im2_tr, _ch_im2), axis=0)
            vt_rank_percentage_im1_im2_tr = np.concatenate((vt_rank_percentage_im1_im2_tr, _vt_rank_percentage_im1_im2), axis=0)
            vt_rank_percentage_im2_im1_tr = np.concatenate((vt_rank_percentage_im2_im1_tr, _vt_rank_percentage_im2_im1), axis=0)
            sq_rank_scores_mean_tr = np.concatenate((sq_rank_scores_mean_tr, _sq_rank_scores_mean), axis=0)
            sq_rank_scores_min_tr = np.concatenate((sq_rank_scores_min_tr, _sq_rank_scores_min), axis=0)
            sq_rank_scores_max_tr = np.concatenate((sq_rank_scores_max_tr, _sq_rank_scores_max), axis=0)
            sq_distance_scores_tr = np.concatenate((sq_distance_scores_tr, _sq_distance_scores), axis=0)
            lcc_im1_15_tr = np.concatenate((lcc_im1_15_tr, _lcc_im1_15), axis=0)
            lcc_im2_15_tr = np.concatenate((lcc_im2_15_tr, _lcc_im2_15), axis=0)
            min_lcc_15_tr = np.concatenate((min_lcc_15_tr, _min_lcc_15), axis=0)
            max_lcc_15_tr = np.concatenate((max_lcc_15_tr, _max_lcc_15), axis=0)
            lcc_im1_20_tr = np.concatenate((lcc_im1_20_tr, _lcc_im1_20), axis=0)
            lcc_im2_20_tr = np.concatenate((lcc_im2_20_tr, _lcc_im2_20), axis=0)
            min_lcc_20_tr = np.concatenate((min_lcc_20_tr, _min_lcc_20), axis=0)
            max_lcc_20_tr = np.concatenate((max_lcc_20_tr, _max_lcc_20), axis=0)
            lcc_im1_25_tr = np.concatenate((lcc_im1_25_tr, _lcc_im1_25), axis=0)
            lcc_im2_25_tr = np.concatenate((lcc_im2_25_tr, _lcc_im2_25), axis=0)
            min_lcc_25_tr = np.concatenate((min_lcc_25_tr, _min_lcc_25), axis=0)
            max_lcc_25_tr = np.concatenate((max_lcc_25_tr, _max_lcc_25), axis=0)
            lcc_im1_30_tr = np.concatenate((lcc_im1_30_tr, _lcc_im1_30), axis=0)
            lcc_im2_30_tr = np.concatenate((lcc_im2_30_tr, _lcc_im2_30), axis=0)
            min_lcc_30_tr = np.concatenate((min_lcc_30_tr, _min_lcc_30), axis=0)
            max_lcc_30_tr = np.concatenate((max_lcc_30_tr, _max_lcc_30), axis=0)
            lcc_im1_35_tr = np.concatenate((lcc_im1_35_tr, _lcc_im1_35), axis=0)
            lcc_im2_35_tr = np.concatenate((lcc_im2_35_tr, _lcc_im2_35), axis=0)
            min_lcc_35_tr = np.concatenate((min_lcc_35_tr, _min_lcc_35), axis=0)
            max_lcc_35_tr = np.concatenate((max_lcc_35_tr, _max_lcc_35), axis=0)
            lcc_im1_40_tr = np.concatenate((lcc_im1_40_tr, _lcc_im1_40), axis=0)
            lcc_im2_40_tr = np.concatenate((lcc_im2_40_tr, _lcc_im2_40), axis=0)
            min_lcc_40_tr = np.concatenate((min_lcc_40_tr, _min_lcc_40), axis=0)
            max_lcc_40_tr = np.concatenate((max_lcc_40_tr, _max_lcc_40), axis=0)
            shortest_path_length_tr = np.concatenate((shortest_path_length_tr, _shortest_path_length), axis=0)
            num_gt_inliers_tr = np.concatenate((num_gt_inliers_tr, _num_gt_inliers), axis=0)
            mds_rank_percentage_im1_im2_tr = np.concatenate((mds_rank_percentage_im1_im2_tr, _mds_rank_percentage_im1_im2), axis=0)
            mds_rank_percentage_im2_im1_tr = np.concatenate((mds_rank_percentage_im2_im1_tr, _mds_rank_percentage_im2_im1), axis=0)
            distance_rank_percentage_im1_im2_gt_tr = np.concatenate((distance_rank_percentage_im1_im2_gt_tr, _distance_rank_percentage_im1_im2_gt), axis=0)
            distance_rank_percentage_im2_im1_gt_tr = np.concatenate((distance_rank_percentage_im2_im1_gt_tr, _distance_rank_percentage_im2_im1_gt), axis=0)
            labels_tr = np.concatenate((labels_tr, _labels), axis=0)
            dsets_tr = np.concatenate((dsets_tr, np.tile(t, (len(_labels),))), axis=0)

    labels_tr[labels_tr < 0] = 0
    weights_tr = get_sample_weights(num_rmatches_tr, labels_tr)
    print ('\tTraining datasets loaded - Tuples: {}  --  Inliers: {}  |  Outliers: {}'.format(len(labels_tr), len(np.where(labels_tr >= 1.0)[0]), len(np.where(labels_tr < 1.0)[0])))

    #################################################################################################################################
    #################################################################################################################################
    ############################################################ Testing ############################################################
    #################################################################################################################################
    #################################################################################################################################    
    for i,t in enumerate(testing_datasets):
        print '\tDataset: {}'.format(t)
        data = dataset.DataSet(t)
        _fns, [_R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
            _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
            _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
            _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
            _lcc_im1_15, _lcc_im2_15, _min_lcc_15, _max_lcc_15, \
            _lcc_im1_20, _lcc_im2_20, _min_lcc_20, _max_lcc_20, \
            _lcc_im1_25, _lcc_im2_25, _min_lcc_25, _max_lcc_25, \
            _lcc_im1_30, _lcc_im2_30, _min_lcc_30, _max_lcc_30, \
            _lcc_im1_35, _lcc_im2_35, _min_lcc_35, _max_lcc_35, \
            _lcc_im1_40, _lcc_im2_40, _min_lcc_40, _max_lcc_40, \
            _shortest_path_length, \
            _mds_rank_percentage_im1_im2, _mds_rank_percentage_im2_im1, \
            _distance_rank_percentage_im1_im2_gt, _distance_rank_percentage_im2_im1_gt, \
            _num_gt_inliers, _labels] \
            = data.load_image_matching_dataset(robust_matches_threshold=options['image_matching_gt_threshold'], rmatches_min_threshold=options['image_match_classifier_min_match'], \
                rmatches_max_threshold=options['image_match_classifier_max_match'], spl=options['shortest_path_length'])

        if i == 0:
            fns_te, R11s_te, R12s_te, R13s_te, R21s_te, R22s_te, R23s_te, R31s_te, R32s_te, R33s_te, num_rmatches_te, num_matches_te, spatial_entropy_1_8x8_te, \
                spatial_entropy_2_8x8_te, spatial_entropy_1_16x16_te, spatial_entropy_2_16x16_te, pe_histogram_te, pe_polygon_area_percentage_te, \
                nbvs_im1_te, nbvs_im2_te, te_histogram_te, ch_im1_te, ch_im2_te, vt_rank_percentage_im1_im2_te, vt_rank_percentage_im2_im1_te, \
                sq_rank_scores_mean_te, sq_rank_scores_min_te, sq_rank_scores_max_te, sq_distance_scores_te, \
                lcc_im1_15_te, lcc_im2_15_te, min_lcc_15_te, max_lcc_15_te, \
                lcc_im1_20_te, lcc_im2_20_te, min_lcc_20_te, max_lcc_20_te, \
                lcc_im1_25_te, lcc_im2_25_te, min_lcc_25_te, max_lcc_25_te, \
                lcc_im1_30_te, lcc_im2_30_te, min_lcc_30_te, max_lcc_30_te, \
                lcc_im1_35_te, lcc_im2_35_te, min_lcc_35_te, max_lcc_35_te, \
                lcc_im1_40_te, lcc_im2_40_te, min_lcc_40_te, max_lcc_40_te, \
                shortest_path_length_te, \
                mds_rank_percentage_im1_im2_te, mds_rank_percentage_im2_im1_te, \
                distance_rank_percentage_im1_im2_gt_te, distance_rank_percentage_im2_im1_gt_te, \
                num_gt_inliers_te, labels_te \
                = _fns, _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
                _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
                _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, \
                _sq_rank_scores_mean, _sq_rank_scores_min, _sq_rank_scores_max, _sq_distance_scores, \
                _lcc_im1_15, _lcc_im2_15, _min_lcc_15, _max_lcc_15, \
                _lcc_im1_20, _lcc_im2_20, _min_lcc_20, _max_lcc_20, \
                _lcc_im1_25, _lcc_im2_25, _min_lcc_25, _max_lcc_25, \
                _lcc_im1_30, _lcc_im2_30, _min_lcc_30, _max_lcc_30, \
                _lcc_im1_35, _lcc_im2_35, _min_lcc_35, _max_lcc_35, \
                _lcc_im1_40, _lcc_im2_40, _min_lcc_40, _max_lcc_40, \
                _shortest_path_length, \
                _mds_rank_percentage_im1_im2, _mds_rank_percentage_im2_im1, \
                _distance_rank_percentage_im1_im2_gt, _distance_rank_percentage_im2_im1_gt, \
                _num_gt_inliers, _labels
            dsets_te = np.tile(t, (len(labels_te),))
        else:
            fns_te = np.concatenate((fns_te, _fns), axis=0)
            R11s_te = np.concatenate((R11s_te, _R11s), axis=0)
            R12s_te = np.concatenate((R12s_te, _R12s), axis=0)
            R13s_te = np.concatenate((R13s_te, _R13s), axis=0)
            R21s_te = np.concatenate((R21s_te, _R21s), axis=0)
            R22s_te = np.concatenate((R22s_te, _R22s), axis=0)
            R23s_te = np.concatenate((R23s_te, _R23s), axis=0)
            R31s_te = np.concatenate((R31s_te, _R31s), axis=0)
            R32s_te = np.concatenate((R32s_te, _R32s), axis=0)
            R33s_te = np.concatenate((R33s_te, _R33s), axis=0)
            num_rmatches_te = np.concatenate((num_rmatches_te, _num_rmatches), axis=0)
            num_matches_te = np.concatenate((num_matches_te, _num_matches), axis=0)
            spatial_entropy_1_8x8_te = np.concatenate((spatial_entropy_1_8x8_te, _spatial_entropy_1_8x8), axis=0)
            spatial_entropy_2_8x8_te = np.concatenate((spatial_entropy_2_8x8_te, _spatial_entropy_2_8x8), axis=0)
            spatial_entropy_1_16x16_te = np.concatenate((spatial_entropy_1_16x16_te, _spatial_entropy_1_16x16), axis=0)
            spatial_entropy_2_16x16_te = np.concatenate((spatial_entropy_2_16x16_te, _spatial_entropy_2_16x16), axis=0)
            pe_histogram_te = np.concatenate((pe_histogram_te, _pe_histogram), axis=0)
            pe_polygon_area_percentage_te = np.concatenate((pe_polygon_area_percentage_te, _pe_polygon_area_percentage), axis=0)
            nbvs_im1_te = np.concatenate((nbvs_im1_te, _nbvs_im1), axis=0)
            nbvs_im2_te = np.concatenate((nbvs_im2_te, _nbvs_im2), axis=0)
            te_histogram_te = np.concatenate((te_histogram_te, _te_histogram), axis=0)
            ch_im1_te = np.concatenate((ch_im1_te, _ch_im1), axis=0)
            ch_im2_te = np.concatenate((ch_im2_te, _ch_im2), axis=0)
            vt_rank_percentage_im1_im2_te = np.concatenate((vt_rank_percentage_im1_im2_te, _vt_rank_percentage_im1_im2), axis=0)
            vt_rank_percentage_im2_im1_te = np.concatenate((vt_rank_percentage_im2_im1_te, _vt_rank_percentage_im2_im1), axis=0)
            sq_rank_scores_mean_te = np.concatenate((sq_rank_scores_mean_te, _sq_rank_scores_mean), axis=0)
            sq_rank_scores_min_te = np.concatenate((sq_rank_scores_min_te, _sq_rank_scores_min), axis=0)
            sq_rank_scores_max_te = np.concatenate((sq_rank_scores_max_te, _sq_rank_scores_max), axis=0)
            sq_distance_scores_te = np.concatenate((sq_distance_scores_te, _sq_distance_scores), axis=0)
            lcc_im1_15_te = np.concatenate((lcc_im1_15_te, _lcc_im1_15), axis=0)
            lcc_im2_15_te = np.concatenate((lcc_im2_15_te, _lcc_im2_15), axis=0)
            min_lcc_15_te = np.concatenate((min_lcc_15_te, _min_lcc_15), axis=0)
            max_lcc_15_te = np.concatenate((max_lcc_15_te, _max_lcc_15), axis=0)
            lcc_im1_20_te = np.concatenate((lcc_im1_20_te, _lcc_im1_20), axis=0)
            lcc_im2_20_te = np.concatenate((lcc_im2_20_te, _lcc_im2_20), axis=0)
            min_lcc_20_te = np.concatenate((min_lcc_20_te, _min_lcc_20), axis=0)
            max_lcc_20_te = np.concatenate((max_lcc_20_te, _max_lcc_20), axis=0)
            lcc_im1_25_te = np.concatenate((lcc_im1_25_te, _lcc_im1_25), axis=0)
            lcc_im2_25_te = np.concatenate((lcc_im2_25_te, _lcc_im2_25), axis=0)
            min_lcc_25_te = np.concatenate((min_lcc_25_te, _min_lcc_25), axis=0)
            max_lcc_25_te = np.concatenate((max_lcc_25_te, _max_lcc_25), axis=0)
            lcc_im1_30_te = np.concatenate((lcc_im1_30_te, _lcc_im1_30), axis=0)
            lcc_im2_30_te = np.concatenate((lcc_im2_30_te, _lcc_im2_30), axis=0)
            min_lcc_30_te = np.concatenate((min_lcc_30_te, _min_lcc_30), axis=0)
            max_lcc_30_te = np.concatenate((max_lcc_30_te, _max_lcc_30), axis=0)
            lcc_im1_35_te = np.concatenate((lcc_im1_35_te, _lcc_im1_35), axis=0)
            lcc_im2_35_te = np.concatenate((lcc_im2_35_te, _lcc_im2_35), axis=0)
            min_lcc_35_te = np.concatenate((min_lcc_35_te, _min_lcc_35), axis=0)
            max_lcc_35_te = np.concatenate((max_lcc_35_te, _max_lcc_35), axis=0)
            lcc_im1_40_te = np.concatenate((lcc_im1_40_te, _lcc_im1_40), axis=0)
            lcc_im2_40_te = np.concatenate((lcc_im2_40_te, _lcc_im2_40), axis=0)
            min_lcc_40_te = np.concatenate((min_lcc_40_te, _min_lcc_40), axis=0)
            max_lcc_40_te = np.concatenate((max_lcc_40_te, _max_lcc_40), axis=0)
            shortest_path_length_te = np.concatenate((shortest_path_length_te, _shortest_path_length), axis=0)
            mds_rank_percentage_im1_im2_te = np.concatenate((mds_rank_percentage_im1_im2_te, _mds_rank_percentage_im1_im2), axis=0)
            mds_rank_percentage_im2_im1_te = np.concatenate((mds_rank_percentage_im2_im1_te, _mds_rank_percentage_im2_im1), axis=0)
            distance_rank_percentage_im1_im2_gt_te = np.concatenate((distance_rank_percentage_im1_im2_gt_te, _distance_rank_percentage_im1_im2_gt), axis=0)
            distance_rank_percentage_im2_im1_gt_te = np.concatenate((distance_rank_percentage_im2_im1_gt_te, _distance_rank_percentage_im2_im1_gt), axis=0)
            num_gt_inliers_te = np.concatenate((num_gt_inliers_te, _num_gt_inliers), axis=0)
            labels_te = np.concatenate((labels_te, _labels), axis=0)
            dsets_te = np.concatenate((dsets_te, np.tile(t, (len(_labels),))), axis=0)
    labels_te[labels_te < 0] = 0
    weights_te = np.zeros((len(labels_te),))
    print ('\tTesting datasets loaded - Tuples: {}  --  Inliers: {}  |  Outliers: {}'.format(len(labels_te), len(np.where(labels_te >= 1.0)[0]), len(np.where(labels_te < 1.0)[0])))
    #################################################################################################################################
    #################################################################################################################################
    ########################################################## Classifiers ##########################################################
    #################################################################################################################################
    #################################################################################################################################    
    trained_classifier = None
    classifier_postfix = get_postfix(training_datasets)

    f1 = plt.figure(1)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('Inlier/Outlier Precision-Recall Curve \n(n_estimators={} max_depth={})'.format( \
        options['n_estimators'], options['max_depth']), \
        fontsize=18)

    auc_s_t = timer()
    auc_baseline_train, auc_roc_baseline_train = calculate_dataset_auc(num_rmatches_tr, labels_tr, color='green', ls='dashed')
    auc_e_t = timer()
    aucpi_s_t = timer()
    _, _, _, auc_per_image_per_dset_means_baseline_train, _, auc_per_image_mean_baseline_train, auc_roc_per_image_mean_baseline_train = \
        calculate_per_image_mean_auc(dsets_tr, fns_tr, num_rmatches_tr, labels_tr)
    aucpi_e_t = timer()
    ppi_s_t = timer()
    _, _, _, _, mean_precision_per_image_baseline_train = calculate_per_image_precision_top_k(dsets_tr, fns_tr, num_rmatches_tr, labels_tr)
    ppi_e_t = timer()
    print ('\tBaseline (Train): AUC: {} ({}) / {} ({}) / {}'.format(\
        round(auc_baseline_train,3), round(auc_roc_baseline_train,3), \
        round(auc_per_image_mean_baseline_train,3), round(auc_roc_per_image_mean_baseline_train,3), \
        round(mean_precision_per_image_baseline_train,3) \
        ))

    auc_s_t = timer()
    auc_baseline_test, auc_roc_baseline_test = calculate_dataset_auc(num_rmatches_te, labels_te, color='red', ls='dashed')
    auc_e_t = timer()
    aucpi_s_t = timer()
    _, _, _, auc_per_image_per_dset_means_baseline_test, _, auc_per_image_mean_baseline_test, auc_roc_per_image_mean_baseline_test = \
        calculate_per_image_mean_auc(dsets_te, fns_te, num_rmatches_te, labels_te)
    aucpi_e_t = timer()
    ppi_s_t = timer()
    _, _, _, _, mean_precision_per_image_baseline_test = calculate_per_image_precision_top_k(dsets_te, fns_te, num_rmatches_te, labels_te)
    ppi_e_t = timer()
    # print (json.dumps(auc_per_image_per_dset_means_baseline_train, sort_keys=True, indent=4, separators=(',', ': ')))
    # print (json.dumps(auc_per_image_per_dset_means_baseline_test, sort_keys=True, indent=4, separators=(',', ': ')))
    # import sys; sys.exit(1)

    print ('\tBaseline (Test): AUC: {} ({}) / {} ({}) / {}'.format(\
        round(auc_baseline_test,3), round(auc_roc_baseline_test,3), \
        round(auc_per_image_mean_baseline_test,3), round(auc_roc_per_image_mean_baseline_test,3), \
        round(mean_precision_per_image_baseline_test, 3) \
        ))
    legends = ['{} : {}'.format('Baseline (Train)', auc_baseline_train), '{} : {}'.format('Baseline (Test)', auc_baseline_test)]
    im_data_folder = os.path.join(options['image_matching_data_folder'], 'image-matching-classifiers-classifier-{}-max_depth-{}-n_estimators-{}-thresholds-{}-{}'.format( \
        options['classifier'], options['max_depth'], options['n_estimators'], options['image_match_classifier_min_match'], \
        options['image_match_classifier_max_match']))

    mkdir_p(im_data_folder)

    # # Used for LCCs (initial values)
    # classifier_training_scores = num_rmatches_tr
    # classifier_testing_scores = num_rmatches_te
    # classifier_training_threshold = 20
    # classifier_testing_scores = 20

    exps = [
        ['RM'],
        # ['MDS'],
        # ['RM', 'MDS'],
        ['RM', 'DIST-GT'],
        # ['PE'],
        # ['SP'],
        # ['SE'],
        # ['TE'],
        # ['NBVS'],
        # ['VD'],
        # ['TM'],
        # # ['VT'],
        # ['HIST'],
        # ['LCC'],
        # # ['SQ'],
        # ['RM', 'PE'],
        # ['RM', 'SP'],
        # ['RM', 'SE'],
        # ['RM', 'TE'],
        # ['RM', 'NBVS'],
        # ['RM', 'VD'],
        # ['RM', 'TM'],
        # # ['RM', 'VT'],
        # ['RM', 'HIST'],
        # ['RM', 'LCC'],
        # # ['RM', 'SQ'],
        # ['RM', 'PE', 'SE'],
        # ['RM', 'PE', 'SE', 'TE'],
        # ['RM', 'PE', 'SE', 'TE', 'MDS'],
        # # ['RM', 'PE', 'NBVS', 'TE'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'MDS'],
        # # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'VD'],
        # # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'TM'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'VD', 'TM'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'VD', 'TM', 'MDS'],
        # ['PE', 'SE', 'TE'],
        # ['PE', 'NBVS', 'TE'],
        # ['TE', 'PE', 'NBVS', 'SE'],
        # ['TE', 'PE', 'NBVS', 'SE', 'VD'],
        # ['TE', 'PE', 'NBVS', 'SE', 'TM'],
        # ['TE', 'PE', 'NBVS', 'SE', 'VD', 'TM'],

        # ['RM'],
        # ['RM', 'SE', 'PE'],
        # ['RM', 'SE', 'PE', 'TM'],
        # ['RM', 'SE', 'PE', 'TE'],
        # ['RM', 'SE', 'PE', 'TE', 'TM'],
        # ['LCC'],
        # ['SQ'],
        # ['TE'],
        # ['PE'],
        # ['NBVS'],
        # ['SE'],
        # ['TM'],
        # ['VD'],
        # ['RM', 'SQ'],
        # ['RM', 'TE'],
        # ['RM', 'PE'],
        # ['RM', 'VT'],
        # ['RM', 'NBVS'],
        # ['RM', 'SE'],
        # ['RM', 'VD'],
        # ['RM', 'TM'],
        # ['RM', 'LCC'],
        # ['RM', 'HIST'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE' ],
        # ['RM', 'SQ', 'TE', 'PE', 'NBVS', 'SE'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'VD', 'TM'],
        # ['RM', 'SQ', 'TE', 'PE', 'NBVS', 'SE', 'VD', 'TM'],
        # ['RM', 'TE', 'PE'],
        # ['RM', 'TE', 'PE', 'SE'],
        
        # ['RM', 'SQ', 'TE', 'PE', 'NBVS', 'SE', 'VD', 'TM', 'HIST'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE'],


        # ['RM', 'TE', 'PE', 'NBVS', 'SE'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'SQ'],
        # ['TE'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SE', 'HIST'],
        
        # ['RM', 'PE', 'NBVS', 'SE'],
        # ['RM', 'PE', 'NBVS', 'SE', 'SQ'],
        
        
        # ['RM', 'TE', 'PE', 'NBVS'],
        # ['RM', 'TE', 'PE', 'NBVS', 'SQ'],

        
        # ['RM', 'PE', 'TE'],
        # ['RM', 'VD', 'SE', 'TE', 'PE'],
        # ['RM', 'VD', 'SE', 'TE', 'PE', 'VT'],
        # ['RM', 'VD', 'SE', 'TE', 'PE', 'HIST', 'VT'],


        # ['RM'],
        # ['TE'],
        # ['RM', 'TE'],
        # ['RM', 'VD'],
        # ['RM', 'PE'],
        # ['RM', 'NBVS'],
        # ['RM', 'VT'],
        # ['RM', 'SE'],
        # ['RM', 'HIST'],
        # ['RM', 'VD', 'SE', 'PE'],
        # ['RM', 'VD', 'SE', 'TE', 'PE']

        # ['RM'],
        # ['RM', 'TE'],

        # ['RM', 'TE', 'PE'],
        # ['RM', 'TE', 'PE', 'NBVS'],
        # ['RM', 'TE', 'PE', 'SE'],
        # ['RM', 'TE', 'PE', 'VT'],
        # ['RM', 'TE', 'PE', 'VT', 'SE'],
        # ['RM', 'TE', 'PE', 'VT', 'NBVS'],
        # ['RM', 'TM', 'SE', 'HIST', 'VT', 'TE', 'PE', 'NBVS', 'VT']
    ]

    for exp in exps:
        print ('\t**************************************************************************************************************************')
        print ('\t**************************************************************************************************************************')
        print ('\t******************************* Performing experiment :{} *******************************'.format(exp))
        print ('\t**************************************************************************************************************************')
        print ('\t**************************************************************************************************************************')
        options['experiment'] = '+'.join(exp)
        for mode in ['train', 'test']:
            dsets = dsets_tr.copy() if mode == 'train' else dsets_te.copy()
            fns = fns_tr.copy() if mode == 'train' else fns_te.copy()
            R11s = R11s_tr.copy() if mode == 'train' else R11s_te.copy()
            R12s = R12s_tr.copy() if mode == 'train' else R12s_te.copy()
            R13s = R13s_tr.copy() if mode == 'train' else R13s_te.copy()
            R21s = R21s_tr.copy() if mode == 'train' else R21s_te.copy()
            R22s = R22s_tr.copy() if mode == 'train' else R22s_te.copy()
            R23s = R23s_tr.copy() if mode == 'train' else R23s_te.copy()
            R31s = R31s_tr.copy() if mode == 'train' else R31s_te.copy()
            R32s = R32s_tr.copy() if mode == 'train' else R32s_te.copy()
            R33s = R33s_tr.copy() if mode == 'train' else R33s_te.copy()
            num_rmatches = num_rmatches_tr.copy() if mode == 'train' else num_rmatches_te.copy()
            num_matches = num_matches_tr.copy() if mode == 'train' else num_matches_te.copy()
            spatial_entropy_1_8x8 = spatial_entropy_1_8x8_tr.copy() if mode == 'train' else spatial_entropy_1_8x8_te.copy()
            spatial_entropy_2_8x8 = spatial_entropy_2_8x8_tr.copy() if mode == 'train' else spatial_entropy_2_8x8_te.copy()
            spatial_entropy_1_16x16 = spatial_entropy_1_16x16_tr.copy() if mode == 'train' else spatial_entropy_1_16x16_te.copy()
            spatial_entropy_2_16x16 = spatial_entropy_2_16x16_tr.copy() if mode == 'train' else spatial_entropy_2_16x16_te.copy()
            pe_histogram = pe_histogram_tr.copy() if mode == 'train' else pe_histogram_te.copy()
            pe_polygon_area_percentage = pe_polygon_area_percentage_tr.copy() if mode == 'train' else pe_polygon_area_percentage_te.copy()
            nbvs_im1 = nbvs_im1_tr.copy() if mode == 'train' else nbvs_im1_te.copy()
            nbvs_im2 = nbvs_im2_tr.copy() if mode == 'train' else nbvs_im2_te.copy()
            te_histogram = te_histogram_tr.copy() if mode == 'train' else te_histogram_te.copy()
            ch_im1 = ch_im1_tr.copy() if mode == 'train' else ch_im1_te.copy()
            ch_im2 = ch_im2_tr.copy() if mode == 'train' else ch_im2_te.copy()
            vt_rank_percentage_im1_im2 = vt_rank_percentage_im1_im2_tr.copy() if mode == 'train' else vt_rank_percentage_im1_im2_te.copy()
            vt_rank_percentage_im2_im1 = vt_rank_percentage_im2_im1_tr.copy() if mode == 'train' else vt_rank_percentage_im2_im1_te.copy()
            sq_rank_scores_mean = sq_rank_scores_mean_tr.copy() if mode == 'train' else sq_rank_scores_mean_te.copy()
            sq_rank_scores_min = sq_rank_scores_min_tr.copy() if mode == 'train' else sq_rank_scores_min_te.copy()
            sq_rank_scores_max = sq_rank_scores_max_tr.copy() if mode == 'train' else sq_rank_scores_max_te.copy()
            sq_distance_scores = sq_distance_scores_tr.copy() if mode == 'train' else sq_distance_scores_te.copy()
            lcc_im1_15 = lcc_im1_15_tr.copy() if mode == 'train' else lcc_im1_15_te.copy()
            lcc_im2_15 = lcc_im2_15_tr.copy() if mode == 'train' else lcc_im2_15_te.copy()
            min_lcc_15 = min_lcc_15_tr.copy() if mode == 'train' else min_lcc_15_te.copy()
            max_lcc_15 = max_lcc_15_tr.copy() if mode == 'train' else max_lcc_15_te.copy()
            lcc_im1_20 = lcc_im1_20_tr.copy() if mode == 'train' else lcc_im1_20_te.copy()
            lcc_im2_20 = lcc_im2_20_tr.copy() if mode == 'train' else lcc_im2_20_te.copy()
            min_lcc_20 = min_lcc_20_tr.copy() if mode == 'train' else min_lcc_20_te.copy()
            max_lcc_20 = max_lcc_20_tr.copy() if mode == 'train' else max_lcc_20_te.copy()
            lcc_im1_25 = lcc_im1_25_tr.copy() if mode == 'train' else lcc_im1_25_te.copy()
            lcc_im2_25 = lcc_im2_25_tr.copy() if mode == 'train' else lcc_im2_25_te.copy()
            min_lcc_25 = min_lcc_25_tr.copy() if mode == 'train' else min_lcc_25_te.copy()
            max_lcc_25 = max_lcc_25_tr.copy() if mode == 'train' else max_lcc_25_te.copy()
            lcc_im1_30 = lcc_im1_30_tr.copy() if mode == 'train' else lcc_im1_30_te.copy()
            lcc_im2_30 = lcc_im2_30_tr.copy() if mode == 'train' else lcc_im2_30_te.copy()
            min_lcc_30 = min_lcc_30_tr.copy() if mode == 'train' else min_lcc_30_te.copy()
            max_lcc_30 = max_lcc_30_tr.copy() if mode == 'train' else max_lcc_30_te.copy()
            lcc_im1_35 = lcc_im1_35_tr.copy() if mode == 'train' else lcc_im1_35_te.copy()
            lcc_im2_35 = lcc_im2_35_tr.copy() if mode == 'train' else lcc_im2_35_te.copy()
            min_lcc_35 = min_lcc_35_tr.copy() if mode == 'train' else min_lcc_35_te.copy()
            max_lcc_35 = max_lcc_35_tr.copy() if mode == 'train' else max_lcc_35_te.copy()
            lcc_im1_40 = lcc_im1_40_tr.copy() if mode == 'train' else lcc_im1_40_te.copy()
            lcc_im2_40 = lcc_im2_40_tr.copy() if mode == 'train' else lcc_im2_40_te.copy()
            min_lcc_40 = min_lcc_40_tr.copy() if mode == 'train' else min_lcc_40_te.copy()
            max_lcc_40 = max_lcc_40_tr.copy() if mode == 'train' else max_lcc_40_te.copy()
            shortest_path_length = shortest_path_length_tr.copy() if mode == 'train' else shortest_path_length_te.copy()
            mds_rank_percentage_im1_im2 = mds_rank_percentage_im1_im2_tr.copy() if mode == 'train' else mds_rank_percentage_im1_im2_te.copy()
            mds_rank_percentage_im2_im1 = mds_rank_percentage_im2_im1_tr.copy() if mode == 'train' else mds_rank_percentage_im2_im1_te.copy()
            distance_rank_percentage_im1_im2_gt = distance_rank_percentage_im1_im2_gt_tr.copy() if mode == 'train' else distance_rank_percentage_im1_im2_gt_te.copy()
            distance_rank_percentage_im2_im1_gt = distance_rank_percentage_im2_im1_gt_tr.copy() if mode == 'train' else distance_rank_percentage_im2_im1_gt_te.copy()
            num_gt_inliers = num_gt_inliers_tr.copy() if mode == 'train' else num_gt_inliers_te.copy()
            labels = labels_tr.copy() if mode == 'train' else labels_te.copy()
            weights = weights_tr.copy() if mode == 'train' else weights_te.copy()
            train_mode = True if mode == 'train' else False

            # if options['classifier'] == 'NN' or options['classifier'] == 'GCN':
            dsets_te_clone = dsets_te.copy()
            fns_te_clone = fns_te.copy()
            R11s_te_clone = R11s_te.copy()
            R12s_te_clone = R12s_te.copy()
            R13s_te_clone = R13s_te.copy()
            R21s_te_clone = R21s_te.copy()
            R22s_te_clone = R22s_te.copy()
            R23s_te_clone = R23s_te.copy()
            R31s_te_clone = R31s_te.copy()
            R32s_te_clone = R32s_te.copy()
            R33s_te_clone = R33s_te.copy()
            num_rmatches_te_clone = num_rmatches_te.copy()
            num_matches_te_clone = num_matches_te.copy()
            spatial_entropy_1_8x8_te_clone = spatial_entropy_1_8x8_te.copy()
            spatial_entropy_2_8x8_te_clone = spatial_entropy_2_8x8_te.copy()
            spatial_entropy_1_16x16_te_clone = spatial_entropy_1_16x16_te.copy()
            spatial_entropy_2_16x16_te_clone = spatial_entropy_2_16x16_te.copy()
            pe_histogram_te_clone = pe_histogram_te.copy()
            pe_polygon_area_percentage_te_clone = pe_polygon_area_percentage_te.copy()
            nbvs_im1_te_clone = nbvs_im1_te.copy()
            nbvs_im2_te_clone = nbvs_im2_te.copy()
            te_histogram_te_clone = te_histogram_te.copy()
            ch_im1_te_clone = ch_im1_te.copy()
            ch_im2_te_clone = ch_im2_te.copy()
            vt_rank_percentage_im1_im2_te_clone = vt_rank_percentage_im1_im2_te.copy()
            vt_rank_percentage_im2_im1_te_clone = vt_rank_percentage_im2_im1_te.copy()
            sq_rank_scores_mean_te_clone = sq_rank_scores_mean_te.copy()
            sq_rank_scores_min_te_clone = sq_rank_scores_min_te.copy()
            sq_rank_scores_max_te_clone = sq_rank_scores_max_te.copy()
            sq_distance_scores_te_clone = sq_distance_scores_te.copy()
            lcc_im1_15_te_clone = lcc_im1_15_te.copy()
            lcc_im2_15_te_clone = lcc_im2_15_te.copy()
            min_lcc_15_te_clone = min_lcc_15_te.copy()
            max_lcc_15_te_clone = max_lcc_15_te.copy()
            lcc_im1_20_te_clone = lcc_im1_20_te.copy()
            lcc_im2_20_te_clone = lcc_im2_20_te.copy()
            min_lcc_20_te_clone = min_lcc_20_te.copy()
            max_lcc_20_te_clone = max_lcc_20_te.copy()
            lcc_im1_25_te_clone = lcc_im1_25_te.copy()
            lcc_im2_25_te_clone = lcc_im2_25_te.copy()
            min_lcc_25_te_clone = min_lcc_25_te.copy()
            max_lcc_25_te_clone = max_lcc_25_te.copy()
            lcc_im1_30_te_clone = lcc_im1_30_te.copy()
            lcc_im2_30_te_clone = lcc_im2_30_te.copy()
            min_lcc_30_te_clone = min_lcc_30_te.copy()
            max_lcc_30_te_clone = max_lcc_30_te.copy()
            lcc_im1_35_te_clone = lcc_im1_35_te.copy()
            lcc_im2_35_te_clone = lcc_im2_35_te.copy()
            min_lcc_35_te_clone = min_lcc_35_te.copy()
            max_lcc_35_te_clone = max_lcc_35_te.copy()
            lcc_im1_40_te_clone = lcc_im1_40_te.copy()
            lcc_im2_40_te_clone = lcc_im2_40_te.copy()
            min_lcc_40_te_clone = min_lcc_40_te.copy()
            max_lcc_40_te_clone = max_lcc_40_te.copy()
            shortest_path_length_te_clone = shortest_path_length_te.copy()
            mds_rank_percentage_im1_im2_te_clone = mds_rank_percentage_im1_im2_te.copy()
            mds_rank_percentage_im2_im1_te_clone = mds_rank_percentage_im2_im1_te.copy()
            distance_rank_percentage_im1_im2_gt_te_clone = distance_rank_percentage_im1_im2_gt_te.copy()
            distance_rank_percentage_im2_im1_gt_te_clone = distance_rank_percentage_im2_im1_gt_te.copy()
            num_gt_inliers_te_clone = num_gt_inliers_te.copy()
            labels_te_clone = labels_te.copy()
            weights_te_clone = weights_te.copy()

            plt.figure(1)
            if 'RM' not in exp:
                num_rmatches = np.zeros((len(labels),1))
                num_rmatches_te_clone = np.zeros((len(labels_te_clone),1))
            if 'TM' not in exp:
                num_matches = np.zeros((len(labels),1))
                num_matches_te_clone = np.zeros((len(labels_te_clone),1))
            if 'NBVS' not in exp:
                nbvs_im1 = np.zeros((len(labels),1))
                nbvs_im2 = np.zeros((len(labels),1))
                nbvs_im1_te_clone = np.zeros((len(labels_te_clone),1))
                nbvs_im2_te_clone = np.zeros((len(labels_te_clone),1))
            if 'SE' not in exp:
                spatial_entropy_1_8x8 = np.zeros((len(labels),1))
                spatial_entropy_2_8x8 = np.zeros((len(labels),1))
                spatial_entropy_1_16x16 = np.zeros((len(labels),1))
                spatial_entropy_2_16x16 = np.zeros((len(labels),1))
                spatial_entropy_1_8x8_te_clone = np.zeros((len(labels_te_clone),1))
                spatial_entropy_2_8x8_te_clone = np.zeros((len(labels_te_clone),1))
                spatial_entropy_1_16x16_te_clone = np.zeros((len(labels_te_clone),1))
                spatial_entropy_2_16x16_te_clone = np.zeros((len(labels_te_clone),1))
            if 'SQ' not in exp:
                sq_rank_scores_mean = np.zeros((len(labels),1))
                sq_rank_scores_min = np.zeros((len(labels),1))
                sq_rank_scores_max = np.zeros((len(labels),1))
                sq_distance_scores = np.zeros((len(labels),1))
                sq_rank_scores_mean_te_clone = np.zeros((len(labels_te_clone),1))
                sq_rank_scores_min_te_clone = np.zeros((len(labels_te_clone),1))
                sq_rank_scores_max_te_clone = np.zeros((len(labels_te_clone),1))
                sq_distance_scores_te_clone = np.zeros((len(labels_te_clone),1))
            if 'LCC' not in exp:
                lcc_im1_15 = np.zeros((len(labels),1))
                lcc_im2_15 = np.zeros((len(labels),1))
                min_lcc_15 = np.zeros((len(labels),1))
                max_lcc_15 = np.zeros((len(labels),1))
                lcc_im1_15_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im2_15_te_clone = np.zeros((len(labels_te_clone),1))
                min_lcc_15_te_clone = np.zeros((len(labels_te_clone),1))
                max_lcc_15_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im1_20 = np.zeros((len(labels),1))
                lcc_im2_20 = np.zeros((len(labels),1))
                min_lcc_20 = np.zeros((len(labels),1))
                max_lcc_20 = np.zeros((len(labels),1))
                lcc_im1_20_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im2_20_te_clone = np.zeros((len(labels_te_clone),1))
                min_lcc_20_te_clone = np.zeros((len(labels_te_clone),1))
                max_lcc_20_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im1_25 = np.zeros((len(labels),1))
                lcc_im2_25 = np.zeros((len(labels),1))
                min_lcc_25 = np.zeros((len(labels),1))
                max_lcc_25 = np.zeros((len(labels),1))
                lcc_im1_25_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im2_25_te_clone = np.zeros((len(labels_te_clone),1))
                min_lcc_25_te_clone = np.zeros((len(labels_te_clone),1))
                max_lcc_25_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im1_30 = np.zeros((len(labels),1))
                lcc_im2_30 = np.zeros((len(labels),1))
                min_lcc_30 = np.zeros((len(labels),1))
                max_lcc_30 = np.zeros((len(labels),1))
                lcc_im1_30_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im2_30_te_clone = np.zeros((len(labels_te_clone),1))
                min_lcc_30_te_clone = np.zeros((len(labels_te_clone),1))
                max_lcc_30_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im1_35 = np.zeros((len(labels),1))
                lcc_im2_35 = np.zeros((len(labels),1))
                min_lcc_35 = np.zeros((len(labels),1))
                max_lcc_35 = np.zeros((len(labels),1))
                lcc_im1_35_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im2_35_te_clone = np.zeros((len(labels_te_clone),1))
                min_lcc_35_te_clone = np.zeros((len(labels_te_clone),1))
                max_lcc_35_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im1_40 = np.zeros((len(labels),1))
                lcc_im2_40 = np.zeros((len(labels),1))
                min_lcc_40 = np.zeros((len(labels),1))
                max_lcc_40 = np.zeros((len(labels),1))
                lcc_im1_40_te_clone = np.zeros((len(labels_te_clone),1))
                lcc_im2_40_te_clone = np.zeros((len(labels_te_clone),1))
                min_lcc_40_te_clone = np.zeros((len(labels_te_clone),1))
                max_lcc_40_te_clone = np.zeros((len(labels_te_clone),1))

            if 'SP' not in exp:
                shortest_path_length = np.zeros((len(labels),))
                shortest_path_length_te_clone = np.zeros((len(labels_te_clone),1))

            if 'VD' not in exp:
                R11s = np.zeros((len(labels),))
                R12s = np.zeros((len(labels),))
                R13s = np.zeros((len(labels),))
                R21s = np.zeros((len(labels),))
                R22s = np.zeros((len(labels),))
                R23s = np.zeros((len(labels),))
                R31s = np.zeros((len(labels),))
                R32s = np.zeros((len(labels),))
                R33s = np.zeros((len(labels),))
                R11s_te_clone = np.zeros((len(labels_te_clone),1))
                R12s_te_clone = np.zeros((len(labels_te_clone),1))
                R13s_te_clone = np.zeros((len(labels_te_clone),1))
                R21s_te_clone = np.zeros((len(labels_te_clone),1))
                R22s_te_clone = np.zeros((len(labels_te_clone),1))
                R23s_te_clone = np.zeros((len(labels_te_clone),1))
                R31s_te_clone = np.zeros((len(labels_te_clone),1))
                R32s_te_clone = np.zeros((len(labels_te_clone),1))
                R33s_te_clone = np.zeros((len(labels_te_clone),1))

            if 'TE' not in exp:
                te_histogram = np.zeros(te_histogram.shape)
                te_histogram_te_clone = np.zeros(te_histogram_te_clone.shape)
            if 'PE' not in exp:
                pe_histogram = np.zeros(pe_histogram.shape)
                pe_polygon_area_percentage = np.zeros(pe_polygon_area_percentage.shape)
                pe_histogram_te_clone = np.zeros(pe_histogram_te_clone.shape)
                pe_polygon_area_percentage_te_clone = np.zeros(pe_polygon_area_percentage_te_clone.shape)
            if 'HIST' not in exp:
                ch_im1 = np.zeros(ch_im1.shape)
                ch_im2 = np.zeros(ch_im2.shape)
                ch_im1_te_clone = np.zeros(ch_im1_te_clone.shape)
                ch_im2_te_clone = np.zeros(ch_im2_te_clone.shape)
            if 'VT' not in exp:
                vt_rank_percentage_im1_im2 = np.zeros(vt_rank_percentage_im1_im2.shape)
                vt_rank_percentage_im2_im1 = np.zeros(vt_rank_percentage_im2_im1.shape)
                vt_rank_percentage_im1_im2_te_clone = np.zeros(vt_rank_percentage_im1_im2_te_clone.shape)
                vt_rank_percentage_im2_im1_te_clone = np.zeros(vt_rank_percentage_im2_im1_te_clone.shape)
            if 'MDS' not in exp:
                mds_rank_percentage_im1_im2 = np.zeros(mds_rank_percentage_im1_im2.shape)
                mds_rank_percentage_im2_im1 = np.zeros(mds_rank_percentage_im2_im1.shape)
                mds_rank_percentage_im1_im2_te_clone = np.zeros(mds_rank_percentage_im1_im2_te_clone.shape)
                mds_rank_percentage_im2_im1_te_clone = np.zeros(mds_rank_percentage_im2_im1_te_clone.shape)
            if 'DIST-GT' not in exp:
                distance_rank_percentage_im1_im2_gt = np.zeros(distance_rank_percentage_im1_im2_gt.shape)
                distance_rank_percentage_im2_im1_gt = np.zeros(distance_rank_percentage_im2_im1_gt.shape)
                distance_rank_percentage_im1_im2_gt_te_clone = np.zeros(distance_rank_percentage_im1_im2_gt_te_clone.shape)
                distance_rank_percentage_im2_im1_gt_te_clone = np.zeros(distance_rank_percentage_im2_im1_gt_te_clone.shape)
            arg = [ \
                dsets, fns, R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, num_rmatches, num_matches, spatial_entropy_1_8x8, \
                spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, pe_histogram, pe_polygon_area_percentage, \
                nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
                sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
                lcc_im1_15, lcc_im2_15, min_lcc_15, max_lcc_15, \
                lcc_im1_20, lcc_im2_20, min_lcc_20, max_lcc_20, \
                lcc_im1_25, lcc_im2_25, min_lcc_25, max_lcc_25, \
                lcc_im1_30, lcc_im2_30, min_lcc_30, max_lcc_30, \
                lcc_im1_35, lcc_im2_35, min_lcc_35, max_lcc_35, \
                lcc_im1_40, lcc_im2_40, min_lcc_40, max_lcc_40, \
                shortest_path_length, \
                mds_rank_percentage_im1_im2, mds_rank_percentage_im2_im1, \
                distance_rank_percentage_im1_im2_gt, distance_rank_percentage_im2_im1_gt, \
                labels, weights, \
                train_mode, trained_classifier, options
            ]

            if options['classifier'] == 'BDT':
                _, _, regr, scores, _, _ = classifier.classify_boosted_dts_image_match(arg)
                print ("\t\tFinished Learning/Classifying Boosted Decision Tree")  
            elif options['classifier'] == 'NN':
                if mode == 'train':
                    arg_te = [ \
                            dsets_te_clone, fns_te_clone, R11s_te_clone, R12s_te_clone, R13s_te_clone, R21s_te_clone, R22s_te_clone, R23s_te_clone, \
                            R31s_te_clone, R32s_te_clone, R33s_te_clone, num_rmatches_te_clone, num_matches_te_clone, spatial_entropy_1_8x8_te_clone, \
                            spatial_entropy_2_8x8_te_clone, spatial_entropy_1_16x16_te_clone, spatial_entropy_2_16x16_te_clone, pe_histogram_te_clone, pe_polygon_area_percentage_te_clone, \
                            nbvs_im1_te_clone, nbvs_im2_te_clone, te_histogram_te_clone, ch_im1_te_clone, ch_im2_te_clone, vt_rank_percentage_im1_im2_te_clone, \
                            vt_rank_percentage_im2_im1_te_clone, sq_rank_scores_mean_te_clone, sq_rank_scores_min_te_clone, sq_rank_scores_max_te_clone, sq_distance_scores_te_clone, \
                            lcc_im1_15_te_clone, lcc_im2_15_te_clone, min_lcc_15_te_clone, max_lcc_15_te_clone, \
                            lcc_im1_20_te_clone, lcc_im2_20_te_clone, min_lcc_20_te_clone, max_lcc_20_te_clone, \
                            lcc_im1_25_te_clone, lcc_im2_25_te_clone, min_lcc_25_te_clone, max_lcc_25_te_clone, \
                            lcc_im1_30_te_clone, lcc_im2_30_te_clone, min_lcc_30_te_clone, max_lcc_30_te_clone, \
                            lcc_im1_35_te_clone, lcc_im2_35_te_clone, min_lcc_35_te_clone, max_lcc_35_te_clone, \
                            lcc_im1_40_te_clone, lcc_im2_40_te_clone, min_lcc_40_te_clone, max_lcc_40_te_clone, \
                            shortest_path_length_te_clone, \
                            mds_rank_percentage_im1_im2_te_clone, mds_rank_percentage_im2_im1_te_clone, \
                            distance_rank_percentage_im1_im2_gt_te_clone, distance_rank_percentage_im2_im1_gt_te_clone, \
                            labels_te_clone, weights_te_clone, \
                            False, trained_classifier, options
                        ]
                    # arg: train set, arg_te: test set
                    _, _, regr, scores, _ = nn.classify_nn_image_match_training(arg, arg_te)
                else:
                    # arg: test set
                    _, _, regr, scores, _ = nn.classify_nn_image_match_inference(arg)
            elif options['classifier'] == 'CONVNET':
                if mode == 'train':
                    arg_te = [ \
                            dsets_te_clone, fns_te_clone, R11s_te_clone, R12s_te_clone, R13s_te_clone, R21s_te_clone, R22s_te_clone, R23s_te_clone, \
                            R31s_te_clone, R32s_te_clone, R33s_te_clone, num_rmatches_te_clone, num_matches_te_clone, spatial_entropy_1_8x8_te_clone, \
                            spatial_entropy_2_8x8_te_clone, spatial_entropy_1_16x16_te_clone, spatial_entropy_2_16x16_te_clone, pe_histogram_te_clone, pe_polygon_area_percentage_te_clone, \
                            nbvs_im1_te_clone, nbvs_im2_te_clone, te_histogram_te_clone, ch_im1_te_clone, ch_im2_te_clone, vt_rank_percentage_im1_im2_te_clone, \
                            vt_rank_percentage_im2_im1_te_clone, sq_rank_scores_mean_te_clone, sq_rank_scores_min_te_clone, sq_rank_scores_max_te_clone, sq_distance_scores_te_clone, \
                            lcc_im1_15_te_clone, lcc_im2_15_te_clone, min_lcc_15_te_clone, max_lcc_15_te_clone, \
                            lcc_im1_20_te_clone, lcc_im2_20_te_clone, min_lcc_20_te_clone, max_lcc_20_te_clone, \
                            lcc_im1_25_te_clone, lcc_im2_25_te_clone, min_lcc_25_te_clone, max_lcc_25_te_clone, \
                            lcc_im1_30_te_clone, lcc_im2_30_te_clone, min_lcc_30_te_clone, max_lcc_30_te_clone, \
                            lcc_im1_35_te_clone, lcc_im2_35_te_clone, min_lcc_35_te_clone, max_lcc_35_te_clone, \
                            lcc_im1_40_te_clone, lcc_im2_40_te_clone, min_lcc_40_te_clone, max_lcc_40_te_clone, \
                            shortest_path_length_te_clone, \
                            mds_rank_percentage_im1_im2_te_clone, mds_rank_percentage_im2_im1_te_clone, \
                            distance_rank_percentage_im1_im2_gt_te_clone, distance_rank_percentage_im2_im1_gt_te_clone, \
                            labels_te_clone, weights_te, \
                            False, trained_classifier, options
                        ]
                    # arg: train set, arg_te: test set
                    _, _, regr, scores, _ = convnet.classify_convnet_image_match_training(arg, arg_te)
                    import sys; sys.exit(1)
                else:
                    # arg: test set
                    _, _, regr, scores, _ = convnet.classify_convnet_image_match_inference(arg)
            elif options['classifier'] == 'GCN':
                if mode == 'train':
                    arg_te = [ \
                            dsets_te_clone, fns_te_clone, R11s_te_clone, R12s_te_clone, R13s_te_clone, R21s_te_clone, R22s_te_clone, R23s_te_clone, \
                            R31s_te_clone, R32s_te_clone, R33s_te_clone, num_rmatches_te_clone, num_matches_te_clone, spatial_entropy_1_8x8_te_clone, \
                            spatial_entropy_2_8x8_te_clone, spatial_entropy_1_16x16_te_clone, spatial_entropy_2_16x16_te_clone, pe_histogram_te_clone, pe_polygon_area_percentage_te_clone, \
                            nbvs_im1_te_clone, nbvs_im2_te_clone, te_histogram_te_clone, ch_im1_te_clone, ch_im2_te_clone, vt_rank_percentage_im1_im2_te_clone, \
                            vt_rank_percentage_im2_im1_te_clone, sq_rank_scores_mean_te_clone, sq_rank_scores_min_te_clone, sq_rank_scores_max_te_clone, sq_distance_scores_te_clone, \
                            lcc_im1_15_te_clone, lcc_im2_15_te_clone, min_lcc_15_te_clone, max_lcc_15_te_clone, \
                            lcc_im1_20_te_clone, lcc_im2_20_te_clone, min_lcc_20_te_clone, max_lcc_20_te_clone, \
                            lcc_im1_25_te_clone, lcc_im2_25_te_clone, min_lcc_25_te_clone, max_lcc_25_te_clone, \
                            lcc_im1_30_te_clone, lcc_im2_30_te_clone, min_lcc_30_te_clone, max_lcc_30_te_clone, \
                            lcc_im1_35_te_clone, lcc_im2_35_te_clone, min_lcc_35_te_clone, max_lcc_35_te_clone, \
                            lcc_im1_40_te_clone, lcc_im2_40_te_clone, min_lcc_40_te_clone, max_lcc_40_te_clone, \
                            shortest_path_length_te_clone, \
                            mds_rank_percentage_im1_im2_te_clone, mds_rank_percentage_im2_im1_te_clone, \
                            distance_rank_percentage_im1_im2_gt_te_clone, distance_rank_percentage_im2_im1_gt_te_clone, \
                            labels_te_clone, \
                            False, trained_classifier, options
                        ]
                    # arg: train set, arg_te: test set
                    _, _, regr, scores, _ = gcn.classify_gcn_image_match_training(arg, arg_te)
                else:
                    # arg: test set
                    _, _, regr, scores, _ = gcn.classify_gcn_image_match_inference(arg)

            
            
            if not train_mode:
                auc, auc_roc = calculate_dataset_auc(scores, labels, 'black', 'solid' if not train_mode else 'dashed')
                _, _, _, _, _,auc_per_image_mean, auc_roc_per_image_mean = calculate_per_image_mean_auc(dsets, fns, scores, labels)
                _, _, _, _, mean_precision_per_image = calculate_per_image_precision_top_k(dsets, fns, scores, labels)
                trained_classifier = None
                # classifier_testing_scores = scores
                # classifier_testing_threshold = 0.3
                # legends.append('{} : {}'.format(exp, auc))
                # legends.append('{} : {} : {} / {} / {}'.format(mode, exp, auc, auc_per_image_mean, mean_precision_per_image))
                print ('\t\tExperiment: {} AUC: {} ({}) / {} ({}) / {}'.format(exp, \
                    round(auc,3), round(auc_roc,3), \
                    round(auc_per_image_mean,3), round(auc_roc_per_image_mean,3), \
                    round(mean_precision_per_image, 3)
                ))
            else:
                auc = 0.0
                trained_classifier = regr
                # classifier_training_scores = scores
                # classifier_training_threshold = 0.3
                if options['classifier'] == 'BDT':
                    save_classifier(regr, os.path.join(im_data_folder, '{}+{}{}-{}-thresholds-{}-{}'.format(\
                        '+'.join(exp), classifier_postfix, options['max_depth'], options['n_estimators'], \
                        options['image_match_classifier_min_match'], options['image_match_classifier_max_match'])))
                elif options['classifier'] == 'NN':
                    pass
  
    plt.legend(legends, loc='lower left', shadow=True, fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(im_data_folder, 'image-matching-PR-{}{}-{}_{}.png'.format(\
        classifier_postfix, options['max_depth'], options['n_estimators'], str(datetime.datetime.now()))))
    plt.gcf().clear()

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    parser.add_argument('-n', '--n_estimators', help='')
    parser.add_argument('-d', '--max_depth', help='')
    parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    parser.add_argument('-x', '--image_match_classifier_max_match', help='')
    parser.add_argument('-c', '--classifier', help='')
    parser.add_argument('-a', '--use_all_training_data', help='')
    parser.add_argument('-v', '--train_on_val', help='')

    parser.add_argument('--convnet_lr')
    parser.add_argument('--convnet_batch_size', default=1, help='8, 16, 32, ...')
    parser.add_argument('--convnet_resnet_model', help='18, 34, 50, 101, 152')
    parser.add_argument('--convnet_loss', help='ce, t')
    parser.add_argument('--convnet_triplet_sampling_strategy', help='n, r, u')
    parser.add_argument('--convnet_features', help='RM, RM+TE, RM+NBVS, RM+TE+NBVS')
    parser.add_argument('--convnet_use_images', help='')
    parser.add_argument('--convnet_use_feature_match_map', help='')
    parser.add_argument('--convnet_use_track_map', help='')
    parser.add_argument('--convnet_use_non_rmatches_map', help='')
    parser.add_argument('--convnet_use_rmatches_map', help='')
    parser.add_argument('--convnet_use_matches_map', help='')
    parser.add_argument('--convnet_use_photometric_error_maps', help='')
    
    
    # parser.set_defaults(use_all_training_data=False)
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset
    datasets = {
        'training': {
            'TanksAndTemples': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Barn',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Caterpillar',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Church',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Courthouse',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Ignatius',
            ],
            'ETH3D': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/courtyard',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/delivery_area',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/electro',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/facade',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/kicker',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/meadow',
            ],
            'SMALL': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/courtyard',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/delivery_area',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/electro',
            ],
            'TUM_RGBD_SLAM': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_360',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_room',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_hemisphere',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_coke',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk_with_person',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_dishes',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_no_loop',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_with_loop',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere2',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_360',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam2',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam3',
            ],
            'GTAV_540': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0000',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0001',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0002',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0003',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0004',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0005',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0006',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0007',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0008',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0009',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0010',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0011',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0012',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0013',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0014',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0015',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0016',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0017',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0018',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0019',
            ]
        },
        'testing': {
            'TanksAndTemples': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Meetingroom',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Truck',
            ],
            'ETH3D': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/exhibition_hall',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/lecture_room',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/living_room',
            ],
            'SMALL': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/lecture_room',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/living_room',
            ],
            'TUM_RGBD_SLAM': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz',
            ],
            'GTAV_540': [
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0020',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0021',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0022',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0023',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0024',
                '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/GTAV_540/0025',
            ]
        }
    }

    if parser_options.convnet_loss == 'ce':
        convnet_loss = 'cross-entropy'
    elif parser_options.convnet_loss == 't':
        convnet_loss = 'triplet'
    else:
        convnet_loss = 'cross-entropy'

    if parser_options.convnet_triplet_sampling_strategy == 'n':
        triplet_sampling_strategy = 'normal'
    elif parser_options.convnet_triplet_sampling_strategy == 'r':
        triplet_sampling_strategy = 'random'
    elif parser_options.convnet_triplet_sampling_strategy == 'u':
        triplet_sampling_strategy = 'uniform-files'
    else:
        triplet_sampling_strategy = 'normal'

    if parser_options.convnet_resnet_model == '18':
        convnet_resnet_model = 'resnet18'
    elif parser_options.convnet_resnet_model == '34':
        convnet_resnet_model = 'resnet34'
    elif parser_options.convnet_resnet_model == '50':
        convnet_resnet_model = 'resnet50'
    elif parser_options.convnet_resnet_model == '101':
        convnet_resnet_model = 'resnet101'
    elif parser_options.convnet_resnet_model == '152':
        convnet_resnet_model = 'resnet152'
    else:
        convnet_resnet_model = 'resnet50'

    options = {
        'feature_matching_data_folder': 'data/feature-matching-classifiers-results',
        'image_matching_data_folder': 'data/image-matching-classifiers-results',
        'image_matching_gt_threshold': 20,
        'use_all_training_data': True if parser_options.use_all_training_data == 'yes' else False,
        'train_on_val': True if parser_options.train_on_val == 'yes' else False,
        # 'classifier': 'BDT',
        'classifier': parser_options.classifier.upper(),
        # 'classifier': 'GCN',
        # BDT options
        'max_depth': int(parser_options.max_depth),
        'n_estimators': int(parser_options.n_estimators),
        'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match),
        'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match),
        'shortest_path_length': 200000,
        # 'feature_selection': False, \
        # NN options
        # 'batch_size': 128,
        # 'batch_size': 1,
        # 'batch_size': 1024 if parser_options.classifier.upper() != 'GCN' else 1,
        'batch_size': int(parser_options.convnet_batch_size) if parser_options.classifier.upper() != 'GCN' else 1,
        # 'shuffle': True,
        'shuffle': True,
        # 'lr': 0.01,
        # 'lr': 0.005,
        'lr': float(parser_options.convnet_lr), 
        'optimizer': 'adam',
        'wd': 0.0001,
        'epochs': 30,
        'start_epoch': 0,
        'resume': True,
        'lr_decay': 0.01,
        'nn_log_dir': 'data/nn-image-matching-classifiers-results/logs',
        'gcn_log_dir': 'data/nn-image-matching-classifiers-results/logs',
        'convnet_log_dir': 'data/nn-image-matching-classifiers-results/logs',
        'subsample_ratio': 1.0,
        'log_interval': 1,
        'opensfm_path': parser_options.opensfm_path,
        # 'fine_tuning': True, \
        # 'class_balance': False, \
        # 'all_features': False, \
        'triplet-sampling-strategy': triplet_sampling_strategy,
        # 'triplet-sampling-strategy': 'random',
        # 'triplet-sampling-strategy': 'uniform-files',
        # 'triplet-sampling-strategy': 'normal',
        # 'sample-inclass': True,
        'sample-inclass': False,
        # 'num_workers': 4,
        'num_workers': 12 if parser_options.classifier.upper() != 'GCN' else 0,
        # 'num_workers': 10,
        # 'use_image_features': True,
        # 'use_image_features': False,
        'loss': convnet_loss,
        'model': convnet_resnet_model,
        'features': parser_options.convnet_features,
        'convnet_input_size': 224,
        'mlp-layer-size': 256,
        'convnet_use_images': True if parser_options.convnet_use_images == 'yes' else False,
        'convnet_use_feature_match_map': True if parser_options.convnet_use_feature_match_map == 'yes' else False,
        'convnet_use_track_map': True if parser_options.convnet_use_track_map == 'yes' else False,
        'convnet_use_warped_images': False,
        'convnet_use_non_rmatches_map': True if parser_options.convnet_use_non_rmatches_map == 'yes' else False,
        'convnet_use_rmatches_map': True if parser_options.convnet_use_rmatches_map == 'yes' else False,
        'convnet_use_matches_map': True if parser_options.convnet_use_matches_map == 'yes' else False,
        'convnet_use_photometric_error_maps': True if parser_options.convnet_use_photometric_error_maps == 'yes' else False,
        'convnet_load_dataset_in_memory': False
    }
    if options['classifier'] == 'GCN':
        mkdir_p(options['gcn_log_dir'])


    dataset_experiments = [
        # {
        #     'training': ['SMALL'], 
        #     'testing': ['SMALL']
        # },
        # {
        #     'training': ['TanksAndTemples'], 
        #     'testing': ['TanksAndTemples']
        # },
        # {
        #     'training': ['TUM_RGBD_SLAM'], 
        #     'testing': ['TUM_RGBD_SLAM']
        # },
        # {
        #     'training': ['TanksAndTemples', 'ETH3D'], 
        #     'testing': ['TanksAndTemples', 'ETH3D']
        # },
        {
            'training': ['TanksAndTemples', 'ETH3D', 'TUM_RGBD_SLAM'], 
            'testing': ['TanksAndTemples', 'ETH3D', 'TUM_RGBD_SLAM']
        },
    ]
    for dataset_exp in dataset_experiments:
        print ('#'*200)
        print ('#'*200)
        print ('############################### Dataset Exps => Training : {}\t\t Testing: {} ###############################'.format( \
            dataset_exp['training'], dataset_exp['testing']
        ))
        print ('#'*200)
        print ('#'*200)
        training_datasets = []
        testing_datasets = []

        for dataset_name in dataset_exp['training']:
            training_datasets.extend(datasets['training'][dataset_name])
            # training_datasets.extend(datasets['testing'][dataset_name])
        for dataset_name in dataset_exp['testing']:
            testing_datasets.extend(datasets['testing'][dataset_name])
            if options['train_on_val']:
                training_datasets.extend(datasets['testing'][dataset_name])
            # testing_datasets.extend(datasets['training'][dataset_name])

        # feature_matching_learned_classifier(options, training_datasets, testing_datasets)
        image_matching_learned_classifier(training_datasets, testing_datasets, options)  

if __name__ == '__main__':
    main(sys.argv)
