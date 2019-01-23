import gflags
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
from sklearn.feature_selection import SelectFromModel

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

def calculate_accuracy(y, y_gt, color, ls, thresholds = np.linspace(0.0, 1.0, 101)):
    if ls == 'solid':
        width = 50
    else:
        width = 10

    precision, recall, threshs = sklearn.metrics.precision_recall_curve(y_gt, y)
    auc = sklearn.metrics.average_precision_score(y_gt, y)
    plt.step(recall, precision, color=color, alpha=0.2 * width,
        where='post')
    return auc

def get_postfix(datasets):
  postfix = ''
  for t in datasets:
    if 'yeh' in t and 'Yeh' not in postfix:
      postfix += 'Yeh+'
    if 'ece' in t and 'ECE' not in postfix:
      postfix += 'ECE+'
    if 'Barn' in t:
      postfix += 'Barn+'
    if 'Church' in t:
      postfix += 'Church+'
    if 'Caterpillar' in t:
      postfix += 'Caterpillar+'
    if 'Ignatius' in t:
      postfix += 'Ignatius+'
    if 'Courthouse' in t:
      postfix += 'Courthouse+'
    if 'Meetingroom' in t:
      postfix += 'Meetingroom+'
    if 'Truck' in t:
      postfix += 'Truck+'
  return postfix

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
    auc_baseline_train = calculate_accuracy(-1.0 * max_distances, labels, color='green', ls='dashed', thresholds=np.linspace(0.0,1.0,101))
    _, _, _, _, regr_bdt, y = classifier.classify_boosted_dts_feature_match([fns, dists1, dists2, size1, size2, angle1, angle2, labels, True, None, options])
    auc_bdts_train = calculate_accuracy(y, labels,'r','dashed')
    training_postfix = get_postfix(training_datasets)
    data_folder = 'data/feature-matching-classifiers-results'
    mkdir_p(data_folder)
    save_classifier(regr_bdt, os.path.join(data_folder, '{}{}-{}'.format(training_postfix, options['max_depth'], options['n_estimators'])))
    
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
    auc_baseline_test = calculate_accuracy(-1.0 * max_distances, labels, color='blue', ls='dashed', thresholds=np.linspace(0.0,1.0,101))
    _, _, _, _, regr_bdt, y = classifier.classify_boosted_dts_feature_match([fns, dists1, dists2, size1, size2, angle1, angle2, labels, False, regr_bdt, options])
    auc_bdts_test = calculate_accuracy(y, labels,'r','solid')
  
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
    plt.savefig(os.path.join(data_folder, 'feature-matching-PR-{}{}-{}_{}.png'.format(\
        training_postfix, options['max_depth'], options['n_estimators'], str(datetime.datetime.now()) 
        ))
    )

def image_matching_learned_classifier(training_datasets, testing_datasets, options={}):
    epsilon = 0.00000001
    #################################################################################################################################
    #################################################################################################################################
    ############################################################ Training ###########################################################
    #################################################################################################################################
    #################################################################################################################################    
    for i,t in enumerate(training_datasets):
        data = dataset.DataSet(t)
        _fns, [_R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
            _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
            _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, _num_gt_inliers, _labels] \
            = data.load_image_matching_dataset(robust_matches_threshold=15, rmatches_min_threshold=options['image_match_classifier_min_match'], \
                rmatches_max_threshold=options['image_match_classifier_max_match'])

        if i == 0:
            fns_tr, R11s_tr, R12s_tr, R13s_tr, R21s_tr, R22s_tr, R23s_tr, R31s_tr, R32s_tr, R33s_tr, num_rmatches_tr, num_matches_tr, spatial_entropy_1_8x8_tr, \
                spatial_entropy_2_8x8_tr, spatial_entropy_1_16x16_tr, spatial_entropy_2_16x16_tr, pe_histogram_tr, pe_polygon_area_percentage_tr, \
                nbvs_im1_tr, nbvs_im2_tr, te_histogram_tr, ch_im1_tr, ch_im2_tr, vt_rank_percentage_im1_im2_tr, vt_rank_percentage_im2_im1_tr, \
                num_gt_inliers_tr, labels_tr \
                = _fns, _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
                _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
                _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, _num_gt_inliers,_labels
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
            num_gt_inliers_tr = np.concatenate((num_gt_inliers_tr, _num_gt_inliers), axis=0)
            labels_tr = np.concatenate((labels_tr, _labels), axis=0)
    labels_tr[labels_tr < 0] = 0
    print ('Training datasets loaded - Tuples: {}'.format(len(labels_tr)))

    #################################################################################################################################
    #################################################################################################################################
    ############################################################ Testing ############################################################
    #################################################################################################################################
    #################################################################################################################################    
    for i,t in enumerate(testing_datasets):
        data = dataset.DataSet(t)
        _fns, [_R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
            _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
            _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, _num_gt_inliers, _labels] \
            = data.load_image_matching_dataset(robust_matches_threshold=15, rmatches_min_threshold=options['image_match_classifier_min_match'], \
                rmatches_max_threshold=options['image_match_classifier_max_match'])

        if i == 0:
            fns_te, R11s_te, R12s_te, R13s_te, R21s_te, R22s_te, R23s_te, R31s_te, R32s_te, R33s_te, num_rmatches_te, num_matches_te, spatial_entropy_1_8x8_te, \
                spatial_entropy_2_8x8_te, spatial_entropy_1_16x16_te, spatial_entropy_2_16x16_te, pe_histogram_te, pe_polygon_area_percentage_te, \
                nbvs_im1_te, nbvs_im2_te, te_histogram_te, ch_im1_te, ch_im2_te, vt_rank_percentage_im1_im2_te, vt_rank_percentage_im2_im1_te, \
                num_gt_inliers_te, labels_te \
                = _fns, _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
                _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
                _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, _num_gt_inliers, _labels
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
            num_gt_inliers_te = np.concatenate((num_gt_inliers_te, _num_gt_inliers), axis=0)
            labels_te = np.concatenate((labels_te, _labels), axis=0)
    labels_te[labels_te < 0] = 0
    print ('Testing datasets loaded - Tuples: {}'.format(len(labels_te)))
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
    auc_baseline_train = calculate_accuracy(num_rmatches_tr, labels_tr, color='green', ls='dashed', thresholds=np.linspace(0,4000,4001).astype(np.int))
    print ('Baseline (Train): AUC: {}'.format(auc_baseline_train))
    auc_baseline_test = calculate_accuracy(num_rmatches_te, labels_te, color='red', ls='dashed', thresholds=np.linspace(0,4000,4001).astype(np.int))
    print ('Baseline (Test): AUC: {}'.format(auc_baseline_test))
    legends = ['{} : {}'.format('Baseline (Train)', auc_baseline_train), '{} : {}'.format('Baseline (Test)', auc_baseline_test)]
    data_folder = 'data/image-matching-classifiers-results/image-matching-classifiers-classifier-{}-max_depth-{}-n_estimators-{}-thresholds-{}-{}'.format( \
        options['classifier'], options['max_depth'], options['n_estimators'], options['image_match_classifier_min_match'], \
        options['image_match_classifier_max_match'])

    mkdir_p(data_folder)

    # # Used for LCCs (initial values)
    # classifier_training_scores = num_rmatches_tr
    # classifier_testing_scores = num_rmatches_te
    # classifier_training_threshold = 20
    # classifier_testing_scores = 20

    exps = [
        ['RM'],
        ['TE'],
        ['RM', 'TE'],
        ['RM', 'VD'],
        ['RM', 'PE'],
        ['RM', 'NBVS'],
        ['RM', 'VT'],
        ['RM', 'SE'],
        ['RM', 'HIST'],
        ['RM', 'VD', 'SE', 'PE'],
        ['RM', 'VD', 'SE', 'TE', 'PE']

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
        print ('**************************************************************************************************************************')
        print ('**************************************************************************************************************************')
        print ('******************************* Performing experiment :{} *******************************'.format(exp))
        print ('**************************************************************************************************************************')
        print ('**************************************************************************************************************************')
        for mode in ['train', 'test']:

            fns = fns_tr if mode == 'train' else fns_te
            R11s = R11s_tr if mode == 'train' else R11s_te
            R12s = R12s_tr if mode == 'train' else R12s_te
            R13s = R13s_tr if mode == 'train' else R13s_te
            R21s = R21s_tr if mode == 'train' else R21s_te
            R22s = R22s_tr if mode == 'train' else R22s_te
            R23s = R23s_tr if mode == 'train' else R23s_te
            R31s = R31s_tr if mode == 'train' else R31s_te
            R32s = R32s_tr if mode == 'train' else R32s_te
            R33s = R33s_tr if mode == 'train' else R33s_te
            num_rmatches = num_rmatches_tr if mode == 'train' else num_rmatches_te
            num_matches = num_matches_tr if mode == 'train' else num_matches_te
            spatial_entropy_1_8x8 = spatial_entropy_1_8x8_tr if mode == 'train' else spatial_entropy_1_8x8_te
            spatial_entropy_2_8x8 = spatial_entropy_2_8x8_tr if mode == 'train' else spatial_entropy_2_8x8_te
            spatial_entropy_1_16x16 = spatial_entropy_1_16x16_tr if mode == 'train' else spatial_entropy_1_16x16_te
            spatial_entropy_2_16x16 = spatial_entropy_2_16x16_tr if mode == 'train' else spatial_entropy_2_16x16_te
            pe_histogram = pe_histogram_tr if mode == 'train' else pe_histogram_te
            pe_polygon_area_percentage = pe_polygon_area_percentage_tr if mode == 'train' else pe_polygon_area_percentage_te
            nbvs_im1 = nbvs_im1_tr if mode == 'train' else nbvs_im1_te
            nbvs_im2 = nbvs_im2_tr if mode == 'train' else nbvs_im2_te
            te_histogram = te_histogram_tr if mode == 'train' else te_histogram_te
            ch_im1 = ch_im1_tr if mode == 'train' else ch_im1_te
            ch_im2 = ch_im2_tr if mode == 'train' else ch_im2_te            
            vt_rank_percentage_im1_im2 = vt_rank_percentage_im1_im2_tr if mode == 'train' else vt_rank_percentage_im1_im2_te
            vt_rank_percentage_im2_im1 = vt_rank_percentage_im2_im1_tr if mode == 'train' else vt_rank_percentage_im2_im1_te
            num_gt_inliers = num_gt_inliers_tr if mode == 'train' else num_gt_inliers_te
            labels = labels_tr if mode == 'train' else labels_te
            train_mode = True if mode == 'train' else False

            plt.figure(1)
            if 'RM' not in exp:
                num_rmatches = np.zeros((len(labels),1))
            if 'TM' not in exp:
                num_matches = np.zeros((len(labels),1))
            if 'NBVS' not in exp:
                nbvs_im1 = np.zeros((len(labels),1))
                nbvs_im2 = np.zeros((len(labels),1))
            if 'SE' not in exp:
                spatial_entropy_1_8x8 = np.zeros((len(labels),1))
                spatial_entropy_2_8x8 = np.zeros((len(labels),1))
                spatial_entropy_1_16x16 = np.zeros((len(labels),1))
                spatial_entropy_2_16x16 = np.zeros((len(labels),1))

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

            if 'TE' not in exp:
                te_histogram = np.zeros(te_histogram.shape)
            if 'PE' not in exp:
                pe_histogram = np.zeros(pe_histogram.shape)
                pe_polygon_area_percentage = np.zeros(pe_polygon_area_percentage.shape)
            if 'HIST' not in exp:
                ch_im1 = np.zeros(ch_im1.shape)
                ch_im2 = np.zeros(ch_im2.shape)
            if 'VT' not in exp:
                vt_rank_percentage_im1_im2 = np.zeros(vt_rank_percentage_im1_im2.shape)
                vt_rank_percentage_im2_im1 = np.zeros(vt_rank_percentage_im2_im1.shape)

            arg = [ \
                fns, R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, num_rmatches, num_matches, spatial_entropy_1_8x8, \
                spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, pe_histogram, pe_polygon_area_percentage, \
                nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, labels, \
                train_mode, trained_classifier, options
            ]

            _, _, regr_bdt, scores, _ = classifier.classify_boosted_dts_image_match(arg)
            print ("\tFinished Learning/Classifying Boosted Decision Tree")  
            
            if not train_mode:
                auc = calculate_accuracy(scores, labels, 'black', 'solid' if not train_mode else 'dashed')
                trained_classifier = None
                # classifier_testing_scores = scores
                # classifier_testing_threshold = 0.3
                # legends.append('{} : {}'.format(exp, auc))
                legends.append('{} : {} : {}'.format(mode, exp, auc))
                print ('\tExperiment: {} AUC: {}'.format(exp, auc))
            else:
                auc = 0.0
                trained_classifier = regr_bdt
                # classifier_training_scores = scores
                # classifier_training_threshold = 0.3
                save_classifier(regr_bdt, os.path.join(data_folder, '{}+{}{}-{}-thresholds-{}-{}'.format(\
                    '+'.join(exp), classifier_postfix, options['max_depth'], options['n_estimators'], \
                    options['image_match_classifier_min_match'], options['image_match_classifier_max_match'])))
  
    plt.legend(legends, loc='lower left', shadow=True, fontsize=18)

    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(data_folder, 'image-matching-PR-{}{}-{}_{}.png'.format(\
        classifier_postfix, options['max_depth'], options['n_estimators'], str(datetime.datetime.now()))))
    plt.gcf().clear()

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    parser.add_argument('-n', '--n_estimators', help='')
    parser.add_argument('-d', '--max_depth', help='')
    parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    parser.add_argument('-x', '--image_match_classifier_max_match', help='')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset

    training_datasets = [
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Barn',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Caterpillar',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Church',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Ignatius',

        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0000',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0001',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0002',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0065',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0071',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0073',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0088',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0089',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0098',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0100',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0102',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0112',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0116',
        # '/hdd/Research/psfm-iccv/data/GTAV_540/0118',

        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_360',
        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk',
        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2',
        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor',
        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant',
        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_room',
        # '/hdd/Research/psfm-iccv/data/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy',

        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/courtyard',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/delivery_area',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/electro',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/facade',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/kicker',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/meadow',
    ]

    testing_datasets = [
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Meetingroom',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Truck',
        
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0060',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0061',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0062',

        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/exhibition_hall',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/lecture_room',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/living_room',
    ]

    options = {
        'classifier': 'BDT', \
        'max_depth': int(parser_options.max_depth),
        'n_estimators': int(parser_options.n_estimators),
        'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        'feature_selection': False
    }
    feature_matching_learned_classifier(options, training_datasets, testing_datasets)
    # image_matching_learned_classifier(training_datasets, testing_datasets, options)  

if __name__ == '__main__':
    main(sys.argv)
