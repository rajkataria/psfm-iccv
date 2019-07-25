import datetime
import json
import math
import numpy as np
import os
import random
import socket
import sys
import time
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
# from mle import var, Normal, Model
# from sklearn.mixture import GMM
from scipy.integrate import trapz, simps
from argparse import ArgumentParser

import matching_classifiers

class Context:
    pass

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def plot_iris(num_rmatches, labels, counts):
    legend = []
    plt.clf()
    fig = plt.figure()
    # plt.xlim([0.0, 1.05])
    
    
    
    plt.subplot(2,2,1)
    plt.title('Filtered matches - inliers/outliers: {}/{}'.format(counts['inliers'], counts['outliers']))
    nbins = int((num_rmatches[labels >= 1].max() - num_rmatches[labels >= 1].min() + 1)/2.0)
    plt.xlabel('Number of rmatches')
    plt.ylabel('Count(inliers)={}'.format(len(np.where(labels >= 1)[0])))
    plt.hist(num_rmatches[labels >= 1], bins=nbins)
    
    plt.subplot(2,2,2)
    plt.title('Filtered matches - inliers/outliers: {}/{}'.format(counts['inliers'], counts['outliers']))
    nbins = int((num_rmatches[labels <= 0].max() - num_rmatches[labels <= 0].min() + 1)/2.0)
    plt.xlabel('Number of rmatches')
    plt.ylabel('Count (outliers)={}'.format(len(np.where(labels <= 0)[0])))
    plt.hist(num_rmatches[labels <= 0], bins=nbins)

    plt.subplot(2,2,3)
    plt.ylim([0.0, 100.0])
    plt.title('Filtered matches - inliers/outliers: {}/{}'.format(counts['inliers'], counts['outliers']))
    nbins = int((num_rmatches[labels >= 1].max() - num_rmatches[labels >= 1].min() + 1)/2.0)
    plt.xlabel('Number of rmatches')
    plt.ylabel('Count(inliers)={}'.format(len(np.where(labels >= 1)[0])))
    plt.hist(num_rmatches[labels >= 1], bins=nbins)
    
    plt.subplot(2,2,4)
    plt.ylim([0.0, 100.0])
    plt.title('Filtered matches - inliers/outliers: {}/{}'.format(counts['inliers'], counts['outliers']))
    nbins = int((num_rmatches[labels <= 0].max() - num_rmatches[labels <= 0].min() + 1)/2.0)
    plt.xlabel('Number of rmatches')
    plt.ylabel('Count (outliers)={}'.format(len(np.where(labels <= 0)[0])))
    plt.hist(num_rmatches[labels <= 0], bins=nbins)

    fig.set_size_inches(20.8, 11.8)
    fig.canvas.draw()
    fig_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.clf()
    return fig, fig_np


def distance_based_filter(dsets, fns, options):
    current_dset = None
    ris = []
    iris = []
    all_distances = []
    for i, dset in enumerate(dsets):
        if current_dset != dset:
            data = dataset.DataSet(dset)
            images = sorted(data.images())
            current_dset = dset
            mds_positions = data.load_mds_positions(label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration']))

        distance_matrix = euclidean_distances([mds_positions[fns[i][0]], mds_positions[fns[i][1]]])
        all_distances.append(distance_matrix[0,1])
        if distance_matrix[0,1] <= options['distance_threshold']:
            ris.append(i)
        else:
            iris.append(i)
        
    return np.array(ris), np.array(iris), np.array(all_distances)

def distance_thresholded_matching_results(datasets, options):
    mds_data_folder = options['mds_data_folder']
    mkdir_p(mds_data_folder)
    for i,t in enumerate(datasets):
        print ('\tDataset: {}'.format(t))
        data = dataset.DataSet(t)
        if not data.reconstruction_exists('reconstruction_gt.json'):
            continue
            
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
            = data.load_image_matching_dataset(robust_matches_threshold=options['image_matching_gt_threshold'], rmatches_min_threshold=0, \
                rmatches_max_threshold=10000, spl=10000, balance=options['balance'])
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

    ris, iris, all_distances = distance_based_filter(dsets_tr, fns_tr, options)
    # import pdb; pdb.set_trace()
    # f1 = plt.figure(1)
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.title('Inlier/Outlier Precision-Recall Curve (distance threshold: {}: tuples: {}/{})'.format(options['distance_threshold'], len(ris), len(labels_tr)), fontsize=18)

    # auc_s_t = timer()
    auc_pr_baseline, auc_roc_baseline, pr_baseline, roc_baseline = matching_classifiers.calculate_dataset_auc(num_rmatches_tr, labels_tr)#, color='green', ls='dashed', markers=[15, 16, 20])
    # try:
    auc_pr_distance_thresholded, auc_roc_distance_thresholded, pr_distance_thresholded, roc_distance_thresholded = matching_classifiers.calculate_dataset_auc(num_rmatches_tr[ris], labels_tr[ris])#, color='green', ls='dashed', markers=[15, 16, 20])
    # except:
    #     import pdb; pdb.set_trace()
    
    labels_zeroed = np.copy(labels_tr)
    labels_zeroed[iris] = 0
    auc_pr_zeroed, auc_roc_zeroed, pr_zeroed, roc_zeroed = matching_classifiers.calculate_dataset_auc(num_rmatches_tr, labels_zeroed)#, color='green', ls='dashed', markers=[15, 16, 20])

    # auc_e_t = timer()
    # aucpi_s_t = timer()
    # _, _, _, auc_per_image_per_dset_means_baseline, _, auc_per_image_mean_baseline, auc_roc_per_image_mean_baseline = \
    #     calculate_per_image_mean_auc(dsets_tr, fns_tr, num_rmatches_tr, labels_tr)
    # aucpi_e_t = timer()
    # ppi_s_t = timer()
    # _, _, _, _, mean_precision_per_image_baseline = calculate_per_image_precision_top_k(dsets_tr, fns_tr, num_rmatches_tr, labels_tr)
    # ppi_e_t = timer()
    # print ('\tBaseline (Train): AUC: {} ({}) / {} ({}) / {}'.format(\
    #     round(auc_pr_baseline,3), round(auc_roc_baseline,3), \
    #     round(auc_per_image_mean_baseline,3), round(auc_roc_per_image_mean_baseline,3), \
    #     round(mean_precision_per_image_baseline,3) \
    #     ))

    fig_prs, _ = matching_classifiers.plot_prs(pr_distance_thresholded, pr_baseline, auc_pr_distance_thresholded, auc_pr_baseline, markers=[15, 16, 20], markers_baseline=[15, 16, 20])
    fig_prs.savefig(os.path.join(mds_data_folder, 'distance-thresholded-image-matching-PR-{}-tuples-{}-{}-it-{}.png'.format(options['distance_threshold'], len(ris), len(labels_tr), options['iteration'] )))

    fig_prs_zeroed, _ = matching_classifiers.plot_prs(pr_zeroed, pr_baseline, auc_pr_zeroed, auc_pr_baseline, markers=[15, 16, 20], markers_baseline=[15, 16, 20])
    fig_prs_zeroed.savefig(os.path.join(mds_data_folder, 'distance-thresholded-image-matching-PR-zeroed-{}-tuples-{}-{}-it-{}.png'.format(options['distance_threshold'], len(labels_zeroed), len(labels_tr), options['iteration'])))

    counts = {'inliers': len(np.where(labels_tr >= 1)[0]), 'outliers': len(np.where(labels_tr <= 0)[0])}
    fig_prs_iris, _ = plot_iris(num_rmatches_tr[iris], labels_tr[iris], counts)
    fig_prs_iris.savefig(os.path.join(mds_data_folder, 'distance-thresholded-image-matching-PR-iris-{}-tuples-{}-{}-it-{}.png'.format(options['distance_threshold'], len(iris), len(labels_tr), options['iteration'])))

    auc_pr_distances, auc_roc_distances, pr_distances, roc_distances = matching_classifiers.calculate_dataset_auc(-1.0 * all_distances + 2.0, labels_tr)#, color='green', ls='dashed', markers=[15, 16, 20])
    fig_prs_distances, _ = matching_classifiers.plot_prs(pr_distances, pr_baseline, auc_pr_distances, auc_pr_baseline, markers=[1.0, 1.1, 1.2], markers_baseline=[15, 16, 20])
    fig_prs_distances.savefig(os.path.join(mds_data_folder, 'distance-thresholded-image-matching-PR-distances-it-{}.png'.format(options['iteration'])))


def path_results(data, options):
    ctx = Context()
    ctx.data = data
    # images = data.images()
    images = data.all_feature_maps()
    
    if not data.reconstruction_exists('reconstruction_gt.json'):
        return
    # closest_images = data.load_closest_images(label='rm-cost-lmds-False')
    # closest_images_lmds = data.load_closest_images(label='rm-cost-lmds-True')
    # closest_images_gt = data.load_closest_images(label='gt')
    max_k = len(images) - 1
    # import pdb; pdb.set_trace()
    mean_precisions_rm_cost_it_0 = np.zeros((max_k,))
    mean_precisions_rm_cost_it_1 = np.zeros((max_k,))
    mean_precisions_rm_cost_it_2 = np.zeros((max_k,))
    mean_precisions_rm_cost_it_3 = np.zeros((max_k,))

    mean_precisions_outlier_logp_it_0 = np.zeros((max_k,))
    mean_precisions_outlier_logp_it_1 = np.zeros((max_k,))
    mean_precisions_outlier_logp_it_2 = np.zeros((max_k,))
    mean_precisions_outlier_logp_it_3 = np.zeros((max_k,))

    mean_precisions_baseline = np.zeros((max_k,))
    
    cache_fn = os.path.join(options['mds_data_folder'], '{}_cache.json'.format(os.path.basename(data.data_path)))
    cache = {}
    if os.path.exists(cache_fn):
        with open(cache_fn,'r') as fin:
            cache = json.load(fin)

            mean_precisions_rm_cost_it_0 = cache['mean_precisions_rm_cost_it_0']
            mean_precisions_rm_cost_it_1 = cache['mean_precisions_rm_cost_it_1']
            mean_precisions_rm_cost_it_2 = cache['mean_precisions_rm_cost_it_2']
            mean_precisions_rm_cost_it_3 = cache['mean_precisions_rm_cost_it_3']

            mean_precisions_outlier_logp_it_0 = cache['mean_precisions_outlier_logp_it_0']
            mean_precisions_outlier_logp_it_1 = cache['mean_precisions_outlier_logp_it_1']
            mean_precisions_outlier_logp_it_2 = cache['mean_precisions_outlier_logp_it_2']
            mean_precisions_outlier_logp_it_3 = cache['mean_precisions_outlier_logp_it_3']
            # mean_precisions_with_sequences[k] = np.mean(precisions_with_sequences)
            mean_precisions_baseline = cache['mean_precisions_baseline']
    else:
        # Baseline metric
        np_images = np.array(sorted(images))
        rmatches_matrix = np.zeros((len(images), len(images)))
        closest_images_baseline = {}
        reverse_image_mapping = {}
        im_matches = {}

        cache_closest_images_rm_cost_it_0 = {}
        cache_closest_images_rm_cost_it_1 = {}
        cache_closest_images_rm_cost_it_2 = {}
        cache_closest_images_rm_cost_it_3 = {}

        cache_closest_images_outlier_logp_it_0 = {}
        cache_closest_images_outlier_logp_it_1 = {}
        cache_closest_images_outlier_logp_it_2 = {}
        cache_closest_images_outlier_logp_it_3 = {}

        cache_closest_images_gt = {}

        for im in sorted(images):
            _, _, im1_all_rmatches = data.load_all_matches(im)
            im_matches[im] = im1_all_rmatches
        for i,im1 in enumerate(sorted(images)):
            reverse_image_mapping[i] = im1
            for j,im2 in enumerate(sorted(images)):
                if im2 in im_matches[im1]:
                    rmatches_matrix[i,j] = len(im_matches[im1][im2])
                    rmatches_matrix[j,i] = rmatches_matrix[i,j]

        for i, im in enumerate(sorted(images)):
            order = np.argsort(-rmatches_matrix[:,i])
            closest_images_baseline[im] = np_images[order].tolist()

        for k in range(0, max_k):
            precisions_rm_cost_it_0 = []
            precisions_rm_cost_it_1 = []
            precisions_rm_cost_it_2 = []
            precisions_rm_cost_it_3 = []

            precisions_outlier_logp_it_0 = []
            precisions_outlier_logp_it_1 = []
            precisions_outlier_logp_it_2 = []
            precisions_outlier_logp_it_3 = []

            precisions_baseline = []
            # precisions_with_sequences = []
            for im in images:
                if not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 0)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 1)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 2)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 3)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 0)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 1)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 2)) or \
                    not data.closest_images_exists(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 3)) or \
                    not data.closest_images_exists(im, label='gt-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format(options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['gt'], options['lmds'], options['iteration'])):

                    continue




                if im not in cache_closest_images_rm_cost_it_0:
                    cache_closest_images_rm_cost_it_0[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 0))
                    cache_closest_images_rm_cost_it_1[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 1))
                    cache_closest_images_rm_cost_it_2[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 2))
                    cache_closest_images_rm_cost_it_3[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 3))

                    cache_closest_images_outlier_logp_it_0[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 0))
                    cache_closest_images_outlier_logp_it_1[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 1))
                    cache_closest_images_outlier_logp_it_2[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 2))
                    cache_closest_images_outlier_logp_it_3[im] = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 3))

                    cache_closest_images_gt[im] = data.load_closest_images(im, label='gt-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format(options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['gt'], options['lmds'], options['iteration']))



                # closest_images_rm_cost_it_0 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 0))
                # closest_images_rm_cost_it_1 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 1))
                # closest_images_rm_cost_it_2 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 2))
                # closest_images_rm_cost_it_3 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('rm-cost', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['rm-cost'], options['lmds'], 3))

                # closest_images_outlier_logp_it_0 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 0))
                # closest_images_outlier_logp_it_1 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 1))
                # closest_images_outlier_logp_it_2 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 2))
                # closest_images_outlier_logp_it_3 = data.load_closest_images(im, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format('outlier-logp', options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['outlier-logp'], options['lmds'], 3))

                # closest_images_gt = data.load_closest_images(im, label='gt-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}'.format(options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold']['gt'], options['lmds'], options['iteration']))
                closest_images_rm_cost_it_0 = cache_closest_images_rm_cost_it_0[im]
                closest_images_rm_cost_it_1 = cache_closest_images_rm_cost_it_1[im]
                closest_images_rm_cost_it_2 = cache_closest_images_rm_cost_it_2[im]
                closest_images_rm_cost_it_3 = cache_closest_images_rm_cost_it_3[im]

                closest_images_outlier_logp_it_0 = cache_closest_images_outlier_logp_it_0[im]
                closest_images_outlier_logp_it_1 = cache_closest_images_outlier_logp_it_1[im]
                closest_images_outlier_logp_it_2 = cache_closest_images_outlier_logp_it_2[im]
                closest_images_outlier_logp_it_3 = cache_closest_images_outlier_logp_it_3[im]

                closest_images_gt = cache_closest_images_gt[im]

                # if im not in closest_images or im not in closest_images_gt or im not in closest_images_lmds:
                #     continue

                common_images_baseline = set(closest_images_baseline[im][0:k]).intersection(set(closest_images_gt[1:k+1]))
                common_images_rm_cost_it_0 = set(closest_images_rm_cost_it_0[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                common_images_rm_cost_it_1 = set(closest_images_rm_cost_it_1[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                common_images_rm_cost_it_2 = set(closest_images_rm_cost_it_2[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                common_images_rm_cost_it_3 = set(closest_images_rm_cost_it_3[1:k+1]).intersection(set(closest_images_gt[1:k+1]))

                common_images_outlier_logp_it_0 = set(closest_images_outlier_logp_it_0[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                common_images_outlier_logp_it_1 = set(closest_images_outlier_logp_it_1[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                common_images_outlier_logp_it_2 = set(closest_images_outlier_logp_it_2[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                common_images_outlier_logp_it_3 = set(closest_images_outlier_logp_it_3[1:k+1]).intersection(set(closest_images_gt[1:k+1]))
                # common_images_with_sequences = set(closest_images_with_sequences[im][1:k+1]).intersection(set(closest_images_gt[im][1:k+1]))

                precisions_baseline.append(1.0* len(common_images_baseline) / (k+1))
                precisions_rm_cost_it_0.append(1.0* len(common_images_rm_cost_it_0) / (k+1))
                precisions_rm_cost_it_1.append(1.0* len(common_images_rm_cost_it_1) / (k+1))
                precisions_rm_cost_it_2.append(1.0* len(common_images_rm_cost_it_2) / (k+1))
                precisions_rm_cost_it_3.append(1.0* len(common_images_rm_cost_it_3) / (k+1))

                precisions_outlier_logp_it_0.append(1.0* len(common_images_outlier_logp_it_0) / (k+1))
                precisions_outlier_logp_it_1.append(1.0* len(common_images_outlier_logp_it_1) / (k+1))
                precisions_outlier_logp_it_2.append(1.0* len(common_images_outlier_logp_it_2) / (k+1))
                precisions_outlier_logp_it_3.append(1.0* len(common_images_outlier_logp_it_3) / (k+1))

                # precisions_with_sequences.append(1.0* len(common_images_with_sequences) / (k+1))
            mean_precisions_rm_cost_it_0[k] = np.mean(precisions_rm_cost_it_0)
            mean_precisions_rm_cost_it_1[k] = np.mean(precisions_rm_cost_it_1)
            mean_precisions_rm_cost_it_2[k] = np.mean(precisions_rm_cost_it_2)
            mean_precisions_rm_cost_it_3[k] = np.mean(precisions_rm_cost_it_3)

            mean_precisions_outlier_logp_it_0[k] = np.mean(precisions_outlier_logp_it_0)
            mean_precisions_outlier_logp_it_1[k] = np.mean(precisions_outlier_logp_it_1)
            mean_precisions_outlier_logp_it_2[k] = np.mean(precisions_outlier_logp_it_2)
            mean_precisions_outlier_logp_it_3[k] = np.mean(precisions_outlier_logp_it_3)
            # mean_precisions_with_sequences[k] = np.mean(precisions_with_sequences)
            mean_precisions_baseline[k] = np.mean(precisions_baseline)

        cache['mean_precisions_rm_cost_it_0'] = mean_precisions_rm_cost_it_0.tolist()
        cache['mean_precisions_rm_cost_it_1'] = mean_precisions_rm_cost_it_1.tolist()
        cache['mean_precisions_rm_cost_it_2'] = mean_precisions_rm_cost_it_2.tolist()
        cache['mean_precisions_rm_cost_it_3'] = mean_precisions_rm_cost_it_3.tolist()

        cache['mean_precisions_outlier_logp_it_0'] = mean_precisions_outlier_logp_it_0.tolist()
        cache['mean_precisions_outlier_logp_it_1'] = mean_precisions_outlier_logp_it_1.tolist()
        cache['mean_precisions_outlier_logp_it_2'] = mean_precisions_outlier_logp_it_2.tolist()
        cache['mean_precisions_outlier_logp_it_3'] = mean_precisions_outlier_logp_it_3.tolist()
        cache['mean_precisions_baseline'] = mean_precisions_baseline.tolist()

        with open(cache_fn, 'w') as fout:
            json.dump(cache, fout, sort_keys=True, indent=4, separators=(',', ': '))

    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_baseline, linewidth=3, color='g')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_rm_cost_it_0, linewidth=2.5, color='r', linestyle=':')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_rm_cost_it_1, linewidth=2, color='c', linestyle=':')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_rm_cost_it_2, linewidth=1.5, color='k', linestyle=':')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_rm_cost_it_3, linewidth=1, color='b', linestyle=':')

    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_outlier_logp_it_0, linewidth=2.5, color='r', linestyle='-.')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_outlier_logp_it_1, linewidth=2, color='c', linestyle='-.')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_outlier_logp_it_2, linewidth=1.5, color='k', linestyle='-.')
    plt.plot(np.linspace(1,max_k,max_k), mean_precisions_outlier_logp_it_3, linewidth=1, color='b', linestyle='-.')

    plt.title('Ranking score of 2d embedding ({})'.format(data.data_path.split('/')[-2]), fontsize=options['fontsize'])
    plt.xlabel('Top k closest distances', fontsize=options['fontsize'])
    plt.ylabel('% common with ground-truth', fontsize=options['fontsize'])
    
    # sequence_legends = []
    # # for seq_cost_factor in [0.25, 1.0, 5.0, 10.0]:
    # for seq_cost_factor in []:
    #     for lmds in [False, True]:
    #         mean_precisions_with_sequences = np.zeros((max_k,))
    #         closest_images_with_sequences = data.load_closest_images(label='rm-seq-cost-{}-lmds-{}'.format(seq_cost_factor, lmds))
    #         for k in range(0, max_k):
    #             # precisions = []
    #             # precisions_baseline = []
    #             precisions_with_sequences = []
    #             for im in images:
    #                 if im not in closest_images_it_0 or im not in closest_images_gt:
    #                     continue
    #                 common_images_with_sequences = set(closest_images_with_sequences[im][1:k+1]).intersection(set(closest_images_gt[im][1:k+1]))
    #                 precisions_with_sequences.append(1.0* len(common_images_with_sequences) / (k+1))

    #             mean_precisions_with_sequences[k] = np.mean(precisions_with_sequences)
    #         sequence_legends.append('Our embedding with sequences ({}, LMDS: {})'.format(seq_cost_factor, lmds))
    #         plt.plot(np.linspace(1,max_k,max_k), mean_precisions_with_sequences)

    legend = ['Baseline (rmatches)', \
        'rmatches cost (Iteration 0)', 'rmatches cost (Iteration 1)', 'rmatches cost (Iteration 2)', 'rmatches cost (Iteration 3)', \
        'outlier p (Iteration 0)', 'outlier p (Iteration 1)', 'outlier p (Iteration 2)', 'outlier p (Iteration 3)', \
        ]
    # legend.extend(sequence_legends)
    plt.legend(legend,  loc='lower right',  shadow=True, fontsize=options['fontsize'])

    plt.axvline(x=(min(len(data.images())-1, 10)), linewidth=6, color='#AA0000', linestyle='-')
    plt.axvline(x=(min(len(data.images())-1, 20)), linewidth=6, color='#00AA00', linestyle='--')
    plt.axvline(x=((len(data.images())-1)*0.1), linewidth=4, color='#0000AA', linestyle='-')
    plt.axvline(x=((len(data.images())-1)*0.2), linewidth=4, color='#AAAA00', linestyle='--')
    plt.axvline(x=(min(len(data.images())-1, max(10, len(data.images())*0.1))), linewidth=2, color='#00AAAA', linestyle='-')
    plt.axvline(x=(min(len(data.images())-1, max(10, len(data.images())*0.2))), linewidth=2, color='#AA00AA', linestyle='--')
    plt.axvline(x=(min(len(data.images())-1, max(20, len(data.images())*0.2))), linewidth=1, color='#000000', linestyle='-')
    plt.axvline(x=(min(len(data.images())-1, max(20, len(data.images())*0.1))), linewidth=1, color='#AAAAAA', linestyle='--')

    # plt.axvline(x=1, linewidth=6, color='#AA0000', linestyle='-')
    # plt.axvline(x=3, linewidth=6, color='#00AA00', linestyle='--')
    # plt.axvline(x=5, linewidth=4, color='#0000AA', linestyle='-')
    # plt.axvline(x=7, linewidth=4, color='#AAAA00', linestyle='--')
    # plt.axvline(x=9, linewidth=2, color='#00AAAA', linestyle='-')
    # plt.axvline(x=11, linewidth=2, color='#AA00AA', linestyle='--')
    # plt.axvline(x=13, linewidth=1, color='#000000', linestyle='-')
    # plt.axvline(x=15, linewidth=1, color='#AAAAAA', linestyle='--')

    if not options['aggregate']:
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.savefig(os.path.join(data.data_path, 'results', 'closest-images-ranking.png'))


def main():
    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('-d', '--dataset', help='dataset to scan')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')

    parser.add_argument('--debug', dest='debug', action='store_true', help='show mask')
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    import opensfm
    from opensfm import commands, features, dataset, matching, classifier, reconstruction, types, io
    from opensfm.commands.validate_results import ransac_based_ate_evaluation
    global ransac_based_ate_evaluation, features, matching, classifier, dataset, types

    eth3d_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/facade/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/kicker/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/meadow/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lecture_room/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/living_room/',
    ]

    uiuctag_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor2_hall/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_ccw/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_cw/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_stairs/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_wall/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_stairs/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_all/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_atrium/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_backward/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_forward/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_all/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_atrium/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_backward/', 
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_forward/'
    ]

    tanks_and_temples_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Barn',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Caterpillar',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Church',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courthouse',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ignatius',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Meetingroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Truck',
    ]

    tum_rgbd_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz/'
        ]

    yan_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/books/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/cereal/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/cup/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/desk/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/oats/',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/street/',
    ]

    options = {
        # 'plot': 'distance-based-thresholding',
        'plot': 'ranks',
        'image_matching_gt_threshold': 15,
        'balance': False,
        'mds_data_folder': 'data/mds-path-analysis',
        'distance_thresholds': [0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5],
        # distance_thresholds': [0.5],
        # 'distance_thresholds': [ 0.75 ],
        'shortest_path_label': 'rm-cost',
        'PCA-n_components': 2,
        'MDS-n_components': 2,
        # 'edge_threshold': 10000000000,#1.0/10.0,
        'edge_threshold': {
            'rm-cost': '10000000000',
            'gt': '10000000000',
            'outlier-logp': '1e-10',
        },
        'lmds': False,
        'iteration': 0,
        'aggregate': True if parser_options.dataset is None else False,
        'debug': True
    }

    if options['aggregate']:
        options['fontsize'] = 32
    else:
        options['fontsize'] = 20

    datasets = [
        # '/hdd/Research/psfm-iccv/data/temp-recons/boulders',
        # '/hdd/Research/psfm-iccv/data/temp-recons/exhibition_hall',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/boulders',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall',
        # '/hdd/Research/psfm-iccv/data/temp-recons/ece_floor3_loop_cw',
        # '/hdd/Research/psfm-iccv/data/temp-recons/ece_floor3_loop_ccw',
    ]
    datasets = tanks_and_temples_datasets

    if options['plot'] == 'distance-based-thresholding' and options['aggregate']:
        mkdir_p(options['mds_data_folder'])
        for d in options['distance_thresholds']:
            options['distance_threshold'] = d
            distance_thresholded_matching_results(datasets, options)
    elif options['plot'] == 'distance-based-thresholding':
        mkdir_p(options['mds_data_folder'])
        distance_thresholded_matching_results([parser_options.dataset], options)
    elif options['aggregate']:
        for i,d in enumerate(datasets):
            data = dataset.DataSet(d)
            if len(datasets) < 5:
                plt.subplot(1, int(math.ceil(len(datasets)/1.0)), i+1)
            else:
                options['fontsize'] = 18
                plt.subplot(4, int(math.ceil(len(datasets)/4.0)), i+1)
            
            path_results(data, options)
        # plt.tight_layout()
        fig = plt.gcf()
        fig.patch.set_facecolor('white')
        if len(datasets) < 5:
            fig.set_size_inches(32, 17)
        else:
            fig.set_size_inches(65, 35)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0.1, wspace = 0.1)
        plt.savefig(os.path.join(options['mds_data_folder'],'closest-images-ranking-aggregated.png'), bbox_inches = 'tight', pad_inches = 0)
    else:
        data = dataset.DataSet(parser_options.dataset)
        path_results(data, options)

if __name__ == '__main__':
    main()