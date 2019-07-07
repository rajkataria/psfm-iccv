import json
import itertools
import math
import numpy as np
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import rcParams
from matplotlib.image import NonUniformImage
from matplotlib.colors import LogNorm
import seaborn as sns; sns.set()
import scipy
import sys
from sklearn.externals import joblib
from sklearn.neighbors.kde import KernelDensity

from argparse import ArgumentParser
# import matching_classifiers # import load_classifier, calculate_per_image_mean_auc, calculate_dataset_auc, mkdir_p
from multiprocessing import Pool
from scipy import special
# import convnet

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def calculate_track_inliers(arg):
    t, dataset_track_raw_data_fn, dataset_track_inlier_analysis_fn, options = arg

    data = dataset.DataSet(t)
    if options['gt']:
        graph = data.load_tracks_graph('tracks-gt-matches.csv')
    else:
        graph = data.load_tracks_graph('tracks.csv')

    image_matching_results_baseline = data.load_image_matching_results(robust_matches_threshold=15, classifier='BASELINE')
    image_matching_results_classifier = data.load_image_matching_results(robust_matches_threshold=15, classifier='CONVNET')
    image_matching_results_gt = data.load_groundtruth_image_matching_results(robust_matches_threshold=15)
    feature_matching_results_baseline = {}
    feature_matching_results_classifier = {}
    feature_matching_results_gt = {}

    with open(dataset_track_raw_data_fn, 'r') as fi:
        track_data = json.load(fi)
        all_robust_matches = {}
        track_degrees = {}
        track_degrees_baseline = {}
        track_degrees_classifier = {}
        track_degrees_gt = {}
        track_degrees_baseline_max_score = {}
        track_degrees_baseline_min_score = {}
        track_degrees_baseline_max_rmatches = {}
        track_degrees_baseline_min_rmatches = {}
        track_degrees_classifier_max_score = {}
        track_degrees_classifier_min_score = {}
        total_tracks_dataset = 0
        for _, str_track_length in enumerate(sorted(track_data['total_tracks'])):
            # if int(str_track_length) == 2:
            #     continue
            # print ('\tTrack length: {} -- Number of tracks: {}'.format(str_track_length, track_data['total_track_count'][str_track_length]))
            for ii, t in enumerate(track_data['total_tracks'][str_track_length]):
                if str_track_length not in track_degrees:
                    track_degrees[str_track_length] = {}
                    track_degrees_baseline[str_track_length] = {}
                    track_degrees_classifier[str_track_length] = {}
                    track_degrees_gt[str_track_length] = {}
                    
                    track_degrees_baseline_max_score[str_track_length] = {}
                    track_degrees_baseline_min_score[str_track_length] = {}
                    track_degrees_baseline_max_rmatches[str_track_length] = {}
                    track_degrees_baseline_min_rmatches[str_track_length] = {}
                    track_degrees_classifier_max_score[str_track_length] = {}
                    track_degrees_classifier_min_score[str_track_length] = {}

                total_matches = 0
                total_matches_sum_baseline = 0.0
                total_matches_sum_classifier = 0.0
                total_matches_sum_gt = 0.0

                total_matches_sum_baseline_max_score = -100000.0
                total_matches_sum_baseline_min_score = 100000.0
                # total_matches_sum_baseline_max_rmatches = -100000.0
                # total_matches_sum_baseline_min_rmatches = 100000.0
                total_matches_sum_classifier_max_score = -100000.0
                total_matches_sum_classifier_min_score = 100000.0

                for i, im1 in enumerate(sorted(graph[t].keys())):
                    # if im1 != '000365.jpg':
                    #     continue
                    if not data.feature_matching_results_exists(im1, lowes_ratio_threshold=options['lowes_ratio_threshold'], classifier='BASELINE'):
                        continue

                    if im1 not in feature_matching_results_baseline:
                        feature_matching_results_baseline[im1] = data.load_feature_matching_results(im1, lowes_ratio_threshold=options['lowes_ratio_threshold'], classifier='BASELINE')
                        feature_matching_results_classifier = feature_matching_results_baseline
                        feature_matching_results_gt = feature_matching_results_baseline
                        # feature_matching_results_classifier[im1] = data.load_feature_matching_results(im1, lowes_ratio_threshold=options['lowes_ratio_threshold'], classifier='BASELINE')
                        # feature_matching_results_gt[im1] = data.load_feature_matching_results(im1, lowes_ratio_threshold=options['lowes_ratio_threshold'], classifier='BASELINE')

                    fid1 = graph[t][im1]['feature_id']
                    if im1 not in all_robust_matches:
                        _, _, im1_all_robust_matches = data.load_all_matches(im1)
                        all_robust_matches[im1] = im1_all_robust_matches
                    for j, im2 in enumerate(sorted(graph[t].keys())):
                        # if im2 != '000371.jpg':
                        #     continue
                        if j <= i:
                            continue
                        if im2 not in all_robust_matches[im1]:
                            continue

                        fid2 = graph[t][im2]['feature_id']
                        rmatches = all_robust_matches[im1][im2]
                        if len(rmatches) == 0:
                            continue

                        # if len(rmatches) < options['rmatches_threshold']:
                        #     continue

                        # import pdb; pdb.set_trace()
                        if [fid1, fid2] in rmatches[:,0:2].tolist():
                            total_matches += 1
                            try:
                                total_matches_sum_baseline += image_matching_results_baseline[im1][im2]['score'] * feature_matching_results_baseline[im1][im2][fid1][fid2]['score']
                            except KeyError:
                               continue 
                                # print ('{} | {} - {} : {} - {}'.format(data.data_path, im1, im2, fid1, fid2))
                                # print (feature_matching_results_baseline[im1].keys())
                                # print (len(feature_matching_results_baseline[im1][im2].keys()))
                                # print (feature_matching_results_baseline[im1][im2][fid1])
                                # import pdb; pdb.set_trace()
                            total_matches_sum_classifier += image_matching_results_classifier[im1][im2]['score'] * feature_matching_results_classifier[im1][im2][fid1][fid2]['score']
                            total_matches_sum_gt += image_matching_results_gt[im1][im2]['score'] * feature_matching_results_gt[im1][im2][fid1][fid2]['score']

                            total_matches_sum_baseline_max_score = np.maximum(image_matching_results_baseline[im1][im2]['score'] * feature_matching_results_baseline[im1][im2][fid1][fid2]['score'], total_matches_sum_baseline_max_score)
                            total_matches_sum_baseline_min_score = np.minimum(image_matching_results_baseline[im1][im2]['score'] * feature_matching_results_baseline[im1][im2][fid1][fid2]['score'], total_matches_sum_baseline_min_score)
                            # total_matches_sum_baseline_max_rmatches = np.maximum(image_matching_results_baseline[im1][im2]['num_rmatches'], total_matches_sum_baseline_max_rmatches)
                            # total_matches_sum_baseline_min_rmatches = np.minimum(image_matching_results_baseline[im1][im2]['num_rmatches'], total_matches_sum_baseline_min_rmatches)
                            total_matches_sum_classifier_max_score = np.maximum(image_matching_results_classifier[im1][im2]['score'] * feature_matching_results_classifier[im1][im2][fid1][fid2]['score'], total_matches_sum_classifier_max_score)
                            total_matches_sum_classifier_min_score = np.minimum(image_matching_results_classifier[im1][im2]['score'] * feature_matching_results_classifier[im1][im2][fid1][fid2]['score'], total_matches_sum_classifier_min_score)

                track_degrees[str_track_length][t] = total_matches
                track_degrees_baseline[str_track_length][t] = total_matches_sum_baseline
                track_degrees_classifier[str_track_length][t] = total_matches_sum_classifier
                track_degrees_gt[str_track_length][t] = total_matches_sum_gt

                track_degrees_baseline_max_score[str_track_length][t] = total_matches_sum_baseline_max_score
                track_degrees_baseline_min_score[str_track_length][t] = total_matches_sum_baseline_min_score
                # track_degrees_baseline_max_rmatches[str_track_length][t] = total_matches_sum_baseline_max_rmatches
                # track_degrees_baseline_min_rmatches[str_track_length][t] = total_matches_sum_baseline_min_rmatches
                track_degrees_classifier_max_score[str_track_length][t] = total_matches_sum_classifier_max_score
                track_degrees_classifier_min_score[str_track_length][t] = total_matches_sum_classifier_min_score

        with open(dataset_track_inlier_analysis_fn, 'w') as fout:
            json.dump({
                'track_degrees': track_degrees, 
                'track_degrees_baseline': track_degrees_baseline, 
                'track_degrees_classifier': track_degrees_classifier,
                'track_degrees_gt': track_degrees_gt, 
                'track_degrees_baseline_max_score': track_degrees_baseline_max_score,
                'track_degrees_baseline_min_score': track_degrees_baseline_min_score,
                # 'track_degrees_baseline_max_rmatches': track_degrees_baseline_max_rmatches,
                # 'track_degrees_baseline_min_rmatches': track_degrees_baseline_min_rmatches,
                'track_degrees_classifier_max_score': track_degrees_classifier_max_score,
                'track_degrees_classifier_min_score': track_degrees_classifier_min_score
                }, fout, sort_keys=True, indent=4, separators=(',', ': '))

def track_inlier_analysis(datasets, options):
    colors = ['r','g','b','c','k','m','y']
    data_folder = 'data/track_analysis'
    processes = options['processes']
    epsilon = 0.00000000001
    args = []

    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        dataset_track_raw_data_fn = os.path.join(data_folder, 'track_raw_data_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))
        dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))

        # if os.path.isfile(dataset_track_inlier_analysis_fn):
        #     continue
        args.append([t, dataset_track_raw_data_fn, dataset_track_inlier_analysis_fn, options])

    p = Pool(processes)
    if processes == 1:
        for arg in args:
            calculate_track_inliers(arg)
    else:
        p.map(calculate_track_inliers, args)
        p.close()

    print ('Finished calculating track inliers')

    # calculate_and_plot_track_inliers_histograms()
    inlier_track_degree_sum = {}
    inlier_track_degree_count = {}
    inlier_track_degree_count_per_match = {}
    inlier_track_degree_count_per_match_baseline = {}
    inlier_track_degree_count_per_match_classifier = {}
    inlier_track_degree_count_per_match_gt = {}
    inlier_track_degree_count_per_match_baseline_max_score = {}
    inlier_track_degree_count_per_match_baseline_min_score = {}
    # inlier_track_degree_count_per_match_baseline_max_rmatches = {}
    # inlier_track_degree_count_per_match_baseline_min_rmatches = {}
    inlier_track_degree_count_per_match_classifier_max_score = {}
    inlier_track_degree_count_per_match_classifier_min_score = {}
    inlier_tracks_baseline = {}
    inlier_tracks_baseline_scores = {}
    inlier_tracks_classifier_scores = {}
    inlier_tracks_baseline_max_scores = {}
    inlier_tracks_baseline_min_scores = {}
    inlier_tracks_classifier_max_scores = {}
    inlier_tracks_classifier_min_scores = {}

    outlier_track_degree_sum = {}
    outlier_track_degree_count = {}
    outlier_track_degree_count_per_match = {}
    outlier_track_degree_count_per_match_baseline = {}
    outlier_track_degree_count_per_match_classifier = {}
    outlier_track_degree_count_per_match_gt = {}
    outlier_track_degree_count_per_match_baseline_max_score = {}
    outlier_track_degree_count_per_match_baseline_min_score = {}
    # outlier_track_degree_count_per_match_baseline_max_rmatches = {}
    # outlier_track_degree_count_per_match_baseline_min_rmatches = {}
    outlier_track_degree_count_per_match_classifier_max_score = {}
    outlier_track_degree_count_per_match_classifier_min_score = {}
    outlier_tracks_baseline = {}
    outlier_tracks_baseline_scores = {}
    outlier_tracks_classifier_scores = {}
    outlier_tracks_baseline_max_scores = {}
    outlier_tracks_baseline_min_scores = {}
    outlier_tracks_classifier_max_scores = {}
    outlier_tracks_classifier_min_scores = {}


    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        dataset_track_raw_data_fn = os.path.join(data_folder, 'track_raw_data_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))
        dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))

        with open(dataset_track_raw_data_fn, 'r') as fi:
            track_data = json.load(fi)
            with open(dataset_track_inlier_analysis_fn, 'r') as fi:
                track_inlier_analysis = json.load(fi)
                track_degrees = track_inlier_analysis['track_degrees']
                track_degrees_baseline = track_inlier_analysis['track_degrees_baseline']
                track_degrees_classifier = track_inlier_analysis['track_degrees_classifier']
                track_degrees_gt = track_inlier_analysis['track_degrees_gt']

                track_degrees_baseline_max_score = track_inlier_analysis['track_degrees_baseline_max_score']
                track_degrees_baseline_min_score = track_inlier_analysis['track_degrees_baseline_min_score']
                # track_degrees_baseline_max_rmatches = track_inlier_analysis['track_degrees_baseline_max_rmatches']
                # track_degrees_baseline_min_rmatches = track_inlier_analysis['track_degrees_baseline_min_rmatches']
                track_degrees_classifier_max_score = track_inlier_analysis['track_degrees_classifier_max_score']
                track_degrees_classifier_min_score = track_inlier_analysis['track_degrees_classifier_min_score']

                for str_track_length in track_degrees:
                    if str_track_length not in inlier_track_degree_sum:
                        inlier_track_degree_sum[str_track_length] = 0
                        inlier_track_degree_count[str_track_length] = 0
                        inlier_track_degree_count_per_match[str_track_length] = {}
                        inlier_track_degree_count_per_match_baseline[str_track_length] = {}
                        inlier_track_degree_count_per_match_classifier[str_track_length] = {}
                        inlier_track_degree_count_per_match_gt[str_track_length] = {}
                        
                        inlier_track_degree_count_per_match_baseline_max_score[str_track_length] = {}
                        inlier_track_degree_count_per_match_baseline_min_score[str_track_length] = {}
                        # inlier_track_degree_count_per_match_baseline_max_rmatches[str_track_length] = {}
                        # inlier_track_degree_count_per_match_baseline_min_rmatches[str_track_length] = {}
                        inlier_track_degree_count_per_match_classifier_max_score[str_track_length] = {}
                        inlier_track_degree_count_per_match_classifier_min_score[str_track_length] = {}

                        inlier_tracks_baseline[str_track_length] = []
                        inlier_tracks_baseline_scores[str_track_length] = []
                        inlier_tracks_classifier_scores[str_track_length] = []

                        inlier_tracks_baseline_max_scores[str_track_length] = []
                        inlier_tracks_baseline_min_scores[str_track_length] = []
                        inlier_tracks_classifier_max_scores[str_track_length] = []
                        inlier_tracks_classifier_min_scores[str_track_length] = []

                        for jj in range(int(str_track_length) - 1, int(special.comb(int(str_track_length),2)) + 1):
                            inlier_track_degree_count_per_match[str_track_length][jj] = 0

                    if str_track_length not in outlier_track_degree_sum:
                        outlier_track_degree_sum[str_track_length] = 0
                        outlier_track_degree_count[str_track_length] = 0
                        outlier_track_degree_count_per_match[str_track_length] = {}
                        outlier_track_degree_count_per_match_baseline[str_track_length] = {}
                        outlier_track_degree_count_per_match_classifier[str_track_length] = {}
                        outlier_track_degree_count_per_match_gt[str_track_length] = {}

                        outlier_track_degree_count_per_match_baseline_max_score[str_track_length] = {}
                        outlier_track_degree_count_per_match_baseline_min_score[str_track_length] = {}
                        # outlier_track_degree_count_per_match_baseline_max_rmatches[str_track_length] = {}
                        # outlier_track_degree_count_per_match_baseline_min_rmatches[str_track_length] = {}
                        outlier_track_degree_count_per_match_classifier_max_score[str_track_length] = {}
                        outlier_track_degree_count_per_match_classifier_min_score[str_track_length] = {}

                        outlier_tracks_baseline[str_track_length] = []
                        outlier_tracks_baseline_scores[str_track_length] = []
                        outlier_tracks_classifier_scores[str_track_length] = []

                        outlier_tracks_baseline_max_scores[str_track_length] = []
                        outlier_tracks_baseline_min_scores[str_track_length] = []
                        outlier_tracks_classifier_max_scores[str_track_length] = []
                        outlier_tracks_classifier_min_scores[str_track_length] = []

                        for jj in range(int(str_track_length) - 1, int(special.comb(int(str_track_length),2)) + 1):
                            outlier_track_degree_count_per_match[str_track_length][jj] = 0

                    for t in track_degrees[str_track_length]:
                        if str_track_length in track_data['triangulated_tracks'] and t in track_data['triangulated_tracks'][str_track_length]:
                            inlier_track_degree_sum[str_track_length] += int(track_degrees[str_track_length][t])
                            inlier_track_degree_count[str_track_length] += 1
                            # print '#'*100
                            # print ('Inlier: {} - {}'.format(str_track_length, int(track_degrees[str_track_length][t])))
                            # print (track_degrees.keys())
                            # print (inlier_track_degree_count_per_match[str_track_length])
                            inlier_track_degree_count_per_match[str_track_length][int(np.round(track_degrees[str_track_length][t]))] = inlier_track_degree_count_per_match[str_track_length].get(int(np.round(track_degrees[str_track_length][t])), 0) + 1
                            inlier_track_degree_count_per_match_baseline[str_track_length][int(np.round(track_degrees_baseline[str_track_length][t]))] = inlier_track_degree_count_per_match_baseline[str_track_length].get(int(np.round(track_degrees_baseline[str_track_length][t])), 0) + 1
                            inlier_track_degree_count_per_match_classifier[str_track_length][int(np.round(track_degrees_classifier[str_track_length][t]))] = inlier_track_degree_count_per_match_classifier[str_track_length].get(int(np.round(track_degrees_classifier[str_track_length][t])), 0) + 1
                            inlier_track_degree_count_per_match_gt[str_track_length][int(np.round(track_degrees_gt[str_track_length][t]))] = inlier_track_degree_count_per_match_gt[str_track_length].get(int(np.round(track_degrees_gt[str_track_length][t])), 0) + 1

                            inlier_track_degree_count_per_match_baseline_max_score[str_track_length][np.round(track_degrees_baseline_max_score[str_track_length][t], 1)] = inlier_track_degree_count_per_match_baseline_max_score[str_track_length].get(np.round(track_degrees_baseline_max_score[str_track_length][t], 1), 0) + 1
                            inlier_track_degree_count_per_match_baseline_min_score[str_track_length][np.round(track_degrees_baseline_min_score[str_track_length][t], 1)] = inlier_track_degree_count_per_match_baseline_min_score[str_track_length].get(np.round(track_degrees_baseline_min_score[str_track_length][t], 1), 0) + 1
                            # inlier_track_degree_count_per_match_baseline_max_rmatches[str_track_length][np.round(track_degrees_baseline_max_rmatches[str_track_length][t], 1)] = inlier_track_degree_count_per_match_baseline_max_rmatches[str_track_length].get(np.round(track_degrees_baseline_max_rmatches[str_track_length][t], 1), 0) + 1
                            # inlier_track_degree_count_per_match_baseline_min_rmatches[str_track_length][np.round(track_degrees_baseline_min_rmatches[str_track_length][t], 1)] = inlier_track_degree_count_per_match_baseline_min_rmatches[str_track_length].get(np.round(track_degrees_baseline_min_rmatches[str_track_length][t], 1), 0) + 1
                            inlier_track_degree_count_per_match_classifier_max_score[str_track_length][np.round(track_degrees_classifier_max_score[str_track_length][t], 1)] = inlier_track_degree_count_per_match_classifier_max_score[str_track_length].get(np.round(track_degrees_classifier_max_score[str_track_length][t], 1), 0) + 1
                            inlier_track_degree_count_per_match_classifier_min_score[str_track_length][np.round(track_degrees_classifier_min_score[str_track_length][t], 1)] = inlier_track_degree_count_per_match_classifier_min_score[str_track_length].get(np.round(track_degrees_classifier_min_score[str_track_length][t], 1), 0) + 1

                            inlier_tracks_baseline[str_track_length].append(track_degrees[str_track_length][t])
                            inlier_tracks_baseline_scores[str_track_length].append(np.round(track_degrees_baseline[str_track_length][t], 2))
                            inlier_tracks_classifier_scores[str_track_length].append(np.round(track_degrees_classifier[str_track_length][t], 2))

                            inlier_tracks_baseline_max_scores[str_track_length].append(np.round(track_degrees_baseline_max_score[str_track_length][t], 2))
                            inlier_tracks_baseline_min_scores[str_track_length].append(np.round(track_degrees_baseline_min_score[str_track_length][t], 2))
                            inlier_tracks_classifier_max_scores[str_track_length].append(np.round(track_degrees_classifier_max_score[str_track_length][t], 2))
                            inlier_tracks_classifier_min_scores[str_track_length].append(np.round(track_degrees_classifier_min_score[str_track_length][t], 2))
                        else:
                            outlier_track_degree_sum[str_track_length] += int(track_degrees[str_track_length][t])
                            outlier_track_degree_count[str_track_length] += 1
                            # print '!'*100
                            # print ('Outlier: {} - {}'.format(str_track_length, int(track_degrees[str_track_length][t])))
                            # print (track_degrees.keys())
                            # print (outlier_track_degree_count_per_match[str_track_length])
                            outlier_track_degree_count_per_match[str_track_length][int(np.round(track_degrees[str_track_length][t]))] = outlier_track_degree_count_per_match[str_track_length].get(int(np.round(track_degrees[str_track_length][t])), 0) + 1
                            outlier_track_degree_count_per_match_baseline[str_track_length][int(np.round(track_degrees_baseline[str_track_length][t]))] = outlier_track_degree_count_per_match_baseline[str_track_length].get(int(np.round(track_degrees_baseline[str_track_length][t])), 0) + 1
                            outlier_track_degree_count_per_match_classifier[str_track_length][int(np.round(track_degrees_classifier[str_track_length][t]))] = outlier_track_degree_count_per_match_classifier[str_track_length].get(int(np.round(track_degrees_classifier[str_track_length][t])), 0) + 1
                            outlier_track_degree_count_per_match_gt[str_track_length][int(np.round(track_degrees_gt[str_track_length][t]))] = outlier_track_degree_count_per_match_gt[str_track_length].get(int(np.round(track_degrees_gt[str_track_length][t])), 0) + 1

                            outlier_track_degree_count_per_match_baseline_max_score[str_track_length][np.round(track_degrees_baseline_max_score[str_track_length][t], 1)] = outlier_track_degree_count_per_match_baseline_max_score[str_track_length].get(np.round(track_degrees_baseline_max_score[str_track_length][t], 1), 0) + 1
                            outlier_track_degree_count_per_match_baseline_min_score[str_track_length][np.round(track_degrees_baseline_min_score[str_track_length][t], 1)] = outlier_track_degree_count_per_match_baseline_min_score[str_track_length].get(np.round(track_degrees_baseline_min_score[str_track_length][t], 1), 0) + 1
                            # outlier_track_degree_count_per_match_baseline_max_rmatches[str_track_length][np.round(track_degrees_baseline_max_rmatches[str_track_length][t], 1)] = outlier_track_degree_count_per_match_baseline_max_rmatches[str_track_length].get(np.round(track_degrees_baseline_max_rmatches[str_track_length][t], 1), 0) + 1
                            # outlier_track_degree_count_per_match_baseline_min_rmatches[str_track_length][np.round(track_degrees_baseline_min_rmatches[str_track_length][t], 1)] = outlier_track_degree_count_per_match_baseline_min_rmatches[str_track_length].get(np.round(track_degrees_baseline_min_rmatches[str_track_length][t], 1), 0) + 1
                            outlier_track_degree_count_per_match_classifier_max_score[str_track_length][np.round(track_degrees_classifier_max_score[str_track_length][t], 1)] = outlier_track_degree_count_per_match_classifier_max_score[str_track_length].get(np.round(track_degrees_classifier_max_score[str_track_length][t], 1), 0) + 1
                            outlier_track_degree_count_per_match_classifier_min_score[str_track_length][np.round(track_degrees_classifier_min_score[str_track_length][t], 1)] = outlier_track_degree_count_per_match_classifier_min_score[str_track_length].get(np.round(track_degrees_classifier_min_score[str_track_length][t], 1), 0) + 1

                            outlier_tracks_baseline[str_track_length].append(track_degrees[str_track_length][t])
                            outlier_tracks_baseline_scores[str_track_length].append(np.round(track_degrees_baseline[str_track_length][t], 2))
                            outlier_tracks_classifier_scores[str_track_length].append(np.round(track_degrees_classifier[str_track_length][t], 2))

                            outlier_tracks_baseline_max_scores[str_track_length].append(np.round(track_degrees_baseline_max_score[str_track_length][t], 2))
                            outlier_tracks_baseline_min_scores[str_track_length].append(np.round(track_degrees_baseline_min_score[str_track_length][t], 2))
                            outlier_tracks_classifier_max_scores[str_track_length].append(np.round(track_degrees_classifier_max_score[str_track_length][t], 2))
                            outlier_tracks_classifier_min_scores[str_track_length].append(np.round(track_degrees_classifier_min_score[str_track_length][t], 2))
                            

    min_track_length = 2
    max_track_length = 9
    legend = []
    fig = plt.figure()
    plt.ylabel('P(success | matches)')
    plt.xlabel('# of matches (scores) in a track')

    relevant_scores = [
        {'outliers': outlier_track_degree_count_per_match, 'inliers': inlier_track_degree_count_per_match, 'label': 'baseline'},
        # {'outliers': outlier_track_degree_count_per_match_baseline, 'inliers': inlier_track_degree_count_per_match_baseline, 'label': 'baseline scores'},
        # {'outliers': outlier_track_degree_count_per_match_classifier, 'inliers': inlier_track_degree_count_per_match_classifier, 'label': 'classifier scores'},

        # {'outliers': outlier_track_degree_count_per_match_baseline_max_score, 'inliers': inlier_track_degree_count_per_match_baseline_max_score, 'label': 'B MAX'},
        # {'outliers': outlier_track_degree_count_per_match_baseline_min_score, 'inliers': inlier_track_degree_count_per_match_baseline_min_score, 'label': 'B MIN'},
        # {'outliers': outlier_track_degree_count_per_match_classifier_max_score, 'inliers': inlier_track_degree_count_per_match_classifier_max_score, 'label': 'C MAX'},
        # {'outliers': outlier_track_degree_count_per_match_classifier_min_score, 'inliers': inlier_track_degree_count_per_match_classifier_min_score, 'label': 'C MIN'},
    ]
    linestyles = [':', '-.', '--', '-']
    for outlier_counter, rs in enumerate(relevant_scores):
        # [ \
        # outlier_track_degree_count_per_match, \
        # outlier_track_degree_count_per_match_baseline, \
        # outlier_track_degree_count_per_match_classifier, \
        # outlier_track_degree_count_per_match_baseline_max_score, \
        # outlier_track_degree_count_per_match_baseline_max_score, \
        # outlier_track_degree_count_per_match_classifier_max_score, \
        # outlier_track_degree_count_per_match_classifier_min_score \
        # ]):

        for l in sorted(rs['outliers'].keys()):
            tl = int(l)
            # if tl >= 2 and tl <= 8:
            if tl >= min_track_length and tl <= max_track_length:
                p_successes = []
                p_successes_baseline = []
                p_successes_classifier = []
                p_successes_gt = []

                p_successes_baseline_max_score = []
                p_successes_baseline_min_score = []
                # p_successes_baseline_max_rmatches = []
                # p_successes_baseline_min_rmatches = []
                p_successes_classifier_max_score = []
                p_successes_classifier_min_score = []

                for v in sorted(rs['outliers'][l].keys()):
                    numerator = (1.0 * epsilon + 1.0 * rs['inliers'][l][v]) if v in rs['inliers'][l] else epsilon
                    denominator = (1000.0 * epsilon + 1.0 * rs['outliers'][l][v]) if v in rs['outliers'][l] else 1000*epsilon
                    p_successes.append(numerator / denominator)
                plt.plot(sorted(rs['outliers'][l].keys()), p_successes, linewidth=2.0, linestyle=linestyles[tl%len(linestyles)], color=colors[tl%len(colors)])
                legend.append('Track length ({}): {}'.format(rs['label'], l))

                # if outlier_counter == 0:
                #     for v in sorted(outlier_track_degree_count_per_match[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match[l][v] if v in inlier_track_degree_count_per_match[l] else epsilon
                #         p_successes.append((numerator)/ (outlier_track_degree_count_per_match[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match[l].keys()), p_successes, linewidth=4.0, linestyle=':', color=colors[outlier_counter%len(colors)])
                #     legend.append('Track length: {}'.format(l))
                # elif outlier_counter == 1:
                #     for v in sorted(outlier_track_degree_count_per_match_baseline[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_baseline[l][v] if v in inlier_track_degree_count_per_match_baseline[l] else epsilon
                #         p_successes_baseline.append((numerator)/ (outlier_track_degree_count_per_match_baseline[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_baseline[l].keys()), p_successes_baseline, linewidth=3.5, linestyle='--', color=colors[outlier_counter%len(colors)])
                #     legend.append('Track length (baseline scores): {}'.format(l))
                # elif outlier_counter == 2: 
                #     for v in sorted(outlier_track_degree_count_per_match_classifier[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_classifier[l][v] if v in inlier_track_degree_count_per_match_classifier[l] else epsilon
                #         p_successes_classifier.append((numerator)/ (outlier_track_degree_count_per_match_classifier[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_classifier[l].keys()), p_successes_classifier, linewidth=3.0, linestyle='-', color=colors[outlier_counter%len(colors)])
                #     legend.append('Track length (classifier scores): {}'.format(l))
                # elif outlier_counter == 3: 
                #     for v in sorted(outlier_track_degree_count_per_match_gt[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_gt[l][v] if v in inlier_track_degree_count_per_match_gt[l] else epsilon
                #         p_successes_gt.append((numerator)/ (outlier_track_degree_count_per_match_gt[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_gt[l].keys()), p_successes_gt, linewidth=2.75, linestyle='-.', color=colors[outlier_counter%len(colors)])
                #     legend.append('Track length (gt scores): {}'.format(l))
                # elif outlier_counter == 4:
                #     for v in sorted(outlier_track_degree_count_per_match_baseline_max_score[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_baseline_max_score[l][v] if v in inlier_track_degree_count_per_match_baseline_max_score[l] else epsilon
                #         p_successes_baseline_max_score.append((numerator)/ (outlier_track_degree_count_per_match_baseline_max_score[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_baseline_max_score[l].keys()), p_successes_baseline_max_score, linewidth=2.5, color=colors[outlier_counter%len(colors)])
                #     legend.append('Track length (baseline max scores): {}'.format(l))
                # elif outlier_counter == 5:
                #     for v in sorted(outlier_track_degree_count_per_match_baseline_min_score[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_baseline_min_score[l][v] if v in inlier_track_degree_count_per_match_baseline_min_score[l] else epsilon
                #         p_successes_baseline_min_score.append((numerator)/ (outlier_track_degree_count_per_match_baseline_min_score[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_baseline_min_score[l].keys()), p_successes_baseline_min_score, linewidth=2.0, color=colors[outlier_counter%len(colors)])
                #     legend.append('Track length (baseline min scores): {}'.format(l))
                # elif outlier_counter == 6:
                #     for v in sorted(outlier_track_degree_count_per_match_classifier_max_score[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_classifier_max_score[l][v] if v in inlier_track_degree_count_per_match_classifier_max_score[l] else epsilon
                #         p_successes_classifier_max_score.append((numerator)/ (outlier_track_degree_count_per_match_classifier_max_score[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_classifier_max_score[l].keys()), p_successes_classifier_max_score, linewidth=1.5)
                #     legend.append('Track length (classifier max scores): {}'.format(l))
                # elif outlier_counter == 7:
                #     for v in sorted(outlier_track_degree_count_per_match_classifier_min_score[l].keys()):
                #         numerator = 1.0 * inlier_track_degree_count_per_match_classifier_min_score[l][v] if v in inlier_track_degree_count_per_match_classifier_min_score[l] else epsilon
                #         p_successes_classifier_min_score.append((numerator)/ (outlier_track_degree_count_per_match_classifier_min_score[l][v] + numerator))
                #     plt.plot(sorted(outlier_track_degree_count_per_match_classifier_min_score[l].keys()), p_successes_classifier_min_score, linewidth=1.0)
                #     legend.append('Track length (classifier min scores): {}'.format(l))
                
    plt.legend(legend, loc='lower right',shadow=True, fontsize=10)
    plt.title('P(success| matches) per track length (robust match threshold: {} / robust triangulation: {})'.format(options['rmatches_threshold'], options['robust_triangulation']))
    # fig.set_size_inches(13.875, 7.875)
    fig.set_size_inches(30, 17)
    # if options['gt']:
    #     plt.savefig(os.path.join(data_folder, 'track-success-probabilities-gt_{}.png'.format(options['rmatches_threshold'])))
    # else:
    plt.savefig(os.path.join(data_folder, 'track-success-probabilities_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json.png'.format(options['gt'], options['rmatches_threshold'], options['robust_triangulation'])))




    # x_nbins, y_nbins = 100, 130
    
    for in_tracks, out_tracks, label in [ \
        [inlier_tracks_baseline, outlier_tracks_baseline, 'BASELINE-BINARY'], \
        [inlier_tracks_baseline_scores, outlier_tracks_baseline_scores, 'BASELINE'], \
        [inlier_tracks_classifier_scores, outlier_tracks_classifier_scores, 'CONVNET']]:

        fig = plt.figure()
        tls = np.array(sorted([int(l) for l in out_tracks.keys()]))
        # tls = tls[0:5]

        # max_matches = [int(special.comb(tl,2)) for tl in tls]
        max_possible_matches = [int(special.comb(tl,2)) for tl in tls]

        max_matches = 0.0
        for tl in tls:
            # if len(out_tracks[str(tl)]) > 0:
            #     max_matches = max(max_matches, max(out_tracks[str(tl)]))
            if len(in_tracks[str(tl)]) > 0:
                max_matches = max(max_matches, max(in_tracks[str(tl)]))
        
        max_matches = min(int(np.round(max_matches)), 250)
        x_nbins = int((max(tls) - min(tls)))
        y_nbins = max_matches

        inlier_xs = []
        inlier_ys = []
        outlier_xs = []
        outlier_ys = []

        for ii, tl in enumerate(tls):
            
            num_scores_inliers = len(in_tracks[str(tl)])
            num_scores_outliers = len(out_tracks[str(tl)])

            inlier_xs.extend([tl]*num_scores_inliers)
            # inlier_ys.extend([1.0*v/max_matches[ii] for v in in_tracks[str(tl)]])
            # inlier_ys.extend([1.0*v for v in in_tracks[str(tl)]])
            inlier_ys.extend(in_tracks[str(tl)])

            

            outlier_xs.extend([tl]*num_scores_outliers)
            # outlier_ys.extend([1.0*v/max_matches[ii] for v in out_tracks[str(tl)]])
            # outlier_ys.extend([1.0*v for v in out_tracks[str(tl)]])
            outlier_ys.extend(out_tracks[str(tl)])

            # import pdb; pdb.set_trace()

        

        plt.subplot(2, 2, 1)
        H_i, xedges_i, yedges_i, image_i = plt.hist2d(np.array(inlier_xs), np.array(inlier_ys), bins=[x_nbins,y_nbins], range=[[min(tls), max(tls)], [0, max_matches]], norm=LogNorm())
        X_i = np.concatenate((np.array(inlier_xs).reshape((-1,1)), np.array(inlier_ys).reshape((-1,1))), axis=1)
        
        kde_i_fn = os.path.join(data_folder, 'kde-inliers-{}-BW-{}.pkl'.format(label, options['bandwidth']))
        if os.path.isfile(kde_i_fn):
            kde_i = joblib.load(kde_i_fn)
        else:
            kde_i = KernelDensity(kernel='gaussian', bandwidth=options['bandwidth']).fit(X_i)
            joblib.dump(kde_i, kde_i_fn)

        # kde_i.score_samples(X_i)

        plt.subplot(2, 2, 2)
        H_o, xedges_o, yedges_o, image_o = plt.hist2d(np.array(outlier_xs), np.array(outlier_ys), bins=[x_nbins,y_nbins], range=[[min(tls), max(tls)], [0, max_matches]], norm=LogNorm())
        X_o = np.concatenate((np.array(outlier_xs).reshape((-1,1)), np.array(outlier_ys).reshape((-1,1))), axis=1)

        kde_o_fn = os.path.join(data_folder, 'kde-ouliers-{}-BW-{}.pkl'.format(label, options['bandwidth']))
        if os.path.isfile(kde_o_fn):
            kde_o = joblib.load(kde_o_fn)
        else:
            kde_o = KernelDensity(kernel='gaussian', bandwidth=options['bandwidth']).fit(X_o)
            joblib.dump(kde_o, kde_o_fn)
        # kde_o = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X_o)
        # kde_o.score_samples(X_o)

        plt.subplot(2, 2, 3)
        # plt.imshow((epsilon + H_i)/(H_o + H_i + 1000.0*epsilon), interpolation='nearest', origin='low', extent=[xedges_i[0], xedges_i[-1], yedges_i[0], yedges_i[-1]], aspect='auto')
        plt.imshow((epsilon + np.matrix(H_i).T)/(np.matrix(H_o).T + np.matrix(H_i).T + 1000.0*epsilon), interpolation='nearest', origin='low', extent=[xedges_o[0], xedges_o[-1], yedges_o[0], yedges_o[-1]], aspect='auto')

        plt.subplot(2, 2, 4)
        c = np.linspace(0, max(tls), max(tls) + 1)
        r = np.linspace(0, max_matches, max_matches + 1)
        X_ = list(itertools.product(c,r))
        # import pdb; pdb.set_trace()
        kde_img = np.zeros((max_matches+1, max(tls)+1)).astype(np.float)
        kde_i_scores = np.exp(kde_i.score_samples(X_))
        kde_o_scores = np.exp(kde_o.score_samples(X_))
        for ii, (track_num,match_score) in enumerate(X_):
            # print ('({}, {})'.format(track_num,match_score))
            # import pdb; pdb.set_trace()
            # if int(match_score) == 50 and int(track_num) == 40:
            #     import pdb; pdb.set_trace()
            kde_img[int(match_score),int(track_num)] = (kde_i_scores[ii] + epsilon)/(kde_i_scores[ii] + kde_o_scores[ii] + 1000.0*epsilon)
        plt.imshow(kde_img, origin='low', aspect='auto')

        plt.colorbar()

        if False:
            H_i, xedges_i, yedges_i = np.histogram2d(np.array(inlier_xs), np.array(inlier_ys), bins=[x_nbins,y_nbins], range=[[min(tls), max(tls)], [0, max_matches[-1]]])
            H_o, xedges_o, yedges_o = np.histogram2d(np.array(outlier_xs), np.array(outlier_ys), bins=[x_nbins,y_nbins], range=[[min(tls), max(tls)], [0, max_matches[-1]]])

            # import pdb; pdb.set_trace()
            
            plt.subplot(1, 3, 1)
            plt.imshow(H_i, interpolation='nearest', origin='low', extent=[xedges_i[0], xedges_i[-1], yedges_i[0], yedges_i[-1]], aspect='auto')
            # plt.imshow(H_i)
            plt.subplot(1, 3, 2)
            plt.imshow(H_o, interpolation='nearest', origin='low', extent=[xedges_o[0], xedges_o[-1], yedges_o[0], yedges_o[-1]], aspect='auto')
            # plt.imshow(H_o)
            plt.subplot(1, 3, 3)
            # plt.imshow((epsilon + H_i)/(H_o + H_i + epsilon), interpolation='nearest', origin='low', extent=[xedges_o[0], xedges_o[-1], yedges_o[0], yedges_o[-1]], aspect='auto')
            plt.imshow((epsilon + H_i)/(H_o + H_i + 1000.0*epsilon), interpolation='nearest', origin='low', extent=[xedges_i[0], xedges_i[-1], yedges_i[0], yedges_i[-1]], aspect='auto')

            plt.colorbar()

        # import pdb; pdb.set_trace()

        # ax = plt.gca()
        # im = NonUniformImage(ax, interpolation='bilinear')
        # xcenters = (xedges[:-1] + xedges[1:]) / 2
        # ycenters = (yedges[:-1] + yedges[1:]) / 2
        # im.set_data(xcenters, ycenters, H)
        # ax.images.append(im)
        # plt.show()
        plt.tight_layout()
        fig.set_size_inches(30, 17)
        plt.savefig(os.path.join(data_folder, 'track-success-probabilities-2dhistograms-{}-BW-{}.png'.format(label, options['bandwidth'])))

    # import sys; sys.exit(1)
    legend = []
    
    # colors = ['r','g','b','c','k','m','y']

    fig = plt.figure()
    plt.xlabel('sum of matches (or scores) in a track')
    plt.ylabel('p(inlier)')
    plt.title('p(inlier) vs match scores (robust match threshold: {} / robust triangulation: {})'.format(options['rmatches_threshold'], options['robust_triangulation']))

    relevant_scores_ = [
        {'outliers': outlier_tracks_baseline, 'inliers': inlier_tracks_baseline, 'label': 'baseline', 'max_range': -1, 'min_range': -1, 'nbins': -1},
        {'outliers': outlier_tracks_baseline_scores, 'inliers': inlier_tracks_baseline_scores, 'label': 'baseline scores', 'max_range': -1, 'min_range': 0, 'nbins': -1},
        # {'outliers': outlier_tracks_classifier_scores, 'inliers': inlier_tracks_classifier_scores, 'label': 'classifier scores', 'max_range': -1, 'min_range': 0, 'nbins': -1},
        
        # {'outliers': outlier_tracks_baseline_max_scores, 'inliers': inlier_tracks_baseline_max_scores, 'label': 'B MAX', 'max_range': 1, 'min_range': 0, 'nbins': 5},
        # {'outliers': outlier_tracks_baseline_min_scores, 'inliers': inlier_tracks_baseline_min_scores, 'label': 'B MIN', 'max_range': 1, 'min_range': 0, 'nbins': 5},
        # {'outliers': outlier_tracks_classifier_max_scores, 'inliers': inlier_tracks_classifier_max_scores, 'label': 'C MAX', 'max_range': 1, 'min_range': 0, 'nbins': 5},
        # {'outliers': outlier_tracks_classifier_min_scores, 'inliers': inlier_tracks_classifier_min_scores, 'label': 'C MIN', 'max_range': 1, 'min_range': 0, 'nbins': 5},
    ]


    for counter, rs in enumerate(relevant_scores_):
        print (min([int(tl_str) for tl_str in set(rs['outliers'].keys()).union(set(rs['inliers'].keys()))]))

        all_track_inliers_distribution = []
        all_track_outliers_distribution = []
        for ii, l in enumerate(sorted([int(v) for v in set(rs['outliers'].keys()).union(set(rs['inliers'].keys())) ] )):
            tl = int(l)
            if tl >= min_track_length and tl <= max_track_length:

                if rs['max_range'] == -1:
                    max_matches = int(special.comb(tl,2))
                else:
                    max_matches = rs['max_range']

                if rs['min_range'] == -1:
                    min_matches = tl - 1
                else:
                    min_matches = rs['min_range']
                
                if rs['nbins'] == -1:
                    nbins = max_matches - min_matches
                else:
                    nbins = rs['nbins']
                
                if nbins == 0:
                    nbins = 1

                # import pdb; pdb.set_trace()
                inliers_distribution = np.histogram(rs['inliers'][str(l)], bins=nbins, range=(0, max_matches))
                outliers_distribution = np.histogram(rs['outliers'][str(l)], bins=nbins, range=(0, max_matches))
                
                # inliers_distribution_classifier = np.histogram(inlier_tracks_classifier_scores[str(l)], bins=nbins, range=(0, max_matches))
                # outliers_distribution_classifier = np.histogram(outlier_tracks_classifier_scores[str(l)], bins=nbins, range=(0, max_matches))

                # inliers_distribution_baseline = np.histogram(inlier_tracks_baseline_scores[str(l)], bins=nbins, range=(0, max_matches))
                # outliers_distribution_baseline = np.histogram(outlier_tracks_baseline_scores[str(l)], bins=nbins, range=(0, max_matches))

                np.save(os.path.join(data_folder, 'track_inliers_distribution_{}_tl-{}-rmt-{}-rt-{}'.format(rs['label'].replace(' ', '-'), l, options['rmatches_threshold'], options['robust_triangulation'])), inliers_distribution)
                np.save(os.path.join(data_folder, 'track_outliers_distribution_{}_tl-{}-rmt-{}-rt-{}'.format(rs['label'].replace(' ', '-'), l, options['rmatches_threshold'], options['robust_triangulation'])), outliers_distribution)

                # np.save(os.path.join(data_folder, 'track_inliers_distribution_CONVNET_tl-{}-rmt-{}-rt-{}'.format(l, options['rmatches_threshold'], options['robust_triangulation'])), inliers_distribution_classifier)
                # np.save(os.path.join(data_folder, 'track_outliers_distribution_CONVNET_tl-{}-rmt-{}-rt-{}'.format(l, options['rmatches_threshold'], options['robust_triangulation'])), outliers_distribution_classifier)

                # np.save(os.path.join(data_folder, 'track_inliers_distribution_BASELINE_tl-{}-rmt-{}-rt-{}'.format(l, options['rmatches_threshold'], options['robust_triangulation'])), inliers_distribution_baseline)
                # np.save(os.path.join(data_folder, 'track_outliers_distribution_BASELINE_tl-{}-rmt-{}-rt-{}'.format(l, options['rmatches_threshold'], options['robust_triangulation'])), outliers_distribution_baseline)

                plt.plot( \
                    np.linspace(0,max_matches,nbins), \
                        (inliers_distribution[0].astype(np.float) + epsilon)/(inliers_distribution[0].astype(np.float) + outliers_distribution[0].astype(np.float) + 1000*epsilon), \
                        color=colors[ii%len(colors)], linewidth=2.0, linestyle=linestyles[counter%len(linestyles)]
                )
                all_track_inliers_distribution.append(inliers_distribution[0].astype(np.float))
                all_track_outliers_distribution.append(outliers_distribution[0].astype(np.float))
                # import pdb; pdb.set_trace()

                # plt.plot( \
                #     np.linspace(0,max_matches,nbins), \
                #         (inliers_distribution_classifier[0].astype(np.float) + epsilon)/(inliers_distribution_classifier[0].astype(np.float) + outliers_distribution_classifier[0].astype(np.float) + 1000*epsilon), \
                #         color=colors[ii%len(colors)], linewidth=1, linestyle='-'
                # )

                # plt.plot( \
                #     np.linspace(0,max_matches,nbins), \
                #         (inliers_distribution_baseline[0].astype(np.float) + epsilon)/(inliers_distribution_baseline[0].astype(np.float) + outliers_distribution_baseline[0].astype(np.float) + 1000*epsilon), \
                #         color=colors[ii%len(colors)], linewidth=2, linestyle=':'
                # )

                legend.append('Track length ({}): {}'.format(rs['label'], l))
                # legend.append('Track length (classifier scores): {}'.format(l))
                # legend.append('Track length (baseline scores): {}'.format(l))


    plt.legend(legend, loc='lower right',shadow=True, fontsize=10)
    fig.set_size_inches(30, 17)
    plt.savefig(os.path.join(data_folder, 'track-success-probabilities-histograms_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json.png'.format(options['gt'], options['rmatches_threshold'], options['robust_triangulation'])))

    plt.figure()
    plt.title('Track inlier distribution at a track level')
    plt.xlabel('Track length')
    plt.ylabel('Inlier %')
    # print ('{}  -  {}'.format(max_track_length - min_track_length + 1, len(all_track_inliers_distribution)))
    # import pdb; pdb.set_trace()
    inlier_means = np.array([np.mean(np.array(i)) for i in all_track_inliers_distribution])
    outlier_means = np.array([np.mean(np.array(o)) for o in all_track_outliers_distribution])
    plt.plot(np.linspace(min_track_length, max_track_length, max_track_length - min_track_length + 1), (inlier_means + epsilon)/(inlier_means + outlier_means + 1000*epsilon))
    fig.set_size_inches(15, 8.5)
    plt.savefig(os.path.join(data_folder, 'all-track-inlier-distribution_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json.png'.format(options['gt'], options['rmatches_threshold'], options['robust_triangulation'])))
    # legend = []
    # fig = plt.figure()
    # plt.ylabel('P(success | matches)')
    # plt.xlabel('# of rmatches in a track')
    # for l in sorted(outlier_track_degree_count_per_match.keys()):
    #     tl = int(l)
    #     if tl >= min_track_length and tl <= max_track_length:
    #         p_successes_baseline_max_rmatches = []
    #         p_successes_baseline_min_rmatches = []

    #         for v in sorted(outlier_track_degree_count_per_match_baseline_max_rmatches[l].keys()):
    #             numerator = 1.0 * inlier_track_degree_count_per_match_baseline_max_rmatches[l][v] if v in inlier_track_degree_count_per_match_baseline_max_rmatches[l] else epsilon
    #             p_successes_baseline_max_rmatches.append((numerator)/ (outlier_track_degree_count_per_match_baseline_max_rmatches[l][v] + numerator))
    #         for v in sorted(outlier_track_degree_count_per_match_baseline_min_rmatches[l].keys()):
    #             numerator = 1.0 * inlier_track_degree_count_per_match_baseline_min_rmatches[l][v] if v in inlier_track_degree_count_per_match_baseline_min_rmatches[l] else epsilon
    #             p_successes_baseline_min_rmatches.append((numerator)/ (outlier_track_degree_count_per_match_baseline_min_rmatches[l][v] + numerator))


    #         plt.plot(sorted(outlier_track_degree_count_per_match_baseline_max_rmatches[l].keys()), p_successes_baseline_max_rmatches)
    #         plt.plot(sorted(outlier_track_degree_count_per_match_baseline_min_rmatches[l].keys()), p_successes_baseline_min_rmatches)
    #         legend.append('Track length (baseline max rmatches): {}'.format(l))
    #         legend.append('Track length (baseline min rmatches): {}'.format(l))
    # plt.legend(legend, loc='lower right',shadow=True, fontsize=10)
    # plt.title('P(success| matches) per track length')
    # fig.set_size_inches(30, 17)
    # plt.savefig(os.path.join(data_folder, 'track-success-probabilities-rmatches.png'))



def triangulate_gt_reconstruction(data, graph, recon_fn, options):
    recon = data.load_reconstruction('reconstruction_gt.json')[0]
    if options['robust_triangulation']:
        _, robust_graph = reconstruction.robustly_retriangulate(graph, recon, data.config)
    else:
        reconstruction.retriangulate(graph, recon, data.config)
    reconstruction.paint_reconstruction(data, graph, recon)
    data.save_reconstruction([recon], filename=recon_fn)
    return recon

def calculate_triangulated_tracks(arg):
    t, dataset_track_raw_data_fn, options = arg

    # gt = options['gt']
    total_track_count = {}
    triangulated_track_count = {}
    total_tracks = {}
    triangulated_tracks = {}

    data = dataset.DataSet(t)
    if options['gt']:
        graph = data.load_tracks_graph('tracks-gt-matches.csv')
        # recon_fn = 'reconstruction_gt_triangulated_gt_robust-triangulation-{}.json'.format(options['robust_triangulation'])
    else:
        graph = data.load_tracks_graph('tracks.csv')
    recon_fn = 'reconstruction_gt_triangulated_gt-{}_robust-triangulation-{}.json'.format(options['gt'], options['robust_triangulation'])


    if data.reconstruction_exists(recon_fn):
        recon_gt_triangulated = data.load_reconstruction(recon_fn)[0]
    else:
        recon_gt_triangulated = triangulate_gt_reconstruction(data, graph, recon_fn, options)
       
    tracks, images = matching.tracks_and_images(graph)
    for t in tracks:
        track_length = len(graph[t].keys())
        total_track_count[track_length] = total_track_count.get(track_length, 0) + 1
        if track_length not in total_tracks:
            total_tracks[track_length] = []
        total_tracks[track_length].append(t)

        if t in recon_gt_triangulated.points.keys():
            triangulated_track_count[track_length] = triangulated_track_count.get(track_length, 0) + 1
            if track_length not in triangulated_tracks:
                triangulated_tracks[track_length] = []
            triangulated_tracks[track_length].append(t)

    with open(dataset_track_raw_data_fn, 'w') as fout:
        json.dump({'total_track_count': total_track_count, 'triangulated_track_count': triangulated_track_count, 'total_tracks': total_tracks, 'triangulated_tracks': triangulated_tracks}, fout, sort_keys=True, indent=4, separators=(',', ': '))

def track_length_analysis(datasets, options={}):
    data_folder = 'data/track_analysis'
    mkdir_p(data_folder)
    processes = options['processes']
    args = []

    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        # if options['gt']:
        dataset_track_raw_data_fn = os.path.join(data_folder, 'track_raw_data_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))
        # else:
            # dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_gt-{}_{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['rmatches_threshold'], options['robust_triangulation']))

        if os.path.isfile(dataset_track_raw_data_fn):
            continue
        args.append([t, dataset_track_raw_data_fn, options])

    p = Pool(processes)
    if processes == 1:
        for arg in args:
            calculate_triangulated_tracks(arg)
    else:
        p.map(calculate_triangulated_tracks, args)
        p.close()

    print ('Finished calculating triangulated tracks')

    total_track_count = {}
    triangulated_track_count = {}
    for i,t in enumerate(datasets):
        print ('Aggregating results - dataset: {}'.format(os.path.basename(t)))
        # if options['gt']:
        #     dataset_track_raw_data_fn = os.path.join(data_folder, 'track_analysis_gt_{}_{}.json'.format(os.path.basename(t), options['rmatches_threshold']))
        # else:
        #     dataset_track_raw_data_fn = os.path.join(data_folder, 'track_analysis_{}_{}.json'.format(os.path.basename(t), options['rmatches_threshold']))
        dataset_track_raw_data_fn = os.path.join(data_folder, 'track_raw_data_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))

        with open(dataset_track_raw_data_fn, 'r') as fin:
            datum = json.load(fin)
            for k in datum['total_track_count'].keys():
                total_track_count[int(k)] = total_track_count.get(int(k),0) + int(datum['total_track_count'].get(k,0))
                triangulated_track_count[int(k)] = triangulated_track_count.get(int(k),0) + int(datum['triangulated_track_count'].get(k,0))

    track_lengths = []
    track_inlier_percentages = []
    track_inlier_counts = []
    for k in sorted(total_track_count.keys()):
        track_inlier_percentage = round(100.0 * triangulated_track_count.get(k,0) / total_track_count.get(k,0), 2)

        track_lengths.append(k)
        track_inlier_counts.append(triangulated_track_count.get(k,0))
        track_inlier_percentages.append(track_inlier_percentage)
        # print ('\t{}:  {} / {} = {}'.format(k, triangulated_track_count.get(k,0), total_track_count.get(k,0), track_inlier_percentage))

    # Plotting track inlier graph
    fig = plt.figure()
    ax1=fig.add_subplot(111, label="1")
    

    ax1.set_xlim(2,max(track_lengths))
    ax1.set_ylim(0,100.0)
    ax1.plot(np.array(track_lengths), np.array(track_inlier_percentages))
    ax1.set_xlabel('Track length')
    ax1.set_ylabel('Inlier % (Percentage reconstructed)', color="b")
    ax1.tick_params(axis='y', colors="b")
    ax1.yaxis.tick_left()
    plt.legend(['% track inliers'], loc='lower left', shadow=True, fontsize=10)

    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax2.set_xlim(2,max(track_lengths))
    ax2.set_ylim(0,max(track_inlier_counts))
    ax2.plot(np.array(track_lengths), np.array(track_inlier_counts), color='g')
    ax2.plot(np.array(track_lengths), np.array([total_track_count[k] for k in total_track_count]), color='r')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', colors="g")
    ax2.set_xlabel('Track length')
    ax2.set_ylabel('Inlier Count (Percentage reconstructed)', color="g")
    
    plt.title('Track inliers vs track length')
    plt.legend(['Inlier track count', 'Total track count'], loc='lower right', shadow=True, fontsize=10)

    fig.set_size_inches(13.875, 7.875)
    # if options['gt']:
    plt.savefig(os.path.join(data_folder, 'track-length-inliers-gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json.png'.format(options['gt'], options['rmatches_threshold'], options['robust_triangulation'])))

    fig = plt.figure()
    ax2=fig.add_subplot(111, label="1")
    ax2.set_xlim(10,max(track_lengths))
    ax2.set_ylim(0,500)
    ax2.plot(np.array(track_lengths), np.array(track_inlier_counts), color='g')
    ax2.plot(np.array(track_lengths), np.array([total_track_count[k] for k in total_track_count]), color='r')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', colors="g")
    ax2.set_xlabel('Track length')
    ax2.set_ylabel('Inlier Count (Percentage reconstructed)', color="g")
    fig.set_size_inches(13.875, 7.875)
    plt.title('Track inliers vs track length')
    plt.legend(['Inlier track count', 'Total track count'], loc='lower left', shadow=True, fontsize=10)

    plt.savefig(os.path.join(data_folder, 'track-length-inliers-gt-{}_rmatches-threshold-{}_robust-triangulation-{}-zoomed.json.png'.format(options['gt'], options['rmatches_threshold'], options['robust_triangulation'])))
    # else:
    #     plt.savefig(os.path.join(data_folder, 'track-length-inliers_{}.png'.format(options['rmatches_threshold'])))


def calculate_marginal_distributions(datasets, options):
    data_folder = 'data/track_analysis'
    mkdir_p(data_folder)
    epsilon = 0.0000000000001

    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        data = dataset.DataSet(t)

        # if options['gt']:
        #     dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_gt_{}_{}.json'.format(os.path.basename(t), options['rmatches_threshold']))
        #     dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_gt_{}_{}.json'.format(os.path.basename(t), options['rmatches_threshold']))
        # else:
        #     dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_{}_{}.json'.format(os.path.basename(t), options['rmatches_threshold'])) # used for reading only
        #     dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_{}_{}.json'.format(os.path.basename(t), options['rmatches_threshold']))
        dataset_track_raw_data_fn = os.path.join(data_folder, 'track_raw_data_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))
        dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_{}_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json'.format(os.path.basename(t), options['gt'], options['rmatches_threshold'], options['robust_triangulation']))

        if options['gt']:
            graph = data.load_tracks_graph('tracks-gt-matches.csv')
        else:
            graph = data.load_tracks_graph('tracks.csv')
        image_matching_results_baseline = data.load_image_matching_results(robust_matches_threshold=15, classifier='BASELINE')
        image_matching_results_classifier = data.load_image_matching_results(robust_matches_threshold=15, classifier='CONVNET')
        image_matching_results_gt = data.load_groundtruth_image_matching_results(robust_matches_threshold=15)

        with open(dataset_track_raw_data_fn, 'r') as fi:
            track_data = json.load(fi)
            triangulated_tracks = track_data['triangulated_tracks']
            total_tracks = track_data['total_tracks']

            with open(dataset_track_inlier_analysis_fn, 'r') as fi:
                track_inlier_analysis = json.load(fi)
            #     track_degrees = track_inlier_analysis['track_degrees']
                track_degrees_baseline = track_inlier_analysis['track_degrees_baseline']
                track_degrees_classifier = track_inlier_analysis['track_degrees_classifier']
            #     track_degrees_gt = track_inlier_analysis['track_degrees_gt']

                # inliers = {}
                # outliers = {}
                inliers_tl = []
                outliers_tl = []
                inliers_s = []
                outliers_s = []
                for _, str_track_length in enumerate(sorted(track_data['total_tracks'])):
                    tl = int(str_track_length)

                    for ii, t in enumerate(total_tracks[str_track_length]):
                        # if str_track_length not in track_data['total_tracks']:

                        # if tl not in inliers_tl:
                        #     inliers[tl] = epsilon
                        # if tl not in outliers_tl:
                        #     outliers[tl] = 1000.0*epsilon

                        if str_track_length in triangulated_tracks and t in triangulated_tracks[str_track_length]:
                            # inliers[tl] += 1
                            inliers_tl.append(tl)
                            if str_track_length in track_degrees_baseline and t in track_degrees_baseline[str_track_length]:
                                inliers_s.append(track_degrees_baseline[str_track_length][t])
                        elif str_track_length in total_tracks and t in total_tracks[str_track_length]:
                            # outliers[tl] += 1
                            outliers_tl.append(tl)

                            if str_track_length in track_degrees_baseline and t in track_degrees_baseline[str_track_length]:
                                outliers_s.append(track_degrees_baseline[str_track_length][t])

                # for k in range(0, max(max(outliers.keys()), max(inliers.keys()))):
                #     if k in inliers:
                #         inliers_tl.append(inliers[k])
                #     else:
                #         inliers_tl.append(0)
                #     if k in outliers:
                #         outliers_tl.append(outliers[k])
                #     else:
                #         outliers_tl.append(0)

                max_track_length = max([int(str_track_length) for str_track_length in total_tracks.keys()])
                # max_scores = int(special.comb(max_track_length,2)) + 1
                max_scores = int(max(max(inliers_s), max(outliers_s)))

                fig = plt.figure()
                plt.title("Histograms of inliers and outliers for: track lengths and scores")
                
                plt.subplot(2, 2, 1)
                plt.ylabel('Inlier Count')
                plt.xlabel('Track Length')
                i_histogram_tl = plt.hist(inliers_tl, bins=max_track_length, range=(0, max_track_length))

                plt.subplot(2, 2, 2)
                plt.ylabel('Outlier Count')
                plt.xlabel('Track Length')
                o_histogram_tl = plt.hist(outliers_tl, bins=max_track_length, range=(0, max_track_length))


                plt.subplot(2, 2, 3)
                plt.ylabel('Inlier Count')
                plt.xlabel('Scores')
                i_histogram_s = plt.hist(inliers_s, bins=max_scores, range=(0, max_scores))

                plt.subplot(2, 2, 4)
                plt.ylabel('Outlier Count')
                plt.xlabel('Scores')
                o_histogram_s = plt.hist(outliers_s, bins=max_scores, range=(0, max_scores))


                np.save(os.path.join(data_folder, 'inlier-marginal-track-length'), i_histogram_tl)
                np.save(os.path.join(data_folder, 'outlier-marginal-track-length'), o_histogram_tl)

                np.save(os.path.join(data_folder, 'inlier-marginal-scores'), i_histogram_s)
                np.save(os.path.join(data_folder, 'outlier-marginal-scores'), o_histogram_s)

                plt.tight_layout()
                fig.set_size_inches(30, 17)
                plt.savefig(os.path.join(data_folder, 'p-marginals-track-length-and-scores_gt-{}_rmatches-threshold-{}_robust-triangulation-{}.json.png'.format(options['gt'], options['rmatches_threshold'], options['robust_triangulation'])))


def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)

    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global dataset, matching, reconstruction

    options = {
        'processes': 12,
        'gt': False,
        'rmatches_threshold': 15,
        'robust_triangulation': True,
        'bandwidth': 0.2,
        'classifier': 'BASELINE',
        'lowes_ratio_threshold': 0.8
    }

    datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Barn',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Caterpillar',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Church',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courthouse',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ignatius',
    
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Meetingroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Truck',

        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/facade',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/kicker',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/meadow',

        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lecture_room',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/living_room',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_360',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_room',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_hemisphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_coke',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk_with_person',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_dishes',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_no_loop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_with_loop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_360',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam3',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz',
    ]

    val_datasets = [
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lecture_room',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/living_room',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Meetingroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Truck',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz'
    ]

    # for i in range(0,2):
    #     if i == 0:
    #         options['gt'] = True
    #     else:
    #         options['gt'] = False

    #     track_length_analysis(val_datasets, options)
    #     track_inlier_analysis(val_datasets, options) # track length analysis needs to run before this to create a list of inliers
    # for rm in [15, 50]:
    #     options['rmatches_threshold'] = rm
    
    # for i,bw in enumerate([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0]):
    #     options['bandwidth'] = bw

    # for v in [True, False]:
    #     options['robust_triangulation'] = v
    #     track_length_analysis(val_datasets, options)
    #     # calculate_marginal_distributions(val_datasets, options)
    #     track_inlier_analysis(val_datasets, options) # track length analysis needs to run before this to create a list of inliers

    # track_length_analysis(val_datasets, options)
    track_inlier_analysis(val_datasets, options)

if __name__ == '__main__':
    main(sys.argv)
