import json
import math
import numpy as np
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import rcParams
import seaborn as sns; sns.set()
import scipy
import sys
import torch

from argparse import ArgumentParser
import matching_classifiers # import load_classifier, calculate_per_image_mean_auc, calculate_dataset_auc, mkdir_p
import convnet

def save_analysis_data(data, fn):
    with open(fn, 'w') as fout:
        json.dump(data.tolist(), fout, sort_keys=True, indent=4, separators=(',', ': '))

def load_analysis_data(fn):
    with open(fn, 'r') as fin:
        data = json.load(fin)
    return np.array(data)

def plot_iccv_figures(training_datasets, testing_datasets, options={}):
    data_folder = 'data/image-matching-classifiers-analysis'
    matching_classifiers.mkdir_p(data_folder)
    if False:
        nbins = 500
        epsilon = 0.00000001
        labels = None

        for i,t in enumerate(training_datasets):
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
                _mds_rank_percentage_im1_im2, _mds_rank_percentage_im2_im1, \
                _distance_rank_percentage_im1_im2_gt, _distance_rank_percentage_im2_im1_gt, \
                _shortest_path_length, \
                _num_gt_inliers, _labels] \
                = data.load_image_matching_dataset(robust_matches_threshold=20, rmatches_min_threshold=options['image_match_classifier_min_match'], \
                    rmatches_max_threshold=options['image_match_classifier_max_match'], spl=100000)
            if i == 0:
                num_rmatches = _num_rmatches.copy()
                labels = _labels.copy()
            else:
                num_rmatches = np.concatenate((num_rmatches, _num_rmatches), axis=0)
                labels = np.concatenate((labels, _labels), axis=0)

        inliers_distribution = np.histogram(num_rmatches[np.where(labels >=1.0)[0]], bins=nbins, range=(8,2008))
        outliers_distribution = np.histogram(num_rmatches[np.where(labels < 1.0)[0]], bins=nbins, range=(8,2008))

        f1 = plt.figure(1)
        plt.ylim(0,1.1)
        plt.xlim(8,100)
        f1.patch.set_facecolor('white')
        plt.plot( \
            np.linspace(8,2008,nbins), \
            (inliers_distribution[0].astype(np.float) + epsilon)/(inliers_distribution[0].astype(np.float) + outliers_distribution[0].astype(np.float) + epsilon), \
            'k-', linewidth=3 
        )
        # plt.axvspan(15, 50, ymin=0.0, ymax=1.0, alpha=0.25, color='red')
        inlier_percentage = (inliers_distribution[0].astype(np.float) + epsilon)/(inliers_distribution[0].astype(np.float) + outliers_distribution[0].astype(np.float) + epsilon)
        bins = inliers_distribution[1]
        min_idx = np.where(bins >= 15)[0][0]
        max_idx = np.where(bins > 50)[0][0]
        plt.fill_between(bins[min_idx:max_idx], \
            y1=0, y2=inlier_percentage[min_idx:max_idx], color='b', alpha=0.25)

        plt.title('Inlier distribution', fontsize=16)
        plt.xlabel('# of robust matches', fontsize=13)
        plt.ylabel('% inliers', fontsize=13)
        plt.tight_layout()
        f1.set_size_inches(10, 10)
        plt.savefig(os.path.join(data_folder, 'inlier-distribution-iccv.png'))
        plt.clf()


    num_rmatches = None
    labels = None
    for i,t in enumerate(testing_datasets):
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
            = data.load_image_matching_dataset(robust_matches_threshold=20, rmatches_min_threshold=15, \
                rmatches_max_threshold=50, spl=100000)
        if i == 0:
            num_rmatches = _num_rmatches.copy()
            labels = _labels.copy()
        else:
            num_rmatches = np.concatenate((num_rmatches, _num_rmatches), axis=0)
            labels = np.concatenate((labels, _labels), axis=0)


    f2 = plt.figure(2)
    f2.patch.set_facecolor('white')
    plt.title('Precision-Recall Curves', fontsize=16)
    plt.ylabel('Precision', fontsize=13)
    plt.xlabel('Recall', fontsize=13)
    plt.ylim(0,1.0)
    plt.xlim(0,1.0)
    f2.patch.set_facecolor('white')
    auc_entire_ds_rm, _ = matching_classifiers.calculate_dataset_auc(num_rmatches, labels, color='green', ls='solid')
    auc_entire_ds_cl = auc_entire_ds_rm
    
    plt.tight_layout()
    f2.set_size_inches(10, 10)
    plt.legend(
        [
        'Baseline AUC=' + str(round(auc_entire_ds_rm, 3)),
        'Siamese Network AUC=' + str(round(auc_entire_ds_cl, 3))
        ],
        loc='lower right',
        shadow=True,
        fontsize=10
    )
    plt.savefig(os.path.join(data_folder, 'precision-recall-curves-iccv.png'))

    plt.show()
    

def classification_and_reconstruction_correlation_analysis(datasets, option={}):
    for i,t in enumerate(datasets):
        print ('#'*100)
        print ('Running analysis for dataset: {}'.format(t))
        data = dataset.DataSet(t)
        ate_results = data.load_ate_results()
        rpe_results = data.load_rpe_results()
        image_matching_results = data.load_image_matching_results()



def classify_images(datasets, options={}):
    data_folder = 'data/image-matching-classifiers-analysis'
    matching_classifiers.mkdir_p(data_folder)
    
    print ('-'*100)
    print ('Classifier: {}'.format(options['classifier']))
    shortest_path_length = 1000000
    if options['classifier'] == 'BDT':
        trained_classifier = matching_classifiers.load_classifier(options['image_match_classifier_file'])
    elif options['classifier'] == 'CONVNET':
        # load convnet checkpoint here
        print ('Loading ConvNet checkpoint: {}'.format(options['convnet_checkpoint']))
        checkpoint = torch.load(options['convnet_checkpoint'])
        kwargs = {}
        options['model'] = 'resnet18'
        # options['convnet_lr'] = 0.01
        options['convnet_use_images'] = False
        options['convnet_use_warped_images'] = False
        options['convnet_use_feature_match_map'] = True
        options['convnet_use_track_map'] = False
        options['convnet_use_non_rmatches_map'] = True
        options['convnet_use_rmatches_map'] = True
        options['convnet_use_matches_map'] = True
        options['convnet_use_photometric_error_maps'] = True

        options['features'] = 'RM'
        options['mlp-layer-size'] = 256
        options['use_small_weights'] = False
        options['num_workers'] = 10
        options['batch_size'] = 64
        options['shuffle'] = False
        options['convnet_input_size'] = 224
        options['triplet-sampling-strategy'] = 'normal'
        options['log_interval'] = 1
        options['experiment'] = 'RM+TM+SE+PE'
        options['convnet_load_dataset_in_memory'] = False

        trained_classifier = convnet.ConvNet(options, **kwargs)
        trained_classifier.cuda()
        trained_classifier.load_state_dict(checkpoint['state_dict'])
    print ('-'*100)
    for i,t in enumerate(datasets):
        print ('#'*100)
        print ('Running classifier for dataset: {}'.format(t))
        data = dataset.DataSet(t)
        results = {}
        # image_matching_classifier_thresholds = data.config.get('image_matching_classifier_thresholds')
        if options['classifier'] == 'CONVNET':
            shortest_paths = data.load_shortest_paths(label='rm-cost')
            dsets_te_ = []
            fns_te_ = []
            num_rmatches_te_ = []
            shortest_paths_te_ = []
            
            for im1 in data.images():
                im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
                for im2 in im1_all_robust_matches:
                    rmatches = im1_all_robust_matches[im2]
                    if len(rmatches) == 0:
                        continue

                    if len(rmatches) < options['image_match_classifier_min_match'] or \
                        len(rmatches) > options['image_match_classifier_max_match']: # or \
                        # len(shortest_paths[im1][im2]["shortest_path"]) > shortest_path_length:

                        if len(rmatches) < options['image_match_classifier_min_match']: # or \
                            # len(shortest_paths[im1][im2]["shortest_path"]) > shortest_path_length:
                            score = 0.0
                        elif len(rmatches) > options['image_match_classifier_min_match']:
                            score = 1.0

                        if im1 not in results:
                            results[im1] = {}
                        if im2 not in results:
                            results[im2] = {}
                        results[im1][im2] = {'im1': im1, 'im2': im2, 'score': score, 'num_rmatches': len(rmatches), 'shortest_path_length': len(shortest_paths[im1][im2]["shortest_path"])}
                        results[im2][im1] = {'im1': im2, 'im2': im1, 'score': score, 'num_rmatches': len(rmatches), 'shortest_path_length': len(shortest_paths[im1][im2]["shortest_path"])}
                    else:
                        fns_te_.append([im1,im2])
                        dsets_te_.append(t)
                        num_rmatches_te_.append(len(im1_all_robust_matches[im2]))
                        shortest_paths_te_.append(len(shortest_paths[im1][im2]["shortest_path"]))
            dsets_te = np.array(dsets_te_)
            fns_te = np.array(fns_te_)
            R11s_te = np.ones((len(dsets_te),))
            R12s_te = np.ones((len(dsets_te),))
            R13s_te = np.ones((len(dsets_te),))
            R21s_te = np.ones((len(dsets_te),))
            R22s_te = np.ones((len(dsets_te),))
            R23s_te = np.ones((len(dsets_te),))
            R31s_te = np.ones((len(dsets_te),))
            R32s_te = np.ones((len(dsets_te),))
            R33s_te = np.ones((len(dsets_te),))
            num_rmatches_te = np.array(num_rmatches_te_)
            num_matches_te = np.ones((len(dsets_te),))
            spatial_entropy_1_8x8_te = np.ones((len(dsets_te),))
            spatial_entropy_2_8x8_te = np.ones((len(dsets_te),))
            spatial_entropy_1_16x16_te = np.ones((len(dsets_te),))
            spatial_entropy_2_16x16_te = np.ones((len(dsets_te),))
            pe_histogram_te = np.ones((len(dsets_te),))
            pe_polygon_area_percentage_te = np.ones((len(dsets_te),))
            nbvs_im1_te = np.ones((len(dsets_te),))
            nbvs_im2_te = np.ones((len(dsets_te),))
            te_histogram_te = np.ones((len(dsets_te),))
            ch_im1_te = np.ones((len(dsets_te),))
            ch_im2_te = np.ones((len(dsets_te),))
            vt_rank_percentage_im1_im2_te = np.ones((len(dsets_te),))
            vt_rank_percentage_im2_im1_te = np.ones((len(dsets_te),))
            sq_rank_scores_mean_te = np.ones((len(dsets_te),))
            sq_rank_scores_min_te = np.ones((len(dsets_te),))
            sq_rank_scores_max_te = np.ones((len(dsets_te),))
            sq_distance_scores_te = np.ones((len(dsets_te),))
            lcc_im1_15_te = np.ones((len(dsets_te),))
            lcc_im2_15_te = np.ones((len(dsets_te),))
            min_lcc_15_te = np.ones((len(dsets_te),))
            max_lcc_15_te = np.ones((len(dsets_te),))
            lcc_im1_20_te = np.ones((len(dsets_te),))
            lcc_im2_20_te = np.ones((len(dsets_te),))
            min_lcc_20_te = np.ones((len(dsets_te),))
            max_lcc_20_te = np.ones((len(dsets_te),))
            lcc_im1_25_te = np.ones((len(dsets_te),))
            lcc_im2_25_te = np.ones((len(dsets_te),))
            min_lcc_25_te = np.ones((len(dsets_te),))
            max_lcc_25_te = np.ones((len(dsets_te),))
            lcc_im1_30_te = np.ones((len(dsets_te),))
            lcc_im2_30_te = np.ones((len(dsets_te),))
            min_lcc_30_te = np.ones((len(dsets_te),))
            max_lcc_30_te = np.ones((len(dsets_te),))
            lcc_im1_35_te = np.ones((len(dsets_te),))
            lcc_im2_35_te = np.ones((len(dsets_te),))
            min_lcc_35_te = np.ones((len(dsets_te),))
            max_lcc_35_te = np.ones((len(dsets_te),))
            lcc_im1_40_te = np.ones((len(dsets_te),))
            lcc_im2_40_te = np.ones((len(dsets_te),))
            min_lcc_40_te = np.ones((len(dsets_te),))
            max_lcc_40_te = np.ones((len(dsets_te),))
            shortest_path_length_te = np.array(shortest_paths_te_)
            mds_rank_percentage_im1_im2_te = np.ones((len(dsets_te),))
            mds_rank_percentage_im2_im1_te = np.ones((len(dsets_te),))
            distance_rank_percentage_im1_im2_gt_te = np.ones((len(dsets_te),))
            distance_rank_percentage_im2_im1_gt_te = np.ones((len(dsets_te),))
            labels_te = np.ones((len(dsets_te),))
        else:
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
                = data.load_image_matching_dataset(robust_matches_threshold=20, rmatches_min_threshold=15, \
                    rmatches_max_threshold=50, spl=shortest_path_length)

            if len(_labels) == 0:
                continue
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
            labels_te[labels_te < 0] = 0

        
        arg = [ \
            dsets_te, fns_te, R11s_te, R12s_te, R13s_te, R21s_te, R22s_te, R23s_te, R31s_te, R32s_te, R33s_te, num_rmatches_te, num_matches_te, spatial_entropy_1_8x8_te, \
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
            labels_te, np.ones((len(labels_te))), \
            False, trained_classifier, options
        ]

        if options['classifier'] == 'BDT':
            results_fns, _, regr_bdt, scores, spl, _ = classifier.classify_boosted_dts_image_match(arg)
        elif options['classifier'] == 'CONVNET':
            results_fns, results_rmatches, _, scores, spl, _ = convnet.classify_convnet_image_match_inference(arg)
            # print '='*100
            # print results_rmatches
            # print spl
            # import sys; sys.exit(1)
        print ("\tFinished classifying data for {} using {}".format(t.split('/')[-1], options['classifier']))  

        # analysis_data = np.concatenate(( \
        #     fns_te, \
        #     num_rmatches_te.reshape((len(labels_te),1)), \
        #     vt_rank_percentage_im1_im2_te.reshape((len(labels_te),1)), \
        #     vt_rank_percentage_im2_im1_te.reshape((len(labels_te),1)), \
        #     scores.reshape((len(labels_te),1)), \
        #     labels_te.reshape((len(labels_te),1))
        # ), axis=1)

        # fn = os.path.join(data_folder, t.split('/')[-1]) + '.json'
        # save_analysis_data(analysis_data, fn)

        for i,(im1,im2) in enumerate(results_fns):
            if im1 not in results:
                results[im1] = {}
            if im2 not in results:
                results[im2] = {}

            score = round(scores[i], 3)
            results[im1][im2] = {'im1': im1, 'im2': im2, 'score': score, 'num_rmatches': results_rmatches[i], 'shortest_path_length': spl[i]}
            results[im2][im1] = {'im1': im2, 'im2': im1, 'score': score, 'num_rmatches': results_rmatches[i], 'shortest_path_length': spl[i]}
        

        # print (results)
        # print (json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))

        data.save_image_matching_results(results, options['robust_matches_threshold'])

def get_precision_recall(fns, labels, criteria, k, debug=False):
    raw_results = []
    aggregated_results = [0.0, 0.0]
    unique_fns = sorted(list(set(np.concatenate((fns[:,0], fns[:,1])).tolist())))
    for f in unique_fns:
        if debug:
            print ('\tPrecision/Recall for image "{}" at top {}'.format(f, k))

        ri = np.where((fns[:,0] == f) | (fns[:,1] == f))[0]
        criteria_ = criteria[ri]
        labels_ = labels[ri]

        order = criteria_.argsort()[::-1][:k]

        precision = np.sum(labels_[order]) / len(labels_[order])
        if np.sum(labels_) == 0:
            recall = 1.0
        else:
            recall = np.sum(labels_[order]) / np.sum(labels_)

        raw_results.append([f, precision, recall])
        aggregated_results[0] += precision
        aggregated_results[1] += recall
        if debug:
            print ('\t\tPrecision: {}  Recall: {}'.format(precision, recall))
    aggregated_results[0] /= len(unique_fns)
    aggregated_results[1] /= len(unique_fns)
    return raw_results, aggregated_results

def compare_metrics(metric1, metric2):
    p_ties, r_ties = 0, 0
    metric1_p_wins, metric1_r_wins = 0, 0
    metric2_p_wins, metric2_r_wins = 0, 0

    for i,_ in enumerate(metric1):
        f1, p1, r1 = metric1[i]
        f2, p2, r2 = metric2[i]

        if p1 == p2:
            p_ties += 1
        elif p1 > p2:
            metric1_p_wins += 1
        elif p2 > p1:
            metric2_p_wins += 1

        if r1 == r2:
            r_ties += 1
        elif r1 > r2:
            metric1_r_wins += 1
        elif r2 > r1:
            metric2_r_wins += 1

        if f1 != f2:
            print ('******** ERROR ********')
            print ('{}: {}/{}    {}: {}/{}'.format(f1, p1,r1,f2, p2,r2))
            import sys;sys.exit(1)
    return metric1_p_wins, metric1_r_wins, p_ties, r_ties, metric2_p_wins, metric2_r_wins

def analyze_datasets(datasets, options={}):
    data_folder = 'data/image-matching-classifiers-analysis'

    for i,t in enumerate(datasets):
        dataset_name = t.split('/')[-1]

        data_fn = os.path.join(data_folder, dataset_name) + '.json'
        analysis_data = load_analysis_data(data_fn)
        print ('#'*100)
        print ('Dataset: {}  Examples: {}'.format(dataset_name, len(analysis_data)))
        
        fns = analysis_data[:,0:2]
        num_rmatches = analysis_data[:,2].astype(np.float)
        vt_rank_percentage_im1_im2 = analysis_data[:,3].astype(np.float)
        vt_rank_percentage_im2_im1 = analysis_data[:,4].astype(np.float)
        scores = analysis_data[:,5].astype(np.float)
        labels = analysis_data[:,6].astype(np.float)

        rmatches_precisions, rmatches_recalls = [], []
        scores_precisions, scores_recalls = [], []
        labels_precisions, labels_recalls = [], []

        rmatches_classifier_precision_wins, rmatches_classifier_recall_wins = [], []
        classifier_rmatches_precision_wins, classifier_rmatches_recall_wins = [], []
        rmatches_classifier_precision_ties, rmatches_classifier_recall_ties = [], []

        rmatches_gt_precision_wins, rmatches_gt_recall_wins = [], []
        gt_rmatches_precision_wins, gt_rmatches_recall_wins = [], []
        rmatches_gt_precision_ties, rmatches_gt_recall_ties = [], []

        vt_classifier_precision_wins, vt_classifier_recall_wins = [], []
        classifier_vt_precision_wins, classifier_vt_recall_wins = [], []
        vt_classifier_precision_ties, vt_classifier_recall_ties = [], []

        vt_rank_mean = (vt_rank_percentage_im1_im2 + vt_rank_percentage_im2_im1) / 2.0
        # ranks = [1, 2, 5, 10, 15, 20, 30, 40, 50]
        ranks = [1]
        for k in ranks:
            raw_results_rmatches, mean_results_rmatches = get_precision_recall(fns, labels, criteria=num_rmatches, k=k)
            raw_results_vt, mean_results_vt = get_precision_recall(fns, labels, criteria=vt_rank_mean, k=k)
            raw_results_scores, mean_results_scores = get_precision_recall(fns, labels, criteria=scores, k=k)
            raw_results_labels, mean_results_labels = get_precision_recall(fns, labels, criteria=labels, k=k)
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'rmatches', mean_results_rmatches[0], mean_results_rmatches[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'vt', mean_results_vt[0], mean_results_vt[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'scores', mean_results_scores[0], mean_results_scores[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}\n'.format(k, dataset_name, 'labels', mean_results_labels[0], mean_results_labels[1]))
            
            rmatches_classifier_p_wins, rmatches_classifier_r_wins, \
                rmatches_classifier_ties_p, rmatches_classifier_ties_r, \
                classifier_rmatches_p_wins, classifier_rmatches_r_wins = \
                compare_metrics(raw_results_rmatches, raw_results_scores)

            rmatches_gt_p_wins, rmatches_gt_r_wins, \
                rmatches_gt_ties_p, rmatches_gt_ties_r, \
                gt_rmatches_p_wins, gt_rmatches_r_wins = \
                compare_metrics(raw_results_rmatches, raw_results_labels)

            vt_classifier_p_wins, vt_classifier_r_wins, \
                vt_classifier_ties_p, vt_classifier_ties_r, \
                classifier_vt_p_wins, classifier_vt_r_wins = \
                compare_metrics(raw_results_vt, raw_results_scores)


            rmatches_precisions.append(mean_results_rmatches[0])
            rmatches_recalls.append(mean_results_rmatches[1])
            scores_precisions.append(mean_results_scores[0])
            scores_recalls.append(mean_results_scores[1])
            labels_precisions.append(mean_results_labels[0])
            labels_recalls.append(mean_results_labels[1])

            rmatches_classifier_precision_wins.append(rmatches_classifier_p_wins)
            rmatches_classifier_recall_wins.append(rmatches_classifier_r_wins)
            classifier_rmatches_precision_wins.append(classifier_rmatches_p_wins)
            classifier_rmatches_recall_wins.append(classifier_rmatches_r_wins)
            rmatches_classifier_precision_ties.append(rmatches_classifier_ties_p)
            rmatches_classifier_recall_ties.append(rmatches_classifier_ties_r)

            rmatches_gt_precision_wins.append(rmatches_gt_p_wins)
            rmatches_gt_recall_wins.append(rmatches_gt_r_wins)
            gt_rmatches_precision_wins.append(gt_rmatches_p_wins)
            gt_rmatches_recall_wins.append(gt_rmatches_r_wins)
            rmatches_gt_precision_ties.append(rmatches_gt_ties_p)
            rmatches_gt_recall_ties.append(rmatches_gt_ties_r)

            vt_classifier_precision_wins.append(vt_classifier_p_wins)
            vt_classifier_recall_wins.append(vt_classifier_r_wins)
            classifier_vt_precision_wins.append(classifier_vt_p_wins)
            classifier_vt_recall_wins.append(classifier_vt_r_wins)
            vt_classifier_precision_ties.append(vt_classifier_ties_p)
            vt_classifier_recall_ties.append(vt_classifier_ties_r)


        plt.figure(1)
        plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        plt.ylabel('Count')
        plt.xlabel('k')
        plt.title('Dataset: {}'.format(dataset_name), fontsize=18)
        plt.plot(ranks, rmatches_classifier_precision_wins, '--', linewidth=2)
        #plt.plot(ranks, rmatches_classifier_recall_wins, dashes=[30, 5, 10, 5])
        plt.plot(ranks, classifier_rmatches_precision_wins, '--', linewidth=2)
        #plt.plot(ranks, classifier_rmatches_recall_wins, dashes=[30, 5, 10, 5])
        plt.plot(ranks, rmatches_classifier_precision_ties, '--', linewidth=2)
        #plt.plot(ranks, rmatches_classifier_recall_ties, dashes=[30, 5, 10, 5])
        plt.legend([
            'rmatches precision winner', \
            #'rmatches recall winner', \
            'classifier precision winner', \
            #'classifier recall winner', \
            'precision ties', \
            #'recall ties'
            ], 
            loc='lower left',
            shadow=True,
            fontsize=10
            )

        plt.figure(2)
        plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        plt.ylabel('Count')
        plt.xlabel('k')
        plt.title('Dataset: {}'.format(dataset_name), fontsize=18)
        plt.plot(ranks, rmatches_gt_precision_wins, '--', linewidth=2)
        #plt.plot(ranks, rmatches_gt_recall_wins, dashes=[30, 5, 10, 5])
        plt.plot(ranks, gt_rmatches_precision_wins, '--', linewidth=2)
        #plt.plot(ranks, gt_rmatches_recall_wins, dashes=[30, 5, 10, 5])
        plt.plot(ranks, rmatches_gt_precision_ties, '--', linewidth=2)
        #plt.plot(ranks, rmatches_gt_recall_ties, dashes=[30, 5, 10, 5])
        plt.legend([
            'rmatches precision winner', \
            #'rmatches recall winner', \
            'gt precision winner', \
            #'gt recall winner', \
            'precision ties', \
            #'recall ties'
            ], 
            loc='lower left',
            shadow=True,
            fontsize=10
            )

        plt.figure(3)
        plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        plt.ylabel('Count')
        plt.xlabel('k')
        plt.title('Dataset: {}'.format(dataset_name), fontsize=18)
        plt.plot(ranks, vt_classifier_precision_wins, '--', linewidth=2)
        # plt.plot(ranks, vt_classifier_recall_wins, dashes=[30, 5, 10, 5])
        plt.plot(ranks, classifier_vt_precision_wins, '--', linewidth=2)
        # plt.plot(ranks, classifier_vt_recall_wins, dashes=[30, 5, 10, 5])
        plt.plot(ranks, vt_classifier_precision_ties, '--', linewidth=2)
        # plt.plot(ranks, vt_classifier_recall_ties, dashes=[30, 5, 10, 5])
        plt.legend([
            'vt precision winner', \
            #'vt recall winner', \
            'classifier precision winner', \
            #'classifier recall winner', \
            'precision ties', \
            #'recall ties'
            ], 
            loc='lower left',
            shadow=True,
            fontsize=10
            )

        plt.figure(4)
        plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.ylim(0,1.0)
        plt.xlim(0,1.0)
        plt.title('Dataset: {}'.format(dataset_name), fontsize=18)

        plt.plot(rmatches_recalls, rmatches_precisions)
        plt.plot(scores_recalls, scores_precisions)
        plt.plot(labels_recalls, labels_precisions)

        plt.legend(
            ['Baseline - rmatches', 'Classifier scores', 'Ground-truth labels'], 
            loc='lower left',
            shadow=True,
            fontsize=10
            )


    plt.figure(1)
    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(data_folder, 'rmatches_classifier_pr_winners.png'))

    plt.figure(2)
    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(data_folder, 'rmatches_gt_pr_winners.png'))

    plt.figure(3)
    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(data_folder, 'vt_classifier_pr_winners.png'))

    plt.figure(4)
    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(data_folder, 'MeanPerImagePR.png'))
    
def analyze_datasets2(datasets, options):
    data_folder = 'data/image-matching-classifiers-analysis'
    analysis_data = {}
    
    for i,t in enumerate(datasets):
        dataset_name = t.split('/')[-1]

        data_fn = os.path.join(data_folder, dataset_name) + '.json'
        if os.path.exists(data_fn):
            analysis_data[dataset_name] = load_analysis_data(data_fn)
        else:
            continue

        print ('#'*100)
        print ('Dataset: {}  Examples: {}'.format(dataset_name, len(analysis_data[dataset_name])))

    for i, dataset_name in enumerate(analysis_data.keys()):
        dset = np.tile(dataset_name, (len(analysis_data[dataset_name]),))
        if i == 0:
            data = analysis_data[dataset_name]
            dsets = dset
        else:
            data = np.concatenate((data, analysis_data[dataset_name]), axis=0)
            dsets = np.concatenate((dsets, dset), axis=0)


    fns = data[:,0:2]
    num_rmatches = data[:,2].astype(np.float)
    vt_rank_percentage_im1_im2 = data[:,3].astype(np.float)
    vt_rank_percentage_im2_im1 = data[:,4].astype(np.float)
    scores = data[:,5].astype(np.float)
    labels = data[:,6].astype(np.float)

        # rmatches_precisions, rmatches_recalls = [], []
        # scores_precisions, scores_recalls = [], []
        # labels_precisions, labels_recalls = [], []

        # rmatches_classifier_precision_wins, rmatches_classifier_recall_wins = [], []
        # classifier_rmatches_precision_wins, classifier_rmatches_recall_wins = [], []
        # rmatches_classifier_precision_ties, rmatches_classifier_recall_ties = [], []

        # rmatches_gt_precision_wins, rmatches_gt_recall_wins = [], []
        # gt_rmatches_precision_wins, gt_rmatches_recall_wins = [], []
        # rmatches_gt_precision_ties, rmatches_gt_recall_ties = [], []

        # vt_classifier_precision_wins, vt_classifier_recall_wins = [], []
        # classifier_vt_precision_wins, classifier_vt_recall_wins = [], []
        # vt_classifier_precision_ties, vt_classifier_recall_ties = [], []

    vt_rank_mean = (vt_rank_percentage_im1_im2 + vt_rank_percentage_im2_im1) / 2.0

    plt.figure(1)

    dsets_rm, fns_rm, p_r_auc_rm, auc_dset_means_rm, _, auc_overall_mean_rm, _ = matching_classifiers.calculate_per_image_mean_auc(dsets, fns, num_rmatches, labels)
    dsets_cl, fns_cl, p_r_auc_cl, auc_dset_means_cl, _, auc_overall_mean_cl, _ = matching_classifiers.calculate_per_image_mean_auc(dsets, fns, scores, labels)
    dsets_gt, fns_gt, p_r_auc_gt, auc_dset_means_gt, _, auc_overall_mean_gt, _ = matching_classifiers.calculate_per_image_mean_auc(dsets, fns, labels, labels)
    auc_entire_ds_rm, _ = matching_classifiers.calculate_dataset_auc(num_rmatches, labels, color='red', ls='dashed')
    auc_entire_ds_cl, _ = matching_classifiers.calculate_dataset_auc(scores, labels, color='red', ls='dashed')

    for i, (d, _) in enumerate(auc_dset_means_rm):
        print ('-'*100)
        print ('\trmatches: Dataset: {}  AUC: {}'.format(d, auc_dset_means_rm[i][1]))
        print ('\tclassifier: Dataset: {}  AUC: {}'.format(d, auc_dset_means_cl[i][1]))
        print ('\tground-truth: Dataset: {}  AUC: {}'.format(d, auc_dset_means_gt[i][1]))
        # print '#'*100
        # print auc_dset_means_rm
        for j,dset in enumerate(dsets_rm):
            if dset == auc_dset_means_rm[i][0]:
                print ('\t\t{} / {}|{} : {} | {}'.format(dset,fns_rm[j], fns_cl[j], round(p_r_auc_rm[j][2],3), round(p_r_auc_cl[j][2],3)))
        print '='*100

    print ('='*100)
    print ('rmatches: Overall AUC: {}'.format(auc_overall_mean_rm))
    print ('classifier: Overall AUC: {}'.format(auc_overall_mean_cl))
    print ('ground-truth: Overall AUC: {}'.format(auc_overall_mean_gt))
    print ('rmatches: Entire Dataset AUC: {}'.format(auc_entire_ds_rm))
    print ('classifier: Entire Dataset AUC: {}'.format(auc_entire_ds_cl))
    # print ('ground-truth: Overall AUC: {}'.format(auc_overall_mean_gt))
    # plt.show()

    
    # fig = plt.gcf()
    # fig.set_size_inches(37, 21)
    # plt.savefig(os.path.join(data_folder, 'aucs-per-image.png'))

    #     plt.subplot(2, math.ceil(len(_dsets)/2.0), i + 1)
    #     plt.ylabel('Precision')
    #     plt.xlabel('Recall')
    #     plt.ylim(0,1.0)
    #     plt.xlim(0,1.0)
    #     plt.title('Dataset: {}'.format(d), fontsize=18)

    #     plt.plot(rmatches_recalls, rmatches_precisions)
    #     plt.plot(scores_recalls, scores_precisions)
    #     plt.plot(labels_recalls, labels_precisions)

    # plt.legend(
    #     ['Baseline - rmatches', 'Classifier scores', 'Ground-truth labels'], 
    #     loc='lower left',
    #     shadow=True,
    #     fontsize=10
    #     )

    # calculate_per_image_mean_auc(dsets, fns, scores, labels, color='green', ls='dashed')
        # ranks = [1, 2, 5, 10, 15, 20, 30, 40, 50]
        # for k in ranks:
            # raw_results_rmatches, mean_results_rmatches = get_precision_recall(fns, labels, criteria=num_rmatches, k=k)
            # raw_results_vt, mean_results_vt = get_precision_recall(fns, labels, criteria=vt_rank_mean, k=k)
            # raw_results_scores, mean_results_scores = get_precision_recall(fns, labels, criteria=scores, k=k)
            # raw_results_labels, mean_results_labels = get_precision_recall(fns, labels, criteria=labels, k=k)
            # print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'rmatches', mean_results_rmatches[0], mean_results_rmatches[1]))
            # print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'vt', mean_results_vt[0], mean_results_vt[1]))
            # print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'scores', mean_results_scores[0], mean_results_scores[1]))
            # print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'labels', mean_results_labels[0], mean_results_labels[1]))
            
            # rmatches_classifier_p_wins, rmatches_classifier_r_wins, \
            #     rmatches_classifier_ties_p, rmatches_classifier_ties_r, \
            #     classifier_rmatches_p_wins, classifier_rmatches_r_wins = \
            #     compare_metrics(raw_results_rmatches, raw_results_scores)

            # rmatches_gt_p_wins, rmatches_gt_r_wins, \
            #     rmatches_gt_ties_p, rmatches_gt_ties_r, \
            #     gt_rmatches_p_wins, gt_rmatches_r_wins = \
            #     compare_metrics(raw_results_rmatches, raw_results_labels)

            # vt_classifier_p_wins, vt_classifier_r_wins, \
            #     vt_classifier_ties_p, vt_classifier_ties_r, \
            #     classifier_vt_p_wins, classifier_vt_r_wins = \
            #     compare_metrics(raw_results_vt, raw_results_scores)


            # rmatches_precisions.append(mean_results_rmatches[0])
            # rmatches_recalls.append(mean_results_rmatches[1])
            # scores_precisions.append(mean_results_scores[0])
            # scores_recalls.append(mean_results_scores[1])
            # labels_precisions.append(mean_results_labels[0])
            # labels_recalls.append(mean_results_labels[1])

            # rmatches_classifier_precision_wins.append(rmatches_classifier_p_wins)
            # rmatches_classifier_recall_wins.append(rmatches_classifier_r_wins)
            # classifier_rmatches_precision_wins.append(classifier_rmatches_p_wins)
            # classifier_rmatches_recall_wins.append(classifier_rmatches_r_wins)
            # rmatches_classifier_precision_ties.append(rmatches_classifier_ties_p)
            # rmatches_classifier_recall_ties.append(rmatches_classifier_ties_r)

            # rmatches_gt_precision_wins.append(rmatches_gt_p_wins)
            # rmatches_gt_recall_wins.append(rmatches_gt_r_wins)
            # gt_rmatches_precision_wins.append(gt_rmatches_p_wins)
            # gt_rmatches_recall_wins.append(gt_rmatches_r_wins)
            # rmatches_gt_precision_ties.append(rmatches_gt_ties_p)
            # rmatches_gt_recall_ties.append(rmatches_gt_ties_r)

            # vt_classifier_precision_wins.append(vt_classifier_p_wins)
            # vt_classifier_recall_wins.append(vt_classifier_r_wins)
            # classifier_vt_precision_wins.append(classifier_vt_p_wins)
            # classifier_vt_recall_wins.append(classifier_vt_r_wins)
            # vt_classifier_precision_ties.append(vt_classifier_ties_p)
            # vt_classifier_recall_ties.append(vt_classifier_ties_r)


        # plt.figure(1)
        # plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        # plt.ylabel('Count')
        # plt.xlabel('k')
        # plt.title('Dataset: {}'.format(dataset_name), fontsize=18)
        # plt.plot(ranks, rmatches_classifier_precision_wins, '--', linewidth=2)
        # #plt.plot(ranks, rmatches_classifier_recall_wins, dashes=[30, 5, 10, 5])
        # plt.plot(ranks, classifier_rmatches_precision_wins, '--', linewidth=2)
        # #plt.plot(ranks, classifier_rmatches_recall_wins, dashes=[30, 5, 10, 5])
        # plt.plot(ranks, rmatches_classifier_precision_ties, '--', linewidth=2)
        # #plt.plot(ranks, rmatches_classifier_recall_ties, dashes=[30, 5, 10, 5])
        # plt.legend([
        #     'rmatches precision winner', \
        #     #'rmatches recall winner', \
        #     'classifier precision winner', \
        #     #'classifier recall winner', \
        #     'precision ties', \
        #     #'recall ties'
        #     ], 
        #     loc='lower left',
        #     shadow=True,
        #     fontsize=10
        #     )

        # plt.figure(2)
        # plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        # plt.ylabel('Count')
        # plt.xlabel('k')
        # plt.title('Dataset: {}'.format(dataset_name), fontsize=18)
        # plt.plot(ranks, rmatches_gt_precision_wins, '--', linewidth=2)
        # #plt.plot(ranks, rmatches_gt_recall_wins, dashes=[30, 5, 10, 5])
        # plt.plot(ranks, gt_rmatches_precision_wins, '--', linewidth=2)
        # #plt.plot(ranks, gt_rmatches_recall_wins, dashes=[30, 5, 10, 5])
        # plt.plot(ranks, rmatches_gt_precision_ties, '--', linewidth=2)
        # #plt.plot(ranks, rmatches_gt_recall_ties, dashes=[30, 5, 10, 5])
        # plt.legend([
        #     'rmatches precision winner', \
        #     #'rmatches recall winner', \
        #     'gt precision winner', \
        #     #'gt recall winner', \
        #     'precision ties', \
        #     #'recall ties'
        #     ], 
        #     loc='lower left',
        #     shadow=True,
        #     fontsize=10
        #     )

        # plt.figure(3)
        # plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
        # plt.ylabel('Count')
        # plt.xlabel('k')
        # plt.title('Dataset: {}'.format(dataset_name), fontsize=18)
        # plt.plot(ranks, vt_classifier_precision_wins, '--', linewidth=2)
        # # plt.plot(ranks, vt_classifier_recall_wins, dashes=[30, 5, 10, 5])
        # plt.plot(ranks, classifier_vt_precision_wins, '--', linewidth=2)
        # # plt.plot(ranks, classifier_vt_recall_wins, dashes=[30, 5, 10, 5])
        # plt.plot(ranks, vt_classifier_precision_ties, '--', linewidth=2)
        # # plt.plot(ranks, vt_classifier_recall_ties, dashes=[30, 5, 10, 5])
        # plt.legend([
        #     'vt precision winner', \
        #     #'vt recall winner', \
        #     'classifier precision winner', \
        #     #'classifier recall winner', \
        #     'precision ties', \
        #     #'recall ties'
        #     ], 
        #     loc='lower left',
        #     shadow=True,
        #     fontsize=10
        #     )

    # plt.figure(4)
    # plt.subplot(2, math.ceil(len(datasets)/2.0), i + 1)
    # plt.ylabel('Precision')
    # plt.xlabel('Recall')
    # plt.ylim(0,1.0)
    # plt.xlim(0,1.0)
    # plt.title('Dataset: {}'.format(dataset_name), fontsize=18)

    # plt.plot(rmatches_recalls, rmatches_precisions)
    # plt.plot(scores_recalls, scores_precisions)
    # plt.plot(labels_recalls, labels_precisions)

    # plt.legend(
    #     ['Baseline - rmatches', 'Classifier scores', 'Ground-truth labels'], 
    #     loc='lower left',
    #     shadow=True,
    #     fontsize=10
    #     )


    # plt.figure(1)
    # fig = plt.gcf()
    # fig.set_size_inches(37, 21)
    # plt.savefig(os.path.join(data_folder, 'rmatches_classifier_pr_winners.png'))

    # plt.figure(2)
    # fig = plt.gcf()
    # fig.set_size_inches(37, 21)
    # plt.savefig(os.path.join(data_folder, 'rmatches_gt_pr_winners.png'))

    # plt.figure(3)
    # fig = plt.gcf()
    # fig.set_size_inches(37, 21)
    # plt.savefig(os.path.join(data_folder, 'vt_classifier_pr_winners.png'))

    plt.figure(4)
    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(os.path.join(data_folder, 'MeanPerImagePR.png'))

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    parser.add_argument('-x', '--image_match_classifier_max_match', help='')
    parser.add_argument('-c', '--classifier_file', help='classifier to run')
    parser.add_argument('--classifier', help='classifier type - BDT/CONVNET')
    # parser.add_argument('--convnet_checkpoint', help='checkpoint file for convnet')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset

    options = {
        'classifier': parser_options.classifier, \
        'image_match_classifier_file': parser_options.classifier_file, \
        'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        'feature_selection': False,
        'convnet_checkpoint': parser_options.classifier_file,
        'robust_matches_threshold': 15
    }

    training_datasets = [
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Barn',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Caterpillar',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Church',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Courthouse',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Ignatius',
    
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/courtyard',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/delivery_area',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/electro',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/facade',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/kicker',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/meadow',
    
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
            
    ]


    testing_datasets = [
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Meetingroom',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/TanksAndTemples/Truck',

        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/exhibition_hall',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/lecture_room',
        '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/living_room',

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
    ]

    training_datasets = ['/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/kicker']
    # testing_datasets = ['/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/lecture_room']

    testing_datasets = ['/hdd/Research/psfm-iccv/data/botanical_garden']


    iccv_datasets = [
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/botanical_garden',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/boulders',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/bridge',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/door',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/exhibition_hall',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/lecture_room',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/living_room',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/lounge',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/observatory',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/office',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/old_computer',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/pipes',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/playground',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/relief',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/relief_2',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/statue',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/terrace',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/terrace_2',
        '/hdd/Research/psfm-iccv/data/iccv-results/ETH3D/terrains',

        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor2_hall',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor3_loop',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor3_loop_ccw',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor3_loop_cw',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor5',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor5_stairs',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_floor5_wall',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/ece_stairs',
        # # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/fox_floor2_extra',
        # # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/fox_floor3_extra',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_day_all',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_day_atrium',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_day_backward',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_day_forward',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_night_all',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_night_atrium',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_night_backward',
        # '/hdd/Research/psfm-iccv/data/iccv-results/UIUCTag/yeh_night_forward',

        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Auditorium',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Ballroom',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Courtroom',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Family',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Francis',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Horse',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Lighthouse',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/M60',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Meetingroom',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Museum',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Palace',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Panther',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Playground',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Temple',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Train',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TanksAndTemples/Truck',

        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static',
        # '/hdd/Research/psfm-iccv/data/iccv-results/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz',
    ]

    # iccv_datasets = [
    #     '/hdd/Research/psfm-iccv/data/completed-classifier-datasets/ETH3D/courtyard/colmap',
    # ]
    # iccv_datasets = [
    #     '/hdd/Research/psfm-iccv/data/iccv-results/Sudipta/group3/oats/',
    # ]

    classify_images(iccv_datasets, options)
    # classification_and_reconstruction_correlation_analysis(datasets, options)
    # plot_iccv_figures(training_datasets, testing_datasets, options)
    
    # analyze_datasets(datasets, options)
    # analyze_datasets2(datasets, options)

if __name__ == '__main__':
    main(sys.argv)
