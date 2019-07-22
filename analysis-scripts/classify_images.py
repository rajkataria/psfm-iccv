import gzip
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import os
import pickle
import sys
import torch

from argparse import ArgumentParser
import convnet

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def rmatches_exists(outdir, dataset_name, label):
    os.path.isfile(os.path.join(outdir, '{}-{}.pkl.gz'.format(dataset_name, label)))

def load_rmatches(outdir, dataset_name, label):
    with gzip.open(os.path.join(outdir, '{}-{}.pkl.gz'.format(dataset_name, label)), 'rb') as fin:
        results = pickle.load(fin)
    return results

def save_rmatches(outdir, dataset_name, num_rmatches, label):
    with gzip.open(os.path.join(outdir, '{}-{}.pkl.gz'.format(dataset_name, label)), 'wb') as fout:
        pickle.dump(num_rmatches, fout)

def plot_baseline_histograms(outdir, inliers_distribution, outliers_distribution, bins, x_max):
    fontsize = 12
    epsilon = 0.0000001


    plt.subplot(4,1,1)
    # plt.xlim(0, 51)
    plt.ylim(0, 1.1)
    plt.xlabel('rmatches', fontsize=fontsize)
    plt.ylabel('Inlier %', fontsize=fontsize)
    # plt.plot(np.linspace(8,50,41), (inliers_distribution_8_50[0].astype(np.float) + epsilon)/(inliers_distribution_8_50[0].astype(np.float) + outliers_distribution_8_50[0].astype(np.float) + epsilon))
    plt.plot(bins[:-16], (inliers_distribution[0].astype(np.float)[:-15] + epsilon)/(inliers_distribution[0].astype(np.float)[:-15] + outliers_distribution[0].astype(np.float)[:-15] + epsilon))

    plt.subplot(4,1,2)
    # plt.xlim(0, 501)
    plt.ylim(0, 1.1)
    plt.xlabel('rmatches', fontsize=fontsize)
    plt.ylabel('Inlier %', fontsize=fontsize)
    # plt.plot(np.linspace(8,500,491), (inliers_distribution_8_500[0].astype(np.float) + epsilon)/(inliers_distribution_8_500[0].astype(np.float) + outliers_distribution_8_500[0].astype(np.float) + epsilon))
    plt.plot(bins[:-11], (inliers_distribution[0].astype(np.float)[:-10] + epsilon)/(inliers_distribution[0].astype(np.float)[:-10] + outliers_distribution[0].astype(np.float)[:-10] + epsilon))

    plt.subplot(4,1,3)
    # plt.xlim(0, 501)
    plt.ylim(0, 1.1)
    plt.xlabel('rmatches', fontsize=fontsize)
    plt.ylabel('Inlier %', fontsize=fontsize)
    # plt.plot(np.linspace(8,500,491), (inliers_distribution_8_500[0].astype(np.float) + epsilon)/(inliers_distribution_8_500[0].astype(np.float) + outliers_distribution_8_500[0].astype(np.float) + epsilon))
    plt.plot(bins[:-8], (inliers_distribution[0].astype(np.float)[:-7] + epsilon)/(inliers_distribution[0].astype(np.float)[:-7] + outliers_distribution[0].astype(np.float)[:-7] + epsilon))

    plt.subplot(4,1,4)
    plt.xlim(0, x_max)
    plt.ylim(0, 1.1)
    plt.xlabel('rmatches', fontsize=fontsize)
    plt.ylabel('Inlier %', fontsize=fontsize)
    # plt.plot(np.linspace(8,x_max,nbins), (inliers_distribution[0].astype(np.float) + epsilon)/(inliers_distribution[0].astype(np.float) + outliers_distribution[0].astype(np.float) + epsilon))
    # import pdb; pdb.set_trace()
    plt.plot(bins[:-1], (inliers_distribution[0].astype(np.float) + epsilon)/(inliers_distribution[0].astype(np.float) + outliers_distribution[0].astype(np.float) + epsilon))

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(outdir, 'baseline-histograms.png'))
    plt.clf()

def create_baseline_classifier(training_datasets):
    num_rmatches_inliers = []
    num_rmatches_outliers = []
    outdir = 'data/baseline-classifiers'
    mkdir_p(outdir)

    print ('#'*100)
    print ('#'*100)
    for ii,t in enumerate(training_datasets):
        dataset_name = os.path.basename(t)
        print ('\tCreating baseline histogram classifier using dataset: {}'.format(dataset_name))
        if rmatches_exists(outdir, dataset_name, 'inliers') and rmatches_exists(outdir, dataset_name, 'outliers'):
            inliers_distribution = load_rmatches(outdir, dataset_name, 'inliers')
            outliers_distribution = load_rmatches(outdir, dataset_name, 'outliers')
        else:
            data = dataset.DataSet(t)
            _num_rmatches_inliers = []
            _num_rmatches_outliers = []

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
                = data.load_image_matching_dataset(robust_matches_threshold=15, rmatches_min_threshold=0, \
                    rmatches_max_threshold=5000, spl=200000)

            for j, _ in enumerate(_fns):
                if _labels[j] >= 1:
                    _num_rmatches_inliers.append(_num_rmatches[j])
                else:
                    _num_rmatches_outliers.append(_num_rmatches[j])
            
            save_rmatches(outdir, dataset_name, _num_rmatches_inliers, 'inliers')
            save_rmatches(outdir, dataset_name, _num_rmatches_outliers, 'outliers')

        num_rmatches_inliers.extend(_num_rmatches_inliers)
        num_rmatches_outliers.extend(_num_rmatches_outliers)

    print ('#'*100)
    print ('#'*100)

    x_max = int(max(num_rmatches_inliers))
    # nbins = (x_max-8-1) / 1
    bins = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 46, 54, 70, 102, 134, 166, 198, 230, 294, 422, 550, 678, 806, 934, 1062, 2086, 3110, 4134, 5158]

    inliers_distribution = np.histogram(num_rmatches_inliers, bins=bins, range=(8,x_max))
    outliers_distribution = np.histogram(num_rmatches_outliers, bins=bins, range=(8,x_max))

    # inliers_distribution_8_50 = np.histogram(num_rmatches_inliers, bins=41, range=(8,50))
    # outliers_distribution_8_50 = np.histogram(num_rmatches_outliers, bins=41, range=(8,50))

    # inliers_distribution_8_500 = np.histogram(num_rmatches_inliers, bins=491, range=(8,500))
    # outliers_distribution_8_500 = np.histogram(num_rmatches_outliers, bins=491, range=(8,500))

    # plot_baseline_histograms(outdir, inliers_distribution_8_50, outliers_distribution_8_50, inliers_distribution_8_500, outliers_distribution_8_500, inliers_distribution, outliers_distribution, bins, x_max)
    plot_baseline_histograms(outdir, inliers_distribution, outliers_distribution, bins, x_max)
    np.save(os.path.join(outdir, 'inliers_distribution'), inliers_distribution)
    np.save(os.path.join(outdir, 'outliers_distribution'), outliers_distribution)


# def get_histogram_bin_index(bins, value):
#     index = min(range(len(bins)), key=lambda i: abs(bins[i]-value))
#     if value > bins[index]:
#         return index + 1
#     return index
def baseline_histogram_classifier(arg):
    epsilon = 0.0000001
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
        labels_te, weights_te, train, trained_classifier, options = arg

    # only rmatches and trained_classifier(which is the model) are significant here
    inliers_distribution, outliers_distribution = trained_classifier
    bins = trained_classifier[0][1]
    relevant_bins = np.digitize(num_rmatches_te, bins) - 1
    inlier_percentage = (inliers_distribution[0][relevant_bins].astype(np.float) + epsilon) / (inliers_distribution[0][relevant_bins].astype(np.float) + outliers_distribution[0][relevant_bins].astype(np.float) + epsilon)

    return fns_te, num_rmatches_te, None, inlier_percentage, shortest_path_length_te, None
    
def classify_images(datasets, options={}):    
    print ('-'*100)
    print ('Classifier: {}'.format(options['classifier']))
    shortest_path_length = 1000000
    if options['classifier'] == 'BASELINE':
        inliers_distribution = np.load('data/baseline-classifiers/inliers_distribution.npy')
        outliers_distribution = np.load('data/baseline-classifiers/outliers_distribution.npy')
        trained_classifier = [inliers_distribution, outliers_distribution]
    elif options['classifier'] == 'CONVNET':
        # load convnet checkpoint here
        print ('Loading ConvNet checkpoint: {}'.format(options['classifier_file']))
        checkpoint = torch.load(options['classifier_file'])
        kwargs = {}
        options['model'] = 'resnet18'
        # options['convnet_lr'] = 0.01
        options['convnet_use_images'] = False
        options['convnet_use_warped_images'] = False
        options['convnet_use_feature_match_map'] = False
        options['convnet_use_track_map'] = False
        options['convnet_use_non_rmatches_map'] = True
        options['convnet_use_rmatches_map'] = True
        options['convnet_use_matches_map'] = False
        options['convnet_use_photometric_error_maps'] = True
        options['convnet_use_rmatches_secondary_motion_map'] = False

        options['range_min'] = 0
        options['range_max'] = 5000
        options['loss'] = 'cross-entropy'

        options['features'] = 'RM'
        options['mlp-layer-size'] = 256
        options['use_small_weights'] = False
        options['num_workers'] = 10
        options['batch_size'] = 64
        options['shuffle'] = False
        options['convnet_input_size'] = 224
        options['triplet-sampling-strategy'] = 'normal'
        options['log_interval'] = 1
        options['experiment'] = 'pe_unfiltered+use-BF-data'

        trained_classifier = convnet.ConvNet(options, **kwargs)
        trained_classifier.cuda()
        trained_classifier.load_state_dict(checkpoint['state_dict'])

    print ('-'*100)
    print ('-'*100)

    for i,t in enumerate(datasets):
        print ('#'*100)
        print ('\tRunning classifier for dataset: {}'.format(os.path.basename(t)))
        data = dataset.DataSet(t)
        results = {}
        dsets_te_ = []
        fns_te_ = []
        num_rmatches_te_ = []
        # print data.all_feature_maps()
        # import sys; sys.exit(1)
        for im1 in sorted(data.all_feature_maps()):
            im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
            for im2 in im1_all_robust_matches:
                rmatches = im1_all_robust_matches[im2]
                # if im1 == 'DSC_1746.JPG' and im2 == 'DSC_1754.JPG':
                #     import pdb; pdb.set_trace()
                if len(rmatches) == 0:
                    if im1 not in results:
                        results[im1] = {}
                    if im2 not in results:
                        results[im2] = {}
                    
                    results[im1][im2] = {'im1': im1, 'im2': im2, 'score': 0.0, 'num_rmatches': 0.0, 'shortest_path_length': 0}
                    results[im2][im1] = {'im1': im2, 'im2': im1, 'score': 0.0, 'num_rmatches': 0.0, 'shortest_path_length': 0}
                    continue
                fns_te_.append([im1,im2])
                dsets_te_.append(t)
                num_rmatches_te_.append(len(im1_all_robust_matches[im2]))

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
        shortest_path_length_te = np.ones((len(dsets_te),)) #np.array(shortest_paths_te_)
        mds_rank_percentage_im1_im2_te = np.ones((len(dsets_te),))
        mds_rank_percentage_im2_im1_te = np.ones((len(dsets_te),))
        distance_rank_percentage_im1_im2_gt_te = np.ones((len(dsets_te),))
        distance_rank_percentage_im2_im1_gt_te = np.ones((len(dsets_te),))
        labels_te = np.ones((len(dsets_te),))
        
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

        if options['classifier'] == 'BASELINE':
            results_fns, results_rmatches, _, scores, spl, _ = baseline_histogram_classifier(arg)
        elif options['classifier'] == 'CONVNET':
            results_fns, results_rmatches, _, scores, spl, _ = convnet.classify_convnet_image_match_inference(arg)
        print ("\tFinished classifying data for {} using {}".format(t.split('/')[-1], options['classifier']))  

        for i,(im1,im2) in enumerate(results_fns):
            if im1 not in results:
                results[im1] = {}
            if im2 not in results:
                results[im2] = {}

            score = round(scores[i], 3)
            results[im1][im2] = {'im1': im1, 'im2': im2, 'score': score, 'num_rmatches': results_rmatches[i], 'shortest_path_length': spl[i]}
            results[im2][im1] = {'im1': im2, 'im2': im1, 'score': score, 'num_rmatches': results_rmatches[i], 'shortest_path_length': spl[i]}
        
        data.save_image_matching_results(results, robust_matches_threshold=options['robust_matches_threshold'], classifier=options['classifier'])

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    # parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    # parser.add_argument('-x', '--image_match_classifier_max_match', help='')
    parser.add_argument('--classifier_file', help='classifier checkpoint or saved classifier to use')
    parser.add_argument('--classifier', help='classifier type - BASELINE/CONVNET')
    # parser.add_argument('--convnet_checkpoint', help='checkpoint file for convnet')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset

    options = {
        'classifier': parser_options.classifier.upper(), \
        # 'image_match_classifier_file': parser_options.classifier_file, \
        # 'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        # 'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        # 'feature_selection': False,
        'classifier_file': parser_options.classifier_file,
        'robust_matches_threshold': 15
    }

    training_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Barn',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Caterpillar',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Church',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courthouse',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ignatius',
    
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/facade',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/kicker',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/meadow',
    
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
            
    ]

    datasets = [
        # Validation set
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Meetingroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Truck',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lecture_room',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/living_room',

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

        # Test set
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/botanical_garden',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/boulders',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/bridge',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/door',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lounge',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/observatory',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/office',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/old_computer',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/pipes',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/playground',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/relief',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/relief_2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/statue',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/terrace',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/terrace_2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/terrains',

        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor2_hall',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_ccw',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_cw',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_stairs',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_wall',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_stairs',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_all',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_atrium',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_backward',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_forward',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_all',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_atrium',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_backward',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_forward',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Auditorium',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ballroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courtroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Family',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Francis',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Horse',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Lighthouse',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/M60',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Museum',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Palace',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Panther',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Playground',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Temple',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Train',
    ]

    # datasets = ['/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/botanical_garden']
    
    yan_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/books',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/cereal',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/cup',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/desk',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/oats',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/Yan/street'
    ]

    # create_baseline_classifier(training_datasets)

    # classify_images(training_datasets + datasets + yan_datasets, options)
    classify_images(['/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_ccw',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_wall'], options)

if __name__ == '__main__':
    main(sys.argv)
