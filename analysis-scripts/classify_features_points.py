import gzip
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import os
import pickle
import sys
import torch
import matching_classifiers

from multiprocessing import Pool
from timeit import default_timer as timer
from argparse import ArgumentParser

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def feature_point_matches_exists(outdir, dataset_name, label):
    return os.path.isfile(os.path.join(outdir, '{}-{}.pkl.gz'.format(dataset_name, label)))

def load_feature_point_matches(outdir, dataset_name, label):
    with gzip.open(os.path.join(outdir, '{}-{}.pkl.gz'.format(dataset_name, label)), 'rb') as fin:
        results = pickle.load(fin)
    return results

def save_feature_point_matches(outdir, dataset_name, points, label):
    with gzip.open(os.path.join(outdir, '{}-{}.pkl.gz'.format(dataset_name, label)), 'wb') as fout:
        pickle.dump(points, fout)

def plot_baseline_histograms(outdir, inliers_distribution, outliers_distribution, nbins, options):
    fontsize = 12
    epsilon = 0.0000001

    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel("Squared Distances", fontsize=fontsize)
    plt.ylabel('Inlier %', fontsize=fontsize)
    plt.title("Inlier % vs Squared Distances (Lowe's Ratio: {})".format(options['lowes_threshold']))
    plt.plot(np.linspace(0.0,1.0,nbins), (inliers_distribution[0].astype(np.float) + epsilon)/(inliers_distribution[0].astype(np.float) + outliers_distribution[0].astype(np.float) + epsilon))

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(outdir, 'baseline-histograms-{}.png'.format(options['lowes_threshold'])))
    plt.clf()

def create_baseline_classifier(training_datasets, options):
    lowes_ratios_inliers = []
    lowes_ratios_outliers = []
    outdir = options['outdir']
    mkdir_p(outdir)
    s_time = timer()
    nbins = 100
    print ('#'*100)
    print ('#'*100)
    classifier_inliers_fn = os.path.join(outdir, 'inliers_distribution_{}'.format(options['lowes_threshold']))
    classifier_outliers_fn = os.path.join(outdir, 'outliers_distribution_{}'.format(options['lowes_threshold']))

    if matching_classifiers.classifier_exists(classifier_inliers_fn) and matching_classifiers.classifier_exists(classifier_outliers_fn):
        inliers_distribution = matching_classifiers.load_classifier(classifier_inliers_fn)
        outliers_distribution = matching_classifiers.load_classifier(classifier_outliers_fn)
    else:
        for ii,t in enumerate(training_datasets):
            dataset_name = os.path.basename(t)
            print ('\tCreating baseline histogram classifier for feature point matching using dataset: {}'.format(dataset_name))
            if feature_point_matches_exists(outdir, dataset_name, 'inliers-{}'.format(options['lowes_threshold'])) and feature_point_matches_exists(outdir, dataset_name, 'outliers-{}'.format(options['lowes_threshold'])):
                _lowes_ratios_inliers = load_feature_point_matches(outdir, dataset_name, 'inliers-{}'.format(options['lowes_threshold']))
                _lowes_ratios_outliers = load_feature_point_matches(outdir, dataset_name, 'outliers-{}'.format(options['lowes_threshold']))
            else:
                data = dataset.DataSet(t)
                _lowes_ratios_inliers = []
                _lowes_ratios_outliers = []

                _fns, [_indices_1, _indices_2, _dists1, _dists2, _errors, _size1, _size2, _angle1, _angle2, _rerr1, \
                    _rerr2, _labels] = data.load_feature_matching_dataset(lowes_threshold=options['lowes_threshold'])

                for j, _ in enumerate(_fns):
                    if _labels[j] >= 1:
                        _lowes_ratios_inliers.append(max(_dists1[j], _dists2[j]))
                    else:
                        _lowes_ratios_outliers.append(max(_dists1[j], _dists2[j]))
                
                save_feature_point_matches(outdir, dataset_name, _lowes_ratios_inliers, 'inliers-{}'.format(options['lowes_threshold']))
                save_feature_point_matches(outdir, dataset_name, _lowes_ratios_outliers, 'outliers-{}'.format(options['lowes_threshold']))

            lowes_ratios_inliers.extend(_lowes_ratios_inliers)
            lowes_ratios_outliers.extend(_lowes_ratios_outliers)

        print ('#'*100)
        print ('#'*100)

        inliers_distribution = np.histogram(lowes_ratios_inliers, bins=nbins, range=(0.0,1.0))
        outliers_distribution = np.histogram(lowes_ratios_outliers, bins=nbins, range=(0.0,1.0))
        matching_classifiers.save_classifier(inliers_distribution, classifier_inliers_fn)
        matching_classifiers.save_classifier(outliers_distribution, classifier_outliers_fn)

    plot_baseline_histograms(outdir, inliers_distribution, outliers_distribution, nbins, options)
    # np.save(os.path.join(outdir, 'inliers_distribution_{}'.format(options['lowes_threshold'])), inliers_distribution)
    # np.save(os.path.join(outdir, 'outliers_distribution_{}'.format(options['lowes_threshold'])), outliers_distribution)
    print ('\tTime to train baseline classifier: {}'.format(np.round(timer() - s_time, 2)))

def baseline_histogram_classifier(arg):
    epsilon = 0.000000001

    dsets_te, fns_te, indices_1_te, indices_2_te, dists1_te, dists2_te, errors_te, size1_te, size2_te, angle1_te, angle2_te, rerr1_te,  \
        rerr2_te, labels_te, train, trained_classifier, options = arg

    # only rmatches and trained_classifier(which is the model) are significant here
    inliers_distribution, outliers_distribution = trained_classifier
    bins = outliers_distribution[1]
    max_distances_te = np.maximum(dists1_te, dists2_te)
    relevant_bins = np.digitize(max_distances_te, bins) - 1
    
    # import pdb; pdb.set_trace()

    inlier_percentage = (inliers_distribution[0][relevant_bins].astype(np.float) + epsilon) / (inliers_distribution[0][relevant_bins].astype(np.float) + outliers_distribution[0][relevant_bins].astype(np.float) + 1000.0*epsilon)
    return fns_te, indices_1_te, indices_2_te, dists1_te, dists2_te, None, inlier_percentage
    # return fns_te, max_distances_te, None, inlier_percentage, None, None

def parallelized_feature_point_classifier(arg):
    data, dset, im, im_unthresholded_matches, p_cached, options, trained_classifier = arg
    results = {}
    # classifier_args = 
    
    # fns, dsets, indices1, indices2, dists1, dists2, sizes1, sizes2, angles1, angles2 = [], [], [], [], [], [], [], [], [], []
    fns, dsets, indices1, indices2, dists1, dists2, sizes1, sizes2, angles1, angles2 = None, None, None, None, None, None, None, None, None, None
    num_matches = 0
    s_time_concatenation = timer()
    for im2 in sorted(im_unthresholded_matches.keys()):
        # import pdb; pdb.set_trace()
        # if im == '000365.jpg' and im2 == '000371.jpg':
        #     import pdb; pdb.set_trace()
        num_matches, _ = im_unthresholded_matches[im2].shape
        if dsets is None:
            # dsets.extend(np.tile(dset,(num_matches,1)).tolist())
            # fns.extend(np.tile([im, im2], (num_matches, 1)).tolist())
            # indices1.extend(im_unthresholded_matches[im2][:,0].tolist())
            # indices2.extend(im_unthresholded_matches[im2][:,1].tolist())
            # dists1.extend(im_unthresholded_matches[im2][:,2].tolist())
            # dists2.extend(im_unthresholded_matches[im2][:,3].tolist())

            # # raj: make sure size (and angle) is the correct index
            # sizes1.extend(p_cached[im][im_unthresholded_matches[im2][:,0].astype(np.int), 2])
            # sizes2.extend(p_cached[im2][im_unthresholded_matches[im2][:,1].astype(np.int), 2])

            # angles1.extend(p_cached[im][im_unthresholded_matches[im2][:,0].astype(np.int), 3])
            # angles2.extend(p_cached[im2][im_unthresholded_matches[im2][:,1].astype(np.int), 3])
            dsets = np.tile(dset,(num_matches,1))
            fns = np.tile([im, im2], (num_matches, 1))
            indices1 = im_unthresholded_matches[im2][:,0]
            indices2 = im_unthresholded_matches[im2][:,1]
            dists1 = im_unthresholded_matches[im2][:,2]
            dists2 = im_unthresholded_matches[im2][:,3]

            # raj: make sure size (and angle) is the correct index
            sizes1 = p_cached[im][im_unthresholded_matches[im2][:,0].astype(np.int), 2]
            sizes2 = p_cached[im2][im_unthresholded_matches[im2][:,1].astype(np.int), 2]

            angles1 = p_cached[im][im_unthresholded_matches[im2][:,0].astype(np.int), 3]
            angles2 = p_cached[im2][im_unthresholded_matches[im2][:,1].astype(np.int), 3]
        else:
            dsets = np.concatenate((dsets, np.tile(dset,(num_matches,1))))
            fns = np.concatenate((fns, np.tile([im, im2], (num_matches, 1))))
            indices1 = np.concatenate((indices1, im_unthresholded_matches[im2][:,0]))
            indices2 = np.concatenate((indices2, im_unthresholded_matches[im2][:,1]))
            dists1 = np.concatenate((dists1, im_unthresholded_matches[im2][:,2]))
            dists2 = np.concatenate((dists2, im_unthresholded_matches[im2][:,3]))

            # raj: make sure size (and angle) is the correct index
            sizes1 = np.concatenate((sizes1, p_cached[im][im_unthresholded_matches[im2][:,0].astype(np.int), 2]))
            sizes2 = np.concatenate((sizes2, p_cached[im2][im_unthresholded_matches[im2][:,1].astype(np.int), 2]))

            angles1 = np.concatenate((angles1, p_cached[im][im_unthresholded_matches[im2][:,0].astype(np.int), 3]))
            angles2 = np.concatenate((angles2, p_cached[im2][im_unthresholded_matches[im2][:,1].astype(np.int), 3]))

        # dsets.append(dset)
        # indices1.append()
    if options['debug']:
        print ('\t\t\tTime for concatenating inputs for {}: {} = {}'.format(dset, im, np.round(timer()-s_time_concatenation,2)))
    if num_matches == 0:
        return

    arg = [dsets, fns, indices1, indices2, dists1, dists2, np.zeros((len(fns),1)), sizes1, sizes2, angles1, angles2, np.zeros((len(fns),1)), np.zeros((len(fns),1)), \
        np.zeros((len(fns),1)), False, trained_classifier, options]

    s_time_classification = timer()
    if options['classifier'] == 'BASELINE':
        results_fns, results_indices1, results_indices2, results_dists1, results_dists2, _, scores = baseline_histogram_classifier(arg)
        # import pdb; pdb.set_trace()
    elif options['classifier'] == 'BDTS':
        results_fns, results_indices1, results_indices2, results_dists1, results_dists2, _, scores = classifier.classify_boosted_dts_feature_match(arg)
    if options['debug']:
        print ('\t\t\tTime for classifying results for {}: {} = {}'.format(dset, im, np.round(timer()-s_time_classification,2)))

    s_time_aggregation = timer()
    for i,(im1,im2) in enumerate(results_fns):
        # if im1 not in results:
        #     results[im1] = {}
        if im2 not in results:
            results[im2] = {}

        # if im1 not in results[im2]:
        #     results[im2][im1] = []
        # if im2 not in results[im1]:
        #     results[im1][im2] = []

        score = round(scores[i], 3)
        # results[im2].append({'im1': im1, 'im2': im2, 'index1': results_indices1[i], 'index2': results_indices2[i], 'score': score, 'dist1': results_dists1[i], 'dist2': results_dists2[i]})
        results[im2][results_indices1[i]] = {results_indices2[i]: {'im1': im1, 'im2': im2, 'index1': results_indices1[i], 'index2': results_indices2[i], 'score': score, 'dist1': results_dists1[i], 'dist2': results_dists2[i]} }
        # results[im2][im1].append({'im1': im2, 'im2': im1, 'index1': results_indices2[i], 'index2': results_indices1[i], 'score': score, 'dist1': results_dists2[i], 'dist2': results_dists1[i]})
    
    if options['debug']:
        print ('\t\t\tTime for aggregating results for {}: {} = {}'.format(dset, im, np.round(timer()-s_time_aggregation,2)))
    data.save_feature_matching_results(im, results, lowes_ratio_threshold=options['lowes_threshold'], classifier=options['classifier'])

def classify_feature_points(datasets, options={}):    
    print ('-'*100)
    print ('Classifier: {}'.format(options['classifier']))
    
    
    args = []
    outdir = options['outdir']

    # if options['classifier'] == 'BASELINE':
    #     # inliers_distribution = np.load(os.path.join(options['outdir'], 'inliers_distribution.npy'))
    #     # outliers_distribution = np.load(os.path.join(options['outdir'], 'outliers_distribution.npy'))
    #     inliers_distribution = matching_classifiers.load_classifier(classifier_inliers_fn)
    #     outliers_distribution = matching_classifiers.load_classifier(classifier_outliers_fn)
    #     trained_classifier = [inliers_distribution, outliers_distribution]
    # elif options['classifier'] == 'CONVNET':
    #     trained_classifier = matching_classifiers.load_classifier(os.path.join(options['outdir'], 'ETH3D'))

    print ('-'*100)
    print ('-'*100)
    if options['classifier'] == 'BASELINE':
        classifier_inliers_fn = os.path.join(outdir, 'inliers_distribution_{}'.format(options['lowes_threshold']))
        classifier_outliers_fn = os.path.join(outdir, 'outliers_distribution_{}'.format(options['lowes_threshold']))
        inliers_distribution = matching_classifiers.load_classifier(classifier_inliers_fn)
        outliers_distribution = matching_classifiers.load_classifier(classifier_outliers_fn)
        trained_classifier = [inliers_distribution, outliers_distribution]
    elif options['classifier'] == 'BDTS':
        trained_classifier = matching_classifiers.load_classifier(os.path.join(options['classifier_location'], 'ETH3D+TUM_RGBD_SLAM+TanksAndTemples+5-40.pkl'))

    for i,t in enumerate(datasets):
        print ('#'*100)
        print ('\tRunning classifier for dataset: {}'.format(os.path.basename(t)))
        data = dataset.DataSet(t)
        p_cached = {}
        s_time_preparation = timer()
        for im in sorted(data.all_feature_maps()):
            p, f, c = data.load_features(im)
            p_cached[im] = p

        for im in sorted(data.all_feature_maps()):
            # if im != '000365.jpg':
            #     continue
            if data.unthresholded_matches_exists(im):
                im_unthresholded_matches = data.load_unthresholded_matches(im)
                # print ('*'*100)
                # print (im)
                # print (len(im_unthresholded_matches))
                # print ('*'*100)
                args.append([data, t, im, im_unthresholded_matches, p_cached, options, trained_classifier])

        if options['debug']:
            print ('\t\tTime for preparing arguments for parallelization for {} = {}'.format(t, np.round(timer()-s_time_preparation,2)))
        p = Pool(options['processes'])
        if options['processes'] == 1:
            for arg in args:
                parallelized_feature_point_classifier(arg)
        else:
            p.map(parallelized_feature_point_classifier, args)
            p.close()

        print ("\tFinished classifying data for {} using {}".format(t.split('/')[-1], options['classifier']))  

def parallelized_feature_point_plotter(arg):
    data, options = arg
    fontsize = 12
    max_dists = []
    scores = []
    for ii, im in enumerate(sorted(data.all_feature_maps())):
        # for jj, im2 in enumerate(sorted(data.all_feature_maps())):
        #     if jj <= ii:
        #         continue
        im_results = data.load_feature_matching_results(image=im, lowes_ratio_threshold=options['lowes_threshold'], classifier=options['classifier'])
        for im2 in im_results.keys():
            
            for f in im_results[im2]:
                max_dists.append(max(f['dist1'], f['dist2']))
                scores.append(f['score'])
        # break

    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.xlabel("Squared Distances", fontsize=fontsize)
    plt.ylabel('Scores for {}'.format(options['classifier']), fontsize=fontsize)
    plt.title("Scores % vs Squared Distances (Lowe's Ratio: {})".format(options['lowes_threshold']))
    order = np.argsort(max_dists)

    # import pdb; pdb.set_trace()
    plt.plot(np.array(max_dists)[order], np.array(scores)[order])
    # print ('need to plot')
    # import sys; sys.exit(1)
    

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(options['outdir'], '{}-scores-vs-distances-{}.png'.format(options['classifier'], options['lowes_threshold'])))
    plt.clf()



def plot_feature_points_results(datasets, options):
    args = []
    for i,t in enumerate(datasets):
        print ('#'*100)
        print ('\tPlotting results for {} for dataset: {}'.format(options['classifier'], os.path.basename(t)))
        data = dataset.DataSet(t)
        args.append([data, options])

    p = Pool(options['processes'])
    if options['processes'] == 1:
        for arg in args:
            parallelized_feature_point_plotter(arg)
    else:
        p.map(parallelized_feature_point_plotter, args)
        p.close()


def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    # parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    # parser.add_argument('-x', '--image_match_classifier_max_match', help='')
    # parser.add_argument('--classifier_file', help='classifier checkpoint or saved classifier to use')
    parser.add_argument('--classifier', help='classifier type - BASELINE/CONVNET')
    parser.add_argument('--lowes_threshold', help="Lowe's threshold")
    # parser.add_argument('--convnet_checkpoint', help='checkpoint file for convnet')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset

    options = {
        'classifier': parser_options.classifier.upper(), \
        'outdir': os.path.join(os.getcwd(), 'data/feature-point-matching-baseline-classifiers'),
        'classifier_location': os.path.join(os.getcwd(), 'data/feature-matching-classifiers-results/5-40'),
        # 'image_match_classifier_file': parser_options.classifier_file, \
        # 'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        # 'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        # 'feature_selection': False,
        # 'classifier_file': parser_options.classifier_file,
        'processes': 1,
        'lowes_threshold': float(parser_options.lowes_threshold),
        'debug': False
    }

    training_datasets = [
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Barn',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Caterpillar',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Church',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courthouse',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ignatius',
    
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/facade',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/kicker',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/meadow',
    
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_360',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_room',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_hemisphere',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_coke',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk_with_person',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_dishes',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_no_loop',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_with_loop',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere2',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_360',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam2',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam3',
            
    ]

    val_datasets = [
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
        ]

    test_datasets = [
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

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor2_hall',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_ccw',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor3_loop_cw',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_stairs',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_floor5_wall',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/ece_stairs',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_all',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_atrium',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_backward',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_day_forward',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_all',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_atrium',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_backward',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/UIUCTag/yeh_night_forward',

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
    
    # create_baseline_classifier(training_datasets, options)
    # create_baseline_classifier(['/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_360'])
    
    
    # classify_feature_points(['/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard', 
    #     '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
    #     '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro'], options)
    classify_feature_points(val_datasets, options)

    # plot_feature_points_results(['/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard', 
    #     '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
    #     '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro'], options)
    
    # plot_feature_points_results(val_datasets, options)



if __name__ == '__main__':
    main(sys.argv)
