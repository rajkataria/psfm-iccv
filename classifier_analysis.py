import json
import math
import numpy as np
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import sys

from argparse import ArgumentParser
from matching_classifiers import load_classifier, mkdir_p

def save_analysis_data(data, fn):
    with open(fn, 'w') as fout:
        json.dump(data.tolist(), fout, sort_keys=True, indent=4, separators=(',', ': '))

def load_analysis_data(fn):
    with open(fn, 'r') as fin:
        data = json.load(fin)
    return np.array(data)

def classify_images(datasets, options={}):
    data_folder = 'data/image-matching-classifiers-analysis'
    mkdir_p(data_folder)
    for i,t in enumerate(datasets):
        data = dataset.DataSet(t)
        _fns, [_R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
            _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
            _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, _num_gt_inliers, _labels] \
            = data.load_image_matching_dataset(robust_matches_threshold=15, rmatches_min_threshold=options['image_match_classifier_min_match'], \
                rmatches_max_threshold=options['image_match_classifier_max_match'])

        fns_te, R11s_te, R12s_te, R13s_te, R21s_te, R22s_te, R23s_te, R31s_te, R32s_te, R33s_te, num_rmatches_te, num_matches_te, spatial_entropy_1_8x8_te, \
            spatial_entropy_2_8x8_te, spatial_entropy_1_16x16_te, spatial_entropy_2_16x16_te, pe_histogram_te, pe_polygon_area_percentage_te, \
            nbvs_im1_te, nbvs_im2_te, te_histogram_te, ch_im1_te, ch_im2_te, vt_rank_percentage_im1_im2_te, vt_rank_percentage_im2_im1_te, \
            num_gt_inliers_te, labels_te \
            = _fns, _R11s, _R12s, _R13s, _R21s, _R22s, _R23s, _R31s, _R32s, _R33s, _num_rmatches, _num_matches, _spatial_entropy_1_8x8, \
            _spatial_entropy_2_8x8, _spatial_entropy_1_16x16, _spatial_entropy_2_16x16, _pe_histogram, _pe_polygon_area_percentage, \
            _nbvs_im1, _nbvs_im2, _te_histogram, _ch_im1, _ch_im2, _vt_rank_percentage_im1_im2, _vt_rank_percentage_im2_im1, _num_gt_inliers, _labels

        labels_te[labels_te < 0] = 0

        trained_classifier = load_classifier(options['image_match_classifier_file'])
        arg = [ \
            fns_te, R11s_te, R12s_te, R13s_te, R21s_te, R22s_te, R23s_te, R31s_te, R32s_te, R33s_te, num_rmatches_te, num_matches_te, spatial_entropy_1_8x8_te, \
            spatial_entropy_2_8x8_te, spatial_entropy_1_16x16_te, spatial_entropy_2_16x16_te, pe_histogram_te, pe_polygon_area_percentage_te, \
            nbvs_im1_te, nbvs_im2_te, te_histogram_te, ch_im1_te, ch_im2_te, vt_rank_percentage_im1_im2_te, vt_rank_percentage_im2_im1_te, labels_te, \
            False, trained_classifier, options
        ]

        _, _, regr_bdt, scores, _ = classifier.classify_boosted_dts_image_match(arg)
        print ("\tFinished classifying data for {} using {}".format(t.split('/')[-1], options['classifier']))  

        analysis_data = np.concatenate(( \
            fns_te, \
            num_rmatches_te.reshape((len(labels_te),1)), \
            vt_rank_percentage_im1_im2_te.reshape((len(labels_te),1)), \
            vt_rank_percentage_im2_im1_te.reshape((len(labels_te),1)), \
            scores.reshape((len(labels_te),1)), \
            labels_te.reshape((len(labels_te),1))
        ), axis=1)

        fn = os.path.join(data_folder, t.split('/')[-1]) + '.json'
        save_analysis_data(analysis_data, fn)

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
        ranks = [1, 2, 5, 10, 15, 20, 30, 40, 50]
        for k in ranks:
            raw_results_rmatches, mean_results_rmatches = get_precision_recall(fns, labels, criteria=num_rmatches, k=k)
            raw_results_vt, mean_results_vt = get_precision_recall(fns, labels, criteria=vt_rank_mean, k=k)
            raw_results_scores, mean_results_scores = get_precision_recall(fns, labels, criteria=scores, k=k)
            raw_results_labels, mean_results_labels = get_precision_recall(fns, labels, criteria=labels, k=k)
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'rmatches', mean_results_rmatches[0], mean_results_rmatches[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'vt', mean_results_vt[0], mean_results_vt[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'scores', mean_results_scores[0], mean_results_scores[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, dataset_name, 'labels', mean_results_labels[0], mean_results_labels[1]))
            
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
    

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    parser.add_argument('-x', '--image_match_classifier_max_match', help='')
    parser.add_argument('-c', '--image_match_classifier_file', help='classifier to run')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset

    options = {
        'classifier': 'BDT', \
        'image_match_classifier_file': parser_options.image_match_classifier_file, \
        'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        'feature_selection': False
    }

    datasets = [
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Meetingroom',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/TanksAndTemples-ClassifierDatasets/Truck',
        
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0060',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0061',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/GTAV_540-ClassifierDatasets/0062',

        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/exhibition_hall',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/lecture_room',
        '/hdd/Research/psfm/pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/data/ETH3D-ClassifierDatasets/living_room',
    ]

    # classify_images(datasets, options)
    analyze_datasets(datasets, options)

if __name__ == '__main__':
    main(sys.argv)
