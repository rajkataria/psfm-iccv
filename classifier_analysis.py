import json
import numpy as np
import os
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

        analysis_data = np.concatenate((fns_te, num_rmatches_te.reshape((len(labels_te),1)), scores.reshape((len(labels_te),1)), labels_te.reshape((len(labels_te),1))), axis=1)
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

def analyze_datasets(datasets, options={}):
    data_folder = 'data/image-matching-classifiers-analysis'
    for i,t in enumerate(datasets):
        data_fn = os.path.join(data_folder, t.split('/')[-1]) + '.json'
        analysis_data = load_analysis_data(data_fn)
        print ('#'*100)
        print ('Dataset: {}  Examples: {}'.format(t.split('/')[-1], len(analysis_data)))
        
        fns = analysis_data[:,0:2]
        num_rmatches = analysis_data[:,2].astype(np.float)
        scores = analysis_data[:,3].astype(np.float)
        labels = analysis_data[:,4].astype(np.float)

        for k in [5, 10, 30]:
            _, mean_results_rmatches = get_precision_recall(fns, labels, criteria=num_rmatches, k=k)
            _, mean_results_scores = get_precision_recall(fns, labels, criteria=scores, k=k)
            _, mean_results_labels = get_precision_recall(fns, labels, criteria=labels, k=k)
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, t.split('/')[-1], 'rmatches', mean_results_rmatches[0], mean_results_rmatches[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, t.split('/')[-1], 'scores', mean_results_scores[0], mean_results_scores[1]))
            print ('\tTop {} - Dataset: {}  Criteria: {}  Mean Precision: {}  Mean Recall: {}'.format(k, t.split('/')[-1], 'labels', mean_results_labels[0], mean_results_labels[1]))

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

    classify_images(datasets, options)
    analyze_datasets(datasets, options)

if __name__ == '__main__':
    main(sys.argv)
