import cv2
import glob
import gzip
import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import random
import scipy
import sys
from argparse import ArgumentParser
from timeit import default_timer as timer

from multiprocessing import Pool
# from patchdataset import load_feature_matching_dataset, load_datasets
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

def mkdir_p(path):
  '''Make a directory including parent directories.
  '''
  try:
    os.makedirs(path)
  except os.error as exc:
    pass

class Context:
    pass

def debug_triplet_error(dset, options):
    data = dataset.DataSet(dset)
    ctx = Context()
    ctx.data = data
    processes = 1

    all_fns = []
    all_Rs = []
    transformations = {}
    im1_anchors = []
    im2_anchors = []
    ris = []

    if data.transformations_exists():
        transformations = data.load_transformations()
    else:
        transformations, t_num_pairs = classifier.calculate_transformations(ctx)

    for i,im1 in enumerate(transformations.keys()):
        for j,im2 in enumerate(transformations[im1].keys()):
            if im1 in options['images'] or im2 in options['images']:
                R = transformations[im1][im2]['rotation']
                all_fns.append(np.array([im1, im2]))
                all_Rs.append(np.array(R).reshape((1,-1)))
                if im1 not in options['images']:
                    im1_anchors.append(im1)
                if im2 not in options['images']:
                    im2_anchors.append(im2)

    common_images = list(set(im1_anchors).intersection(set(im2_anchors)))
    if options['filter']:
        common_images = list(set(common_images).intersection(set([options['filter']])))

    for i, (im1, im2) in enumerate(all_fns):
        if im1 in common_images or im2 in common_images or im1 in options['images'] and im2 in options['images']:
            ris.append(i)

    fns = np.array(all_fns)[np.array(ris).astype(int)]
    Rs = np.array(all_Rs)[np.array(ris).astype(int)]
    args = classifier.triplet_arguments(fns, Rs)
    triplet_results = {}
    
    p = Pool(processes)
    if processes > 1:
        t_results = p.map(classifier.calculate_rotation_triplet_errors, args)
    else:
        t_results = []
        for arg in args:
            t_results.append(classifier.calculate_rotation_triplet_errors(arg))
    for r in t_results:
        fn1, triplets = r
        triplet_results[fn1] = triplets

    t_fns1, t_fns2, t_fns3, t_errors = classifier.flatten_triplets(triplet_results)
    t_fns = np.concatenate((t_fns1.reshape(-1,1), t_fns2.reshape(-1,1), t_fns3.reshape(-1,1)), axis=1)
    args = classifier.triplet_pairwise_arguments(np.array(fns), t_fns, t_errors, processes)
    p = Pool(processes)
    p_results = []
    if processes == 1:
        for arg in args:
            p_results.append(classifier.calculate_triplet_pairwise_errors(arg))
    else:
        p_results = p.map(classifier.calculate_triplet_pairwise_errors, args)
    p.close()
    triplet_pairwise_results = {}
    for i, r in enumerate(p_results):
        histograms, histograms_list = r
        for k in histograms:
            if k not in triplet_pairwise_results:
                triplet_pairwise_results[k] = histograms[k]
            else:
                triplet_pairwise_results[k].update(histograms[k])
    

    im1, im2 = options['images']
    if im1 > im2:
        im1, im2 = im2, im1

    print json.dumps(triplet_pairwise_results[im1][im2], sort_keys=True, indent=4, separators=(',', ': '))

def plot_consistency_errors(data, im1, im2):
    cutoffs = [2, 3, 4]
    edge_threshold = 30
    offset = 0
    plt.clf()
    visualization_gt_fn = os.path.join(data.data_path, 'match_visualizations_gt/{}---{}.jpeg'.format(im1.split('.')[0], im2.split('.')[0]))
    visualization_fn = os.path.join(data.data_path, 'match_visualizations/{}---{}.jpeg'.format(im1.split('.')[0], im2.split('.')[0]))
    if os.path.exists(visualization_fn):
        img_gt = cv2.cvtColor(cv2.imread(visualization_gt_fn), cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(cv2.imread(visualization_fn), cv2.COLOR_BGR2RGB)
        plt.subplot(len(cutoffs) + 1, 2, 1)
        # plt.imshow(img[:,:(img.shape[1]/2),:])
        plt.imshow(img_gt)
        plt.title ('Ground-truth visualization')
        plt.subplot(len(cutoffs) + 1, 2, 2)
        # plt.imshow(img[:,(img.shape[1]/2 - 1):,:])
        plt.imshow(img)
        plt.title ('Robust matches visualization')
        offset = 1
    for c, cutoff in enumerate(cutoffs):
        print ('{} : {}    cutoff: {}'.format(im1, im2, cutoff))
        consistency_errors = data.load_consistency_errors(cutoff=cutoff, edge_threshold=edge_threshold)

        histogram = np.array(consistency_errors[im1][im2]['histogram-counts'])
        bins = np.array(consistency_errors[im1][im2]['bins']) * 180.0 / np.pi
        width = 0.9 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2

        errors = []
        for i in range(0,len(histogram)):
            v = [(bins[i] + bins[i+1])/2.0] * histogram[i]
            errors.extend(v)
        if len(errors) == 0:
            print ('\tCutoff: {}  Edge threshold: {} - No histogram counts'.format(cutoff, edge_threshold))
            continue
        # mu, sigma = scipy.stats.norm.fit(errors)

        # mu, sigma = scipy.stats.norm.fit(errors)
        # mu = np.mean(errors)
        # sigma = 0.0

        plt.subplot(len(cutoffs) + offset, 2, 2*c + 1 + 2*offset)
        plt.cla()
        plt.bar(center, histogram, align='center', width=width)
        plt.xlabel('R11 angle error', fontsize=16)
        plt.ylabel('Error count', fontsize=16)
        plt.title('Graph edge threshold: {} Cycle length: {}'.format(edge_threshold, cutoff))

        histogram_cumsum = np.array(consistency_errors[im1][im2]['histogram-cumsum'])
        plt.subplot(len(cutoffs) + offset, 2, 2*c + 2 + 2*offset)
        plt.cla()
        plt.bar(center, histogram_cumsum, align='center', width=width)
        plt.xlabel('R11 angle error', fontsize=16)
        plt.ylabel('Error count', fontsize=16)
        plt.title('Graph edge threshold: {} Cycle length: {}'.format(edge_threshold, cutoff))

    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig('consistency-error-{}-{}.png'.format(os.path.basename(im1), os.path.basename(im2)))

    
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
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    from opensfm.commands import formulate_graphs
    global matching, classifier, dataset, formulate_graphs

    mkdir_p(os.path.join(parser_options.dataset,'triplet_debugging'))

    options = {
      'images': ['DSC_1140.JPG','DSC_1157.JPG'],
      # 'filter': 'DSC_1155.JPG'
      'filter': None
    }
    
    # debug_triplet_error(parser_options.dataset, options)


    data = dataset.DataSet(parser_options.dataset)
    images = data.images()

    exifs = {im: data.load_exif(im) for im in images}
    ctx = Context()
    ctx.data = data
    ctx.cameras = ctx.data.load_camera_models()
    ctx.exifs = exifs


    # im1s = ['DSC_1744.JPG', 'DSC_1761.JPG', 'DSC_1744.JPG', 'DSC_1761.JPG', \
    #     'DSC_1745.JPG', 'DSC_1800.JPG', 'DSC_1800.JPG', 'DSC_1770.JPG', \
    #     'DSC_1770.JPG', 'DSC_1770.JPG', 'DSC_1755.JPG']
    # im2s = ['DSC_1760.JPG', 'DSC_1780.JPG', 'DSC_1780.JPG', 'DSC_1762.JPG', \
    #     'DSC_1802.JPG', 'DSC_1807.JPG', 'DSC_1812.JPG', 'DSC_1773.JPG', \
    #     'DSC_1812.JPG', 'DSC_1794.JPG', 'DSC_1795.JPG']

    im1s = ['DSC_1744.JPG']
    im2s = ['DSC_1800.JPG']

    for i, _ in enumerate(im1s):
        plot_consistency_errors (data, im1s[i], im2s[i])
    
 
if __name__ == '__main__':
    main()