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
# from draw_matches import iterate_matches, iterate_gt_matches
import draw_matches

from multiprocessing import Pool
logger = logging.getLogger(__name__)

def mkdir_p(path):
  '''Make a directory including parent directories.
  '''
  try:
    os.makedirs(path)
  except os.error as exc:
    pass

def shortest_path_analysis(dset, options):
    colors = [(int(random.random()*255), int(random.random()*255), int(random.random()*255)) for i in xrange(0,30)]
    data = dataset.DataSet(dset)
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
        _num_gt_inliers, _labels] \
        = data.load_image_matching_dataset(robust_matches_threshold=options['image_matching_gt_threshold'], rmatches_min_threshold=0, \
            rmatches_max_threshold=100000, spl=100000)
    # ri = np.where( \
    #     (_labels >= 1.0) \
    #     )[0]
    ri = np.linspace(0,len(_fns)-1, len(_fns)).astype(np.int)
    graph_pruned = data.load_tracks_graph('tracks-pruned-matches.csv')

    # im = 'DSC_0827.JPG'
    # for track in graph_pruned[im]:
    #     if track == '8008':
    #         print graph_pruned[im][track]['feature']
    # import sys; sys.exit(1)

    for i, (im1, im2) in enumerate(_fns[ri]):
        if _shortest_path_length[ri[i]] <= options['shortest_path_length'] and _labels[ri[i]] >= 1.0 and _num_rmatches[ri[i]] > 20 and _num_rmatches[ri[i]] < 50:
            # print ('#'*100)
            im1_im2_rank = len(np.where(((_fns[:,0] == im1) | (_fns[:,1] == im1)) & (_num_rmatches > _num_rmatches[ri[i]]))[0])
            im2_im1_rank = len(np.where(((_fns[:,0] == im2) | (_fns[:,1] == im2)) & (_num_rmatches > _num_rmatches[ri[i]]))[0])
            # print ('{} - {}  :  {} / {}  SPL: {}  Rank: {}/{}'.format(im1, im2, _num_rmatches[ri[i]], _num_gt_inliers[ri[i]], _shortest_path_length[ri[i]], im1_im2_rank + 1, im2_im1_rank + 1))
            # print ('#'*100)
            print ('{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}'.format(im1, im2, _num_rmatches[ri[i]], _num_gt_inliers[ri[i]], _shortest_path_length[ri[i]], im1_im2_rank + 1, im2_im1_rank + 1))
        # draw_matches.iterate_gt_matches(data, colors, [im1], [im2], features=features)
            draw_matches.iterate_matches(data, colors, [im1], [im2], features=features)
    
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
    from opensfm import features, dataset, matching, classifier, reconstruction, types, io
    from opensfm.commands import formulate_graphs
    global features, matching, classifier, dataset, formulate_graphs

    # mkdir_p(os.path.join(parser_options.dataset,'shortest_path_analysis'))

    options = {
      'image_matching_gt_threshold': 20,
      'shortest_path_length': 2000000
    }
    
    shortest_path_analysis(parser_options.dataset, options)   
    
 
if __name__ == '__main__':
    main()