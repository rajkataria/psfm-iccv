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
from multiprocessing import Pool
from scipy import special
import convnet

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def get_tracks_graph_stats(graph, recon):
    inlier_tracks = 0
    track_lengths = []
    # debug_track = '9259'
    
    tracks, images = matching.tracks_and_images(graph)
    for t in tracks:
        # if t == debug_track:
        #     print ('\tProcessing track: {}'.format(debug_track))
        if t in recon.points:
            inlier_tracks += 1
            track_lengths.append(len(graph[t].keys()))
        #     if t == debug_track:
        #         print ('\tProcessing track: {}  Status: Triangulated'.format(debug_track))
        # else:
        #     if t == debug_track:
        #         print ('\tProcessing track: {}  Status: NOT triangulated'.format(debug_track))

    return {'inlier_tracks': inlier_tracks, 'mean_track_length': np.mean(np.array(track_lengths)), 'track_lengths': track_lengths}

def triangulate_gt_reconstruction(data, graph):
    recon_orig = data.load_reconstruction('reconstruction_gt.json')[0]
    _ = reconstruction.retriangulate(graph, recon_orig, data.config)
    reconstruction.paint_reconstruction(data, graph, recon_orig)
    data.save_reconstruction([recon_orig], filename='test-reconstruction-original-triangulation.json')
    graph_stats = get_tracks_graph_stats(graph, recon_orig)
    print ('\tOriginal triangulation: Points: {}'.format(len(recon_orig.points.keys())))
    print ('\tTriangulated tracks: Number: {}   Average track length: {}'.format(graph_stats['inlier_tracks'], graph_stats['mean_track_length']))

    recon_robust = data.load_reconstruction('reconstruction_gt.json')[0]
    _, robust_graph = reconstruction.robustly_retriangulate(graph, recon_robust, data.config)
    # reconstruction.paint_reconstruction(data, graph, recon_robust)
    reconstruction.paint_reconstruction(data, robust_graph, recon_robust)
    data.save_reconstruction([recon_robust], filename='test-reconstruction-robust-triangulation.json')
    robust_graph_stats = get_tracks_graph_stats(robust_graph, recon_robust)
    
    print ('\tRobust triangulation: Points: {}'.format(len(recon_robust.points.keys())))
    print ('\tTriangulated tracks: Number: {}   Average track length: {}'.format(robust_graph_stats['inlier_tracks'], robust_graph_stats['mean_track_length']))
    print ('-'*100)
    print ('\tPoints not triangulated: {}'.format(len(list(set(recon_orig.points.keys()) - set(recon_robust.points.keys())))))

def test_robust_triangulation(dset, options={}):
    data_folder = 'data/test_robust_triangulation'
    mkdir_p(data_folder)

    data = dataset.DataSet(dset)
    graph = data.load_tracks_graph('tracks.csv')
    recon_gt_triangulated = triangulate_gt_reconstruction(data, graph)

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    parser.add_argument('--dataset', help='dataset')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)

    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global dataset, matching, reconstruction

    options = {}
    test_robust_triangulation(parser_options.dataset, options)

if __name__ == '__main__':
    main(sys.argv)
