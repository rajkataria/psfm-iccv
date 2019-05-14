import warnings
warnings.filterwarnings('ignore')

import json
import logging
import math
import numpy as np
import os
import pyquaternion
import sys
from timeit import default_timer as timer

from networkx.algorithms import bipartite

from opensfm import classifier
from opensfm import dataset
from opensfm import evaluate_ate_scale, associate
from opensfm import io
from opensfm import matching
from opensfm import types
from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.patches import Ellipse

logger = logging.getLogger(__name__)

class Command:
    name = 'validate_results'
    help = "Report results"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        options = {
            'robust_matches_threshold': 15
        }

        data = dataset.DataSet(args.dataset)

        reconstruction_results = get_reconstruction_results(data)
        data.save_reconstruction_results(reconstruction_results)

        if not data.reconstruction_exists('reconstruction_gt.json'):
            logger.info('Skipping ground-truth calculations since no ground-truth exists...')
        else:
            ate_results, rpe_results = get_gt_results(data, options)
            data.save_ate_results(ate_results)
            data.save_rpe_results(rpe_results)

def get_reconstruction_results(data):
    reconstruction_results = {}
    relevant_reconstructions = []
    options = {
        'min_triangulated': 5,
        'debug': False
    }
    baseline_command_keys = [
        'focal_from_exif',
        'detect_features',
        'evaluate_vt_rankings',
        'match_features',
        'create_tracks',
        'reconstruct'
    ]
    classifier_command_keys = [
        'focal_from_exif',
        'detect_features',
        'evaluate_vt_rankings',
        'match_features',
        'calculate_features',
        'classify_images',
        'create_tracks_classifier',
        'reconstruct'
    ]

    if data.tracks_graph_exists('tracks.csv'):
        tracks_graph = data.load_tracks_graph('tracks.csv')    
    if data.tracks_graph_exists('tracks-pruned-matches.csv'):
        tracks_graph_pruned = data.load_tracks_graph('tracks-pruned-matches.csv')
    if data.tracks_graph_exists('tracks-all-matches.csv'):
        tracks_graph_all = data.load_tracks_graph('tracks-all-matches.csv')
    if data.tracks_graph_exists('tracks-thresholded-matches.csv'):
        tracks_graph_thresholded = data.load_tracks_graph('tracks-thresholded-matches.csv')
    if data.tracks_graph_exists('tracks-pruned-thresholded-matches.csv'):
        tracks_graph_pruned_thresholded = data.load_tracks_graph('tracks-pruned-thresholded-matches.csv')
    if data.tracks_graph_exists('tracks-all-weighted-matches.csv'):
        tracks_graph_all_weighted = data.load_tracks_graph('tracks-all-weighted-matches.csv')
    if data.tracks_graph_exists('tracks-thresholded-weighted-matches.csv'):
        tracks_graph_thresholded_weighted = data.load_tracks_graph('tracks-thresholded-weighted-matches.csv')
    if data.tracks_graph_exists('tracks-pruned-thresholded-weighted-matches.csv'):
        tracks_graph_pruned_thresholded_weighted = data.load_tracks_graph('tracks-pruned-thresholded-weighted-matches.csv')
    if data.tracks_graph_exists('tracks-gt-matches.csv'):
        tracks_graph_gt = data.load_tracks_graph('tracks-gt-matches.csv')
    if data.tracks_graph_exists('tracks-gt-matches-pruned.csv'):
        tracks_graph_gt_pruned = data.load_tracks_graph('tracks-gt-matches-pruned.csv')

    # Get results for baselines
    if data.reconstruction_exists('reconstruction.json'):
        logger.info('Computing reconstruction results for baseline...')
        stats_label = 'baseline'
        reconstruction_baseline = data.load_reconstruction('reconstruction.json')[0]
        relevant_reconstructions.append([tracks_graph, reconstruction_baseline, baseline_command_keys, stats_label])

    if data.reconstruction_exists('reconstruction_colmap.json'):
        logger.info('Computing reconstruction results for colmap...')
        stats_label = 'colmap'
        reconstruction_colmap = data.load_reconstruction('reconstruction_colmap.json')[0]
        relevant_reconstructions.append([tracks_graph, reconstruction_colmap, {}, stats_label])

    for imc in [True, False]:
        for wr in [True, False]:
            for colmapr in [True, False]:
                for gm in [True, False]:
                    for gsm in [True, False]:
                        for wfm in [True, False]:
                            for imt in [True, False]:
                                for imtv in [0.2, 0.3, 0.4, 0.5]:
                                    for spp in [True, False]:
                                        for cip in [False]:
                                            stats_label = 'imc-{}-wr-{}-colmapr-{}-gm-{}-gsm-{}-wfm-{}-imt-{}-imtv-{}-spp-{}-cip-{}-cipgt-False-cipk-H'.format(imc, wr, colmapr, gm, gsm, wfm, imt, imtv, spp, cip)
                                            reconstruction_fn = 'reconstruction-{}.json'.format(stats_label)
                                            if data.reconstruction_exists(reconstruction_fn):
                                                logger.info('Computing reconstruction results - {}'.format(reconstruction_fn))
                                                reconstruction_ = data.load_reconstruction(reconstruction_fn)[0]

                                                if gm is True and spp is True:
                                                    graph = tracks_graph_gt_pruned
                                                elif gm is True and spp is False:
                                                    graph = tracks_graph_gt
                                                elif imc is False and wfm is False and imt is False and spp is False:
                                                    graph = tracks_graph
                                                elif imc is False and wfm is False and imt is False and spp is True:
                                                    graph = tracks_graph_pruned
                                                elif imc is True and wfm is False and imt is True and spp is False:
                                                    graph = tracks_graph_thresholded
                                                elif imc is True and wfm is False and imt is True and spp is True:
                                                    graph = tracks_graph_pruned_thresholded


                                                relevant_reconstructions.append([graph, reconstruction_, classifier_command_keys, stats_label])

    for datum in relevant_reconstructions:
        g, r, k, l = datum
        reconstruction_results[l] = calculate_reconstruction_results(data, g, r, options, k)

    return reconstruction_results

def get_gt_results(data, options):
    relevant_reconstructions = []

    if data.image_matching_dataset_exists(options['robust_matches_threshold']):
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
            = data.load_image_matching_dataset(robust_matches_threshold=options['robust_matches_threshold'])
        image_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])

        fns = []
        y_gt = []
        y = []
        for im1 in image_matching_results:
            for im2 in image_matching_results[im1]:
                fns.append([im1,im2])
                y.append(image_matching_results[im1][im2]['score'])
                ri = np.where((_fns[:,0] == im1) & (_fns[:,1] == im2) | (_fns[:,1] == im1) & (_fns[:,0] == im2))
                y_gt.append(_labels[ri])

        auc, _ = classifier.calculate_dataset_auc(np.array(y), np.array(y_gt), debug=False)
        _, _, f_auc, _ = classifier.calculate_per_image_mean_auc(np.array(fns), np.array(y), np.array(y_gt), debug=False)

        baseline_auc, _ = classifier.calculate_dataset_auc(np.array(y), np.array(y_gt), debug=False)
        _, _, baseline_f_auc, _ = classifier.calculate_per_image_mean_auc(np.array(fns), np.array(y), np.array(y_gt), debug=False)

    else:
        baseline_auc, baseline_f_auc, auc, f_auc = None, None, None, None

    # Get results for baselines
    if data.reconstruction_exists('reconstruction.json'):
        logger.info('Computing ground-truth evaluation results for baseline...')
        stats_label = 'baseline'
        reconstruction_baseline_gt = data.load_reconstruction('reconstruction_gt.json')[0]
        reconstruction_baseline = data.load_reconstruction('reconstruction.json')[0]
        intersect_reconstructions(data, reconstruction_baseline_gt, reconstruction_baseline)
        relevant_reconstructions.append([reconstruction_baseline_gt, reconstruction_baseline, stats_label])

    if data.reconstruction_exists('reconstruction_colmap.json'):
        logger.info('Computing ground-truth evaluation results for colmap reconstruction...')
        stats_label = 'colmap'
        reconstruction_colmap_gt = data.load_reconstruction('reconstruction_gt.json')[0]
        reconstruction_colmap = data.load_reconstruction('reconstruction_colmap.json')[0]
        intersect_reconstructions(data, reconstruction_colmap_gt, reconstruction_colmap)
        relevant_reconstructions.append([reconstruction_colmap_gt, reconstruction_colmap, stats_label])

    for imc in [True, False]:
        for wr in [True, False]:
            for colmapr in [True, False]:
                for gm in [True, False]:
                    for gsm in [True, False]:
                        for wfm in [True, False]:
                            for imt in [True, False]:
                                for imtv in [0.2, 0.3, 0.4, 0.5]:
                                    for spp in [True, False]:
                                        for cip in [False]:
                                            stats_label = 'imc-{}-wr-{}-colmapr-{}-gm-{}-gsm-{}-wfm-{}-imt-{}-imtv-{}-spp-{}-cip-{}-cipgt-False-cipk-H'.format(imc, wr, colmapr, gm, gsm, wfm, imt, imtv, spp, cip)
                                            reconstruction_fn = 'reconstruction-{}.json'.format(stats_label)
                                            if data.reconstruction_exists(reconstruction_fn):
                                                logger.info('Computing ground-truth evaluation results - {}'.format(reconstruction_fn))
                                                reconstruction_ = data.load_reconstruction(reconstruction_fn)[0]

                                                reconstruction_gt = data.load_reconstruction('reconstruction_gt.json')[0]
                                                reconstruction_ = data.load_reconstruction(reconstruction_fn)[0]
                                                intersect_reconstructions(data, reconstruction_gt, reconstruction_)

                                                relevant_reconstructions.append([reconstruction_gt, reconstruction_, stats_label])

    for datum in relevant_reconstructions:
        r_gt, r, label = datum
        camera = types.PerspectiveCamera()
        camera_gt = types.PerspectiveCamera()
        r.add_camera(camera)
        r_gt.add_camera(camera_gt)
     
    ate_results = ransac_based_ate_evaluation(data, relevant_reconstructions)
    rpe_results = rpe_evaluation(data, relevant_reconstructions)

    auc_results = {
        'Baseline AUC': baseline_auc,
        'Baseline AUCPI': baseline_f_auc,
        'Experiment AUC': auc,
        'Experiment AUCPI': f_auc
    }
    for k in ate_results.keys():
        ate_results[k].update(auc_results)
    for k in rpe_results.keys():
        rpe_results[k].update(auc_results)

    return ate_results, rpe_results

def prune_reconstructions(data, reconstruction):
    # Only really applies to TUM datasets, but we can blindly run it on all
    images = data.images()
    images.pop()
    
    for r in reconstruction:
        additional_shots = []
        for s in r.shots:
            if s not in images:
                additional_shots.append(s)

        for s in additional_shots:
            del r.shots[s]

def intersect_reconstructions(data, reconstruction_gt, reconstruction):
    images_gt = []
    images = []

    images_gt = set(reconstruction_gt.shots.keys())
    images = set(reconstruction.shots.keys())
    
    common_images = images.intersection(images_gt)

    remove_shots = []
    for s in reconstruction.shots:
        if s in list(images - common_images):
            remove_shots.append(s)
    for s in remove_shots:
        del reconstruction.shots[s]

    remove_shots = []
    for s in reconstruction_gt.shots:
        if s in list(images_gt - common_images):
            remove_shots.append(s)
    for s in remove_shots:
        del reconstruction_gt.shots[s]

def calculate_reconstruction_results(data, graph, reconstruction, options, command_keys):
    registered = 0
    error = 0.0
    count = 0
    missing_errors = 0
    total_time = 0.0
    cameras = 0
    times = {}

    tracks, images = matching.tracks_and_images(graph)
    cameras = len(data.images())
    for s in reconstruction.shots:
        if s not in graph:
            continue
        pts_triangulated = set(reconstruction.points.keys()).intersection(set(graph[s].keys()))
        if len(pts_triangulated) >= options['min_triangulated']:
            registered += 1

    for pid in reconstruction.points:
        if reconstruction.points[pid].reprojection_error:
            error = error + reconstruction.points[pid].reprojection_error
            count = count + 1
        else:
            missing_errors = missing_errors + 1
    
    profile_fn = os.path.join(data.data_path, 'profile.log')
    if os.path.exists(profile_fn):
        with open(profile_fn, 'r') as f:
            for line in f.readlines():
                datums = line.split(':')
                if datums[0] in command_keys:
                    times[datums[0]] = float(datums[1])

        for key in times:
            total_time = total_time + times[key]


    results = {
        'dataset': os.path.basename(os.path.normpath(data.data_path)),
        'registered images': registered,
        'total images in dataset': cameras,
        'points triangulated ': len(reconstruction.points.keys()),
        'average reprojection error': 1.0*error/count,
        'points with reprojection error': count,
        'missing reprojection error': missing_errors,
        'time': round(total_time, 2)
    }
    return results

def get_sample_matches(matches):
    samples = np.random.choice(len(matches), 2, replace=False)
    return np.array(matches)[samples].tolist()

def plot_best_trajectories(data, gt_full_list, osfm_full_list, full_matches, s, R, t, comparison_label):
    gt_stamps = gt_full_list.keys()
    gt_stamps.sort()
    gt_xyz_full = np.matrix([[float(value) for value in gt_full_list[b][0:3]] for b in gt_stamps]).transpose()
    
    osfm_stamps = osfm_full_list.keys()
    osfm_stamps.sort()
    osfm_xyz_full = np.matrix([[float(value) for value in osfm_full_list[b][0:3]] for b in osfm_stamps]).transpose()
    osfm_xyz_full_aligned = s * R * osfm_xyz_full + t

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.add_subplot(111)
    evaluate_ate_scale.plot_traj(ax,gt_stamps,gt_xyz_full.transpose().A,'-','green','ground truth')
    evaluate_ate_scale.plot_traj(ax,osfm_stamps,osfm_xyz_full_aligned.transpose().A,'-','blue','estimated')

    label='difference'
    for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(full_matches,gt_xyz_full.transpose().A,osfm_xyz_full_aligned.transpose().A):
        ax.plot([x1,x2],[y1,y2],'--',color='red',label=label)
        label=''
        
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(os.path.join(data.data_path, 'results', 'ate-{}.png'.format(comparison_label)),dpi=90)

def ransac_based_ate_evaluation(data, relevant_reconstructions):
    ate_options = {
        'i_scale': 1.0,
        'offset': 0.0,
        'max_difference': 0.02
    }
    
    total_ransac_iterations = 20000
    results = {}
    for r_gt, r_osfm, label in relevant_reconstructions:
        
        data.save_tum_format([r_gt], suffix='full-gt')
        data.save_tum_format([r_osfm], suffix='full-osfm')
        
        gt_full_list = associate.read_file_list(os.path.join(data.data_path, 'results', 'reconstruction-0-full-gt.txt'))
        osfm_full_list = associate.read_file_list(os.path.join(data.data_path, 'results', 'reconstruction-0-full-osfm.txt'))
        full_matches = associate.associate(gt_full_list, osfm_full_list, ate_options['offset'], ate_options['max_difference'])
        gt_xyz_full = np.matrix([[float(value) for value in gt_full_list[a][0:3]] for a,b in full_matches]).transpose()
        osfm_xyz_full = np.matrix([[float(value) for value in osfm_full_list[b][0:3]] for a,b in full_matches]).transpose()

        max_score_cm = -sys.float_info.max
        max_score_dm = -sys.float_info.max
        max_score_m = -sys.float_info.max
        for ransac_iteration in range(0, total_ransac_iterations):
            images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error, s, R, t = align_and_calculate_ate(data, gt_full_list, osfm_full_list, \
                gt_xyz_full, osfm_xyz_full, full_matches, label, ate_options)

            if max_score_cm < images_aligned_cm:
                max_score_cm = images_aligned_cm
                best_model_cm = {
                    's':s, 'R': R, 't': t, 'matches': full_matches, \
                    'images_aligned_cm': images_aligned_cm, 'images_aligned_dm': images_aligned_dm, 'images_aligned_m': images_aligned_m,\
                    'translation_error': translation_error
                    }
            if max_score_dm < images_aligned_dm:
                max_score_dm = images_aligned_dm
                best_model_dm = {
                    's':s, 'R': R, 't': t, 'matches': full_matches, \
                    'images_aligned_cm': images_aligned_cm, 'images_aligned_dm': images_aligned_dm, 'images_aligned_m': images_aligned_m,\
                    'translation_error': translation_error
                    }
            if max_score_m < images_aligned_m:
                max_score_m = images_aligned_m
                best_model_m = {
                    's':s, 'R': R, 't': t, 'matches': full_matches, \
                    'images_aligned_cm': images_aligned_cm, 'images_aligned_dm': images_aligned_dm, 'images_aligned_m': images_aligned_m,\
                    'translation_error': translation_error
                    }
        
        if len(best_model_cm['matches']) >= 2:
            plot_best_trajectories(data, gt_full_list, osfm_full_list, best_model_cm['matches'], \
                best_model_cm['s'], best_model_cm['R'], best_model_cm['t'], label)

        results[label] = {
            'absolute translational error %f m': best_model_cm['translation_error'],
            'images aligned %f cm': best_model_cm['images_aligned_cm'],
            'best model s cm': best_model_cm['s'],
            'best model R cm': best_model_cm['R'].tolist(),
            'best model t cm': best_model_cm['t'].reshape((3,)).tolist(),
            'best model s dm': best_model_dm['s'],
            'best model R dm': best_model_dm['R'].tolist(),
            'best model t dm': best_model_dm['t'].reshape((3,)).tolist(),
            'best model s m': best_model_m['s'],
            'best model R m': best_model_m['R'].tolist(),
            'best model t m': best_model_m['t'].reshape((3,)).tolist(),
            'images aligned %f dm': best_model_dm['images_aligned_dm'],
            'images aligned %f m': best_model_m['images_aligned_m'],
            'trajectory': os.path.basename(os.path.normpath(data.data_path)),
            'total images %f': len(gt_full_list)
        }
        
    return results

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    """
    return np.dot(np.linalg.inv(a),b)

def scale(a,scalar):
    """
    Scale the translational components of a 4x4 homogeneous matrix by a scale factor.
    """
    return np.array(
        [[a[0,0], a[0,1], a[0,2], a[0,3]*scalar],
         [a[1,0], a[1,1], a[1,2], a[1,3]*scalar],
         [a[2,0], a[2,1], a[2,2], a[2,3]*scalar],
         [a[3,0], a[3,1], a[3,2], a[3,3]]]
                       )

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3,3])

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos( min(1,max(-1, (np.trace(transform[0:3,0:3]) - 1)/2) ))

def distances_along_trajectory(traj):
    """
    Compute the translational distances along a trajectory. 
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i+1]],traj[keys[i]]) for i in range(len(keys)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
        distances.append(sum)
    return distances
    
def rotations_along_trajectory(traj,scale):
    """
    Compute the angular rotations along a trajectory. 
    """
    keys = traj.keys()
    keys.sort()
    motion = [ominus(traj[keys[i+1]],traj[keys[i]]) for i in range(len(keys)-1)]
    distances = [0]
    sum = 0
    for t in motion:
        sum += compute_angle(t)*scale
        distances.append(sum)
    return distances
    
def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    _EPS = np.finfo(float).eps * 4.0
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def rpe_evaluation(data, relevant_reconstructions):
    rpe_options = {
        'max_difference': 0.02,
        'max_pairs': 10000,
        'fixed_delta': False,
        'delta': 1.00,
        'delta_unit': 's',
        'offset': 0.0,
        'i_scale': 1.0,
    }

    results = {}
    for r_gt, r_osfm, label in relevant_reconstructions:
        rotational_errors = []
        data.save_tum_format([r_gt], suffix='full-gt')
        data.save_tum_format([r_osfm], suffix='full-osfm')
        
        gt_full_list = associate.read_file_list(os.path.join(data.data_path, 'results', 'reconstruction-0-full-gt.txt'))
        osfm_full_list = associate.read_file_list(os.path.join(data.data_path, 'results', 'reconstruction-0-full-osfm.txt'))
        full_matches = associate.associate(gt_full_list, osfm_full_list, rpe_options['offset'], rpe_options['max_difference'])

        pairs = []
        for i,im1 in enumerate(sorted(r_osfm.shots.keys())):
            for j,im2 in enumerate(sorted(r_osfm.shots.keys())):
                if j <= i:
                    continue
                pairs.append([(i, im1), (j, im2)])

        result = []
        total_pairs = 0
        rotational_errors_distributions = {
            'error < 0.5 deg': 0,
            'error < 1.0 deg': 0,
            'error < 2.0 deg': 0,
            'error < 5.0 deg': 0,
            'error < 10.0 deg': 0,
            'error >= 10.0 deg': 0
        }
        
        for im1_tuple, im2_tuple in pairs:
            i, im1 = im1_tuple
            j, im2 = im2_tuple

            osfm_tum_pose_i = [0] + [float(a) for a in osfm_full_list[i]]
            osfm_tum_pose_j = [0] + [float(a) for a in osfm_full_list[j]]
            gt_tum_pose_i = [0] + [float(a) for a in gt_full_list[i]]
            gt_tum_pose_j = [0] + [float(a) for a in gt_full_list[j]]

            error44 = ominus(  scale(
                               ominus( transform44(osfm_tum_pose_j), transform44(osfm_tum_pose_i) ), rpe_options['i_scale']),
                               ominus( transform44(gt_tum_pose_j), transform44(gt_tum_pose_i) ) )
            
            trans_error = compute_distance(error44)
            rot_error = compute_angle(error44)
            rot_error_deg = rot_error * 180.0 / np.pi
            
            rotational_errors.append(rot_error)
            if rot_error_deg  < 0.5:
                rotational_errors_distributions['error < 0.5 deg'] = rotational_errors_distributions.get('error < 0.5 deg', 0) + 1
            if rot_error_deg < 1.0:
                rotational_errors_distributions['error < 1.0 deg'] = rotational_errors_distributions.get('error < 1.0 deg', 0) + 1
            if rot_error_deg < 2.0:
                rotational_errors_distributions['error < 2.0 deg'] = rotational_errors_distributions.get('error < 2.0 deg', 0) + 1
            if rot_error_deg < 5.0:
                rotational_errors_distributions['error < 5.0 deg'] = rotational_errors_distributions.get('error < 5.0 deg', 0) + 1
            if rot_error_deg < 10.0:
                rotational_errors_distributions['error < 10.0 deg'] = rotational_errors_distributions.get('error < 10.0 deg', 0) + 1
            if rot_error_deg >= 10.0:
                rotational_errors_distributions['error >= 10.0 deg'] = rotational_errors_distributions.get('error >= 10.0 deg', 0) + 1

            total_pairs += 1

        results[label] = {
            'rotational error rmse (deg)':   (np.sqrt(np.dot(rotational_errors, rotational_errors) / len(rotational_errors)) * 180.0 / np.pi),
            'rotational error mean (deg)':   (np.mean(rotational_errors) * 180.0 / np.pi),
            'rotational error median (deg)': (np.median(rotational_errors) * 180.0 / np.pi),
            'rotational error std (deg)':    (np.std(rotational_errors) * 180.0 / np.pi),
            'rotational error min (deg)':    (np.min(rotational_errors) * 180.0 / np.pi),
            'rotational error max (deg)':    (np.max(rotational_errors) * 180.0 / np.pi),
            'total pairs': total_pairs,
            'trajectory': os.path.basename(os.path.normpath(data.data_path))
        }
        results[label].update(rotational_errors_distributions)
    return results

def calculate_thresholded_ate(gt, osfm):
    alignment_error = osfm - gt
    translation_error_per_image = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
    translation_error = np.sqrt(np.dot(translation_error_per_image,translation_error_per_image) / len(translation_error_per_image))
    image_alignment_threshold_cm = 0.01
    image_alignment_threshold_dm = 0.1
    image_alignment_threshold_m = 1.0
    images_aligned_cm = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold_cm])
    images_aligned_dm = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold_dm])
    images_aligned_m = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold_m])

    return images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error

def align_and_calculate_ate(data, gt_full_list, osfm_full_list, gt_xyz_full, osfm_xyz_full, full_matches, comparison_label, ate_options):
    if len(full_matches) < 2:
        return 0, 0, 0, 0.0, 1.0, np.zeros((3,3)), np.zeros((1,3)) 

    sampled_matches = get_sample_matches(full_matches)
    gt_xyz_sampled = np.matrix([[float(value) for value in gt_full_list[a][0:3]] for a,b in sampled_matches]).transpose()
    osfm_xyz_sampled = np.matrix([[float(value) for value in osfm_full_list[b][0:3]] for a,b in sampled_matches]).transpose()
    
    # Don't use translation error from the sampled trajectroies
    R,t,_,s = evaluate_ate_scale.align(osfm_xyz_sampled, gt_xyz_sampled)
    osfm_xyz_full_aligned = s * R * osfm_xyz_full + t

    images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error = calculate_thresholded_ate(gt_xyz_full, osfm_xyz_full_aligned)
    return images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error, s, R, t

def write_report(data, graph,
                 features_time, matches_time, tracks_time):
    tracks, images = matching.tracks_and_images(graph)
    image_graph = bipartite.weighted_projected_graph(graph, images)
    view_graph = []
    for im1 in data.images():
        for im2 in data.images():
            if im1 in image_graph and im2 in image_graph[im1]:
                weight = image_graph[im1][im2]['weight']
                view_graph.append((im1, im2, weight))

    report = {
        "wall_times": {
            "load_features": features_time,
            "load_matches": matches_time,
            "compute_tracks": tracks_time,
        },
        "wall_time": features_time + matches_time + tracks_time,
        "num_images": len(images),
        "num_tracks": len(tracks),
        "view_graph": view_graph
    }
    data.save_report(io.json_dumps(report), 'validate_results.json')
