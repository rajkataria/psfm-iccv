import json
import logging
import math
import numpy as np
import os
import pyquaternion
import sys
from timeit import default_timer as timer

from networkx.algorithms import bipartite

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
        data = dataset.DataSet(args.dataset)
        
        ate_results = self.get_ate_results(data)
        reconstruction_results = self.get_reconstruction_results(data)

        data.save_ate_results(ate_results)
        data.save_reconstruction_results(reconstruction_results)

    def get_reconstruction_results(self, data):
        reconstruction_results = {}
        relevant_reconstructions = []
        options = {
            'min_triangulated': 0,
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
            graph_ = data.load_tracks_graph('tracks.csv')
        if data.tracks_graph_exists('tracks-thresholded-matches.csv'):
            graph_thresholded_matches = data.load_tracks_graph('tracks-thresholded-matches.csv')
        if data.tracks_graph_exists('tracks-all-matches.csv'):
            graph_all_matches = data.load_tracks_graph('tracks-all-matches.csv')
        if data.tracks_graph_exists('tracks-thresholded-weighted-matches.csv'):
            graph_thresholded_weighted_matches = data.load_tracks_graph('tracks-thresholded-weighted-matches.csv')
        if data.tracks_graph_exists('tracks-all-weighted-matches.csv'):
            graph_all_weighted_matches = data.load_tracks_graph('tracks-all-weighted-matches.csv')
        if data.tracks_graph_exists('tracks-gt-matches.csv'):
            graph_gt_matches = data.load_tracks_graph('tracks-gt-matches.csv')

        # Get results for baselines
        if data.reconstruction_exists('reconstruction.json'):
            logger.info('Computing reconstruction results for baseline...')
            stats_label = 'baseline'
            reconstruction_baseline = data.load_reconstruction('reconstruction.json')[0]
            relevant_reconstructions.append([graph_, reconstruction_baseline, baseline_command_keys, stats_label])

        if data.reconstruction_exists('reconstruction_colmap.json'):
            logger.info('Computing reconstruction results for colmap...')
            stats_label = 'colmap'
            reconstruction_colmap = data.load_reconstruction('reconstruction_colmap.json')[0]
            relevant_reconstructions.append([graph_, reconstruction_colmap, {}, stats_label])

        for imc in [True, False]:
            for wr in [True, False]:
                for gm in [True, False]:
                    for wfm in [True, False]:
                        for imt in [True, False]:
                            reconstruction_fn = 'reconstruction-imc-{}-wr-{}-gm-{}-wfm-{}-imt-{}.json'.format(imc, wr, gm, wfm, imt)
                            if data.reconstruction_exists(reconstruction_fn):
                                logger.info('Computing reconstruction results - imc-{}-wr-{}-gm-{}-wfm-{}-imt-{}.json'.format(imc, wr, gm, wfm, imt))
                                if imc is False and wr is False and gm is False and wfm is False and imt is False:
                                    stats_label = 'baseline'
                                else:
                                    stats_label = 'imc-{}-wr-{}-gm-{}-wfm-{}-imt-{}'.format(imc, wr, gm, wfm, imt)
                                reconstruction_ = data.load_reconstruction(reconstruction_fn)[0]

                                if gm is True:
                                    graph = data.load_tracks_graph('tracks-gt-matches.csv')
                                elif imc is True and wfm is True and imt is True:
                                    graph = graph_thresholded_weighted_matches
                                elif imc is True and wfm is True:
                                    graph = graph_all_weighted_matches
                                elif imc is True and imt is True:
                                    graph = graph_thresholded_matches
                                elif imc is True:
                                    graph = graph_all_matches
                                else:
                                    graph = graph_

                                relevant_reconstructions.append([graph, reconstruction_, classifier_command_keys, stats_label])

        for datum in relevant_reconstructions:
            g, r, k, l = datum
            reconstruction_results[l] = self.calculate_reconstruction_results(data, g, r, options, k)

        return reconstruction_results

    def get_ate_results(self, data):
        if not data.reconstruction_exists('reconstruction_gt.json'):
            logger.info('Skipping ATE calculation since no ground-truth exists...')
        else:
            relevant_reconstructions = []

            # Get results for baselines
            if data.reconstruction_exists('reconstruction.json'):
                logger.info('Computing ATE for baseline...')
                stats_label = 'baseline'
                reconstruction_baseline_gt = data.load_reconstruction('reconstruction_gt.json')[0]
                reconstruction_baseline = data.load_reconstruction('reconstruction.json')[0]
                self.intersect_reconstructions(data, reconstruction_baseline_gt, reconstruction_baseline)
                relevant_reconstructions.append([reconstruction_baseline_gt, reconstruction_baseline, stats_label])

            if data.reconstruction_exists('reconstruction_colmap.json'):
                logger.info('Computing ATE for colmap reconstruction...')
                stats_label = 'colmap'
                reconstruction_colmap_gt = data.load_reconstruction('reconstruction_gt.json')[0]
                reconstruction_colmap = data.load_reconstruction('reconstruction_colmap.json')[0]
                self.intersect_reconstructions(data, reconstruction_colmap_gt, reconstruction_colmap)
                relevant_reconstructions.append([reconstruction_colmap_gt, reconstruction_colmap, stats_label])

            for imc in [True, False]:
                for wr in [True, False]:
                    for gm in [True, False]:
                        for wfm in [True, False]:
                            for imt in [True, False]:
                                reconstruction_fn = 'reconstruction-imc-{}-wr-{}-gm-{}-wfm-{}-imt-{}.json'.format(imc, wr, gm, wfm, imt)
                                if data.reconstruction_exists(reconstruction_fn):
                                    logger.info('Computing reconstruction results - imc-{}-wr-{}-gm-{}-wfm-{}-imt-{}.json'.format(imc, wr, gm, wfm, imt))
                                    if imc is False and wr is False and gm is False and wfm is False and imt is False:
                                        stats_label = 'baseline'
                                    else:
                                        stats_label = 'imc-{}-wr-{}-gm-{}-wfm-{}-imt-{}'.format(imc, wr, gm, wfm, imt)

                                    reconstruction_gt = data.load_reconstruction('reconstruction_gt.json')[0]
                                    reconstruction_ = data.load_reconstruction(reconstruction_fn)[0]
                                    self.intersect_reconstructions(data, reconstruction_gt, reconstruction_)

                                    relevant_reconstructions.append([reconstruction_gt, reconstruction_, stats_label])

            for datum in relevant_reconstructions:
                r_gt, r, label = datum
                camera = types.PerspectiveCamera()
                camera_gt = types.PerspectiveCamera()
                r.add_camera(camera)
                r_gt.add_camera(camera_gt)
             
            ate_results = self.ransac_based_ate_evaluation(data, relevant_reconstructions)
        return ate_results

    def prune_reconstructions(self, data, reconstruction):
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

    def intersect_reconstructions(self, data, reconstruction_gt, reconstruction):
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

    def calculate_reconstruction_results(self, data, graph, reconstruction, options, command_keys):
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
            'time': round(total_time, 2)
        }
        return results

    def get_sample_matches(self, matches):
        samples = np.random.choice(len(matches), 2, replace=False)
        return np.array(matches)[samples].tolist()

    def plot_best_trajectories(self, data, gt_full_list, osfm_full_list, full_matches, s, R, t, comparison_label):
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
        evaluate_ate_scale.plot_traj(ax,gt_stamps,gt_xyz_full.transpose().A,'-','black','ground truth')
        evaluate_ate_scale.plot_traj(ax,osfm_stamps,osfm_xyz_full_aligned.transpose().A,'-','blue','estimated')

        label='difference'
        for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(full_matches,gt_xyz_full.transpose().A,osfm_xyz_full_aligned.transpose().A):
            ax.plot([x1,x2],[y1,y2],'-',color='red',label=label)
            label=''
            
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(os.path.join(data.data_path, 'results', 'ate-{}.png'.format(comparison_label)),dpi=90)

    def ransac_based_ate_evaluation(self, data, relevant_reconstructions):
        ate_options = {
            'i_scale': 1.0,
            'offset': 0.0,
            'max_difference': 0.02
        }
        
        total_ransac_iterations = 10000
        results = {}
        for r_gt, r_osfm, label in relevant_reconstructions:
            
            data.save_tum_format([r_gt], suffix='full-gt')
            data.save_tum_format([r_osfm], suffix='full-osfm')
            
            gt_full_list = associate.read_file_list(os.path.join(data.data_path, 'results', 'reconstruction-0-full-gt.txt'))
            osfm_full_list = associate.read_file_list(os.path.join(data.data_path, 'results', 'reconstruction-0-full-osfm.txt'))
            full_matches = associate.associate(gt_full_list, osfm_full_list, ate_options['offset'], ate_options['max_difference'])
            gt_xyz_full = np.matrix([[float(value) for value in gt_full_list[a][0:3]] for a,b in full_matches]).transpose()
            osfm_xyz_full = np.matrix([[float(value) for value in osfm_full_list[b][0:3]] for a,b in full_matches]).transpose()

            max_score = -sys.float_info.max
            for ransac_iteration in range(0, total_ransac_iterations):
                images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error, s, R, t = self.align_and_calculate_ate(data, gt_full_list, osfm_full_list, \
                    gt_xyz_full, osfm_xyz_full, full_matches, label, ate_options)

                if max_score < images_aligned_cm:
                    max_score = images_aligned_cm
                    best_model = {
                        's':s, 'R': R, 't': t, 'matches': full_matches, \
                        'images_aligned_cm': images_aligned_cm, 'images_aligned_dm': images_aligned_dm, 'images_aligned_m': images_aligned_m,\
                        'translation_error': translation_error
                        }
            
            if len(best_model['matches']) >= 2:
                self.plot_best_trajectories(data, gt_full_list, osfm_full_list, best_model['matches'], \
                    best_model['s'], best_model['R'], best_model['t'], label)

            results[label] = {
                'absolute translational error %f m': best_model['translation_error'],
                'images aligned %f cm': best_model['images_aligned_cm'],
                'images aligned %f dm': best_model['images_aligned_dm'],
                'images aligned %f m': best_model['images_aligned_m'],
                'trajectory': os.path.basename(os.path.normpath(data.data_path)),
                'total images %f': len(gt_full_list)
            }
            
        return results

    def calculate_thresholded_ate(self,gt, osfm):
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

    def align_and_calculate_ate(self, data, gt_full_list, osfm_full_list, gt_xyz_full, osfm_xyz_full, full_matches, comparison_label, ate_options):
        if len(full_matches) < 2:
            return 0, 0, 0, 0.0, 1.0, np.zeros((3,3)), np.zeros((1,3)) 

        sampled_matches = self.get_sample_matches(full_matches)
        gt_xyz_sampled = np.matrix([[float(value) for value in gt_full_list[a][0:3]] for a,b in sampled_matches]).transpose()
        osfm_xyz_sampled = np.matrix([[float(value) for value in osfm_full_list[b][0:3]] for a,b in sampled_matches]).transpose()
        
        # Don't use translation error from the sampled trajectroies
        R,t,_,s = evaluate_ate_scale.align(osfm_xyz_sampled, gt_xyz_sampled)
        osfm_xyz_full_aligned = s * R * osfm_xyz_full + t

        images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error = self.calculate_thresholded_ate(gt_xyz_full, osfm_xyz_full_aligned)
        return images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error, s, R, t

    def write_report(self, data, graph,
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
