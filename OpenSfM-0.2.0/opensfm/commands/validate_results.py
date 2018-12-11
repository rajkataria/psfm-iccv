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
        options = {
            'min_triangulated': 0,
            'debug': False,
            'display_mode': 0,
            'show_header': True,
            'show_dataset': True,
            'dataset': data.data_path
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



        logger.info('Baseline:')
        graph = data.load_tracks_graph('tracks.csv')
        

        reconstruction_baseline_gt = data.load_reconstruction('reconstruction_gt.json')[0]
        reconstruction_baseline = data.load_reconstruction('reconstruction.json')[0]
        self.intersect_reconstructions(data, reconstruction_baseline_gt, reconstruction_baseline)

        reconstruction_colmap_gt = data.load_reconstruction('reconstruction_gt.json')[0]
        reconstruction_colmap = data.load_reconstruction('reconstruction_colmap.json')[0]
        self.intersect_reconstructions(data, reconstruction_colmap_gt, reconstruction_colmap)

        if False:
            reconstruction_classifier_weighted_gt = data.load_reconstruction('reconstruction_gt.json')[0]
            reconstruction_classifier_weighted = data.load_reconstruction('reconstruction-classifier-weighted.json')[0]
            self.intersect_reconstructions(data, reconstruction_classifier_weighted_gt, reconstruction_classifier_weighted)

            reconstruction_classifier_gt = data.load_reconstruction('reconstruction_gt.json')[0]
            reconstruction_classifier = data.load_reconstruction('reconstruction-classifier.json')[0]
            self.intersect_reconstructions(data, reconstruction_classifier_gt, reconstruction_classifier)

        # print ('{} / {}'.format(len(reconstruction[0].shots.keys()), len(reconstruction_gt[0].shots.keys())))
        # data.save_tum_format(reconstruction_gt, suffix='gt')
        # data.save_tum_format(reconstruction, suffix='opensfm')
        # data.save_tum_format(reconstruction_classifier_weighted, suffix='opensfm-classifier-weighted')
        # data.save_tum_format(reconstruction_classifier, suffix='opensfm-classifier')

        relevant_reconstructions = [
            [reconstruction_baseline_gt, reconstruction_baseline, 'baseline'],
            [reconstruction_colmap_gt, reconstruction_colmap, 'colmap'],
            # [reconstruction_classifier_weighted_gt, reconstruction_classifier_weighted, 'classifier-weighted'],
            # [reconstruction_classifier_gt, reconstruction_classifier, 'classifier'],
        ]
        for datum in relevant_reconstructions:
            r_gt, r, label = datum
            camera = types.PerspectiveCamera()
            camera_gt = types.PerspectiveCamera()
            
            r.add_camera(camera)
            r_gt.add_camera(camera_gt)
            # print (r_gt.cameras)


        ate_results = self.ransac_based_ate_evaluation(data, relevant_reconstructions)

        # self.calculate_statistics(data, graph, recon, options, baseline_command_keys)
        # self.save_tum_format(recon)
        if False:
            logger.info('Thresholded classifier scores with baseline resectioning:')
            graph = data.load_tracks_graph('tracks-thresholded-matches.csv')
            recon = data.load_reconstruction('reconstruction-classifier.json')
            self.calculate_statistics(data, graph, recon, options, classifier_command_keys)
            
            logger.info('All matches with weighted resectioning:')
            graph = data.load_tracks_graph('tracks-all-matches.csv')
            recon = data.load_reconstruction('reconstruction-classifier-weighted.json')
            self.calculate_statistics(data, graph, recon, options, classifier_command_keys)


        # start = timer()
        # features, colors = self.load_features(data)
        # features_end = timer()
        # matches = self.load_matches(data)
        # matches_end = timer()
        # tracks_graph = matching.create_tracks_graph(features, colors, matches,
        #                                             data.config)
        # tracks_end = timer()
        # data.save_tracks_graph(tracks_graph)
        # end = timer()

        # with open(data.profile_log(), 'a') as fout:
        #     fout.write('create_tracks: {0}\n'.format(end - start))

        # self.write_report(data,
        #                   tracks_graph,
        #                   features_end - start,
        #                   matches_end - features_end,
        #                   tracks_end - matches_end)


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

        # for r in reconstruction_gt:
        images_gt = set(reconstruction_gt.shots.keys())
        images = set(reconstruction.shots.keys())
        
        common_images = images.intersection(images_gt)

        # for r in reconstruction:
        #     additional_shots = []
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
                # del reconstruction_gt[0].shots[s]

    def calculate_statistics(self, data, graph, reconstruction, options, command_keys):
        registered = 0
        error = 0.0
        count = 0
        missing_errors = 0
        total_time = 0.0
        cameras = 0
        times = {}

        # graph = data.load_tracks_graph()
        tracks, images = matching.tracks_and_images(graph)
        recon0 = reconstruction[0]
        cameras = len(data.images())
        for s in recon0.shots:
            pts_triangulated = set(recon0.points.keys()).intersection(set(graph[s].keys()))
            if len(pts_triangulated) >= options['min_triangulated']:
                if options['debug']:
                    print 'Image: {}   Tracks: {}   Points Triangulated: {}'.format(s, len(graph[s].keys()), len(pts_triangulated))
                registered += 1

        for pid in recon0.points:
            if recon0.points[pid].reprojection_error:
                error = error + recon0.points[pid].reprojection_error
                count = count + 1
            else:
                missing_errors = missing_errors + 1

        profile_fn = os.path.join(options['dataset'], 'profile.log')
        if os.path.exists(profile_fn):
            with open(profile_fn, 'r') as f:
                for data in f.readlines():
                    datums = data.split(':')
                    if datums[0] in command_keys:
                        times[datums[0]] = float(datums[1])

            for key in times:
                total_time = total_time + times[key]
                if options['debug']:
                    print key + ' - ' + str(times[key])

        if options['display_mode'] == 0:
            logger.info('\t{} - Cameras Registered ({}/{}) Reprojection Error ({} points) : {} Missing Errors: {} ' \
                'Total Time: {}'.format(options['dataset'], registered, cameras, count, error/count, missing_errors, round(total_time,2)))
        # elif options['display_mode'] == 1    :
        #     if options['show_header']:
        #         logger.info('Dataset, Cameras Registered, Cameras Registered (minimum {} points triangulated), Points Triangulated, ' \
        #             'Reprojection Error, Missing Images, Total Time'.format(options['min_triangulated']))
        #     logger.info('{},{},{},{},{},{},{}'.format(options['dataset'], registered, cameras, count, error/count, missing_errors, round(total_time,2)))
        elif options['display_mode'] == 1:
            if options['show_dataset']:
                if options['show_header']:
                    logger.info('\tDataset, Cameras Registered, Points Triangulated, Total Time')
                logger.info('\t{},{},{},{}'.format(options['dataset'], registered, count, round(total_time,2)))
            else:
                if options['show_header']:
                    logger.info('\tCameras Registered, Points Triangulated, Total Time')
                logger.info('\t{},{},{}'.format(registered, count, round(total_time,2)))
        return

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

            max_score = sys.float_info.min
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
            
        print (json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))
        return results

    def calculate_thresholded_ate(self,gt, osfm):
        alignment_error = osfm - gt
        translation_error_per_image = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        translation_error = np.sqrt(np.dot(translation_error_per_image,translation_error_per_image) / len(translation_error_per_image))
        # image_alignment_threshold = np.sqrt(np.sum(np.power(np.max(gt, axis=1) - np.min(gt, axis=1), 2))) * 0.01
        # ransac_threshold = image_alignment_threshold * 0.1
        # ransac_score = len(translation_error_per_image[translation_error_per_image < ransac_threshold])
        # images_aligned = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold])
        image_alignment_threshold_cm = 0.01
        image_alignment_threshold_dm = 0.1
        image_alignment_threshold_m = 1.0
        images_aligned_cm = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold_cm])
        images_aligned_dm = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold_dm])
        images_aligned_m = len(translation_error_per_image[translation_error_per_image < image_alignment_threshold_m])

        return images_aligned_cm, images_aligned_dm, images_aligned_m, translation_error

    def align_and_calculate_ate(self, data, gt_full_list, osfm_full_list, gt_xyz_full, osfm_xyz_full, full_matches, comparison_label, ate_options):
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
