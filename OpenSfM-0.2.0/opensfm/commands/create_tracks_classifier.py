import logging
import networkx as nx
import numpy as np
import os
import opensfm 
from timeit import default_timer as timer

from networkx.algorithms import bipartite

from opensfm import dataset
from opensfm import io
from opensfm import matching
from sklearn.metrics import euclidean_distances

logger = logging.getLogger(__name__)


class Command:
    name = 'create_tracks_classifier'
    help = "Link unthresholded matches pair-wise matches into tracks"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        config = data.config
        start = timer()
        features, colors = self.load_features(data)
        features_end = timer()
        options = {
            'robust_matches_threshold': 15,
            'shortest_path_label': 'rm-cost',
            'PCA-n_components': 2,
            'MDS-n_components': 2,
            # 'edge_threshold': 1.0/10.0,
            'edge_threshold': 10000000000,
            'lmds': False,
            'iteration': 0,
            'iteration_distance_filter_value': 0.6,
            'use_soft_tracks': data.config['use_soft_tracks'],
            'debug': False if 'aws' in os.uname()[2] else True
        }
        
        # matches_pruned = self.load_pruned_matches(data, spl=2, options=options)
        # matches_distance_pruned = self.load_distance_pruned_matches(data, options=options)
        # matches_distance_w_seq_pruned = self.load_distance_w_seq_pruned_matches(data, options=options)
        

        matches_all = self.load_all_matches(data, options=options)
        # matches_thresholded = self.load_thresholded_matches(data, options=options)
        # matches_pruned_thresholded = self.load_pruned_thresholded_matches(data, spl=2, options=options)
        matches_closest_images_thresholded = self.load_closest_images_thresholded_matches(data, options=options)
        matches_distance_thresholded = self.load_distance_thresholded_matches(data, options=options)
        matches_distance_ratio_thresholded = self.load_distance_ratio_thresholded_matches(data, options=options)
        matches_mst_adaptive_distance_thresholded = self.load_mst_adaptive_distance_thresholded_matches(data, options=options)
       
        
        

        # matches_distance_w_seq_pruned_thresholded = self.load_distance_w_seq_pruned_thresholded_matches(data, options=options)
        # matches_all_weighted = self.load_all_weighted_matches(data, options=options)
        # matches_thresholded_weighted = self.load_thresholded_weighted_matches(data, options=options)
        # matches_pruned_thresholded_weighted = self.load_pruned_thresholded_weighted_matches(data, spl=2, options=options)
        if data.reconstruction_exists('reconstruction_gt.json'):
            matches_gt = self.load_gt_matches(data, options=options)
        # if not config.get('production_mode', True) and data.reconstruction_exists('reconstruction_gt.json'):
        #     matches_gt_distance_pruned = self.load_gt_distance_pruned_matches(data, options=options)
        #     matches_gt_distance_pruned_thresholded = self.load_gt_distance_pruned_thresholded_matches(data, options=options)
        #     matches_gt = self.load_gt_matches(data, options=options)
        #     matches_gt_pruned = self.load_gt_pruned_matches(data, spl=2, options=options)
        #     matches_gt_selective = self.load_selective_gt_matches(data, options=options)
        # elif data.reconstruction_exists('reconstruction_gt.json'):
        #     matches_gt_distance_pruned = self.load_gt_distance_pruned_matches(data, options=options)
        #     matches_gt_distance_pruned_thresholded = self.load_gt_distance_pruned_thresholded_matches(data, options=options)

        matches_end = timer()
        # logger.info('Creating tracks graph using pruned rmatches')
        # tracks_graph_pruned = matching.create_tracks_graph(features, colors, matches_pruned,
        #                                             data.config)
        # logger.info('Creating tracks graph using distance pruned rmatches')
        # tracks_graph_distance_pruned = matching.create_tracks_graph(features, colors, matches_distance_pruned,
        #                                             data.config)
        # logger.info('Creating tracks graph using distance with sequences pruned rmatches')
        # tracks_graph_distance_w_seq_pruned = matching.create_tracks_graph(features, colors, matches_distance_w_seq_pruned,
        #                                             data.config)
        logger.info('Creating tracks graph using all matches')
        tracks_graph_all = matching.create_tracks_graph(features, colors, matches_all,
                                                    data.config)
        # logger.info('Creating tracks graph using thresholded matches')
        # tracks_graph_thresholded = matching.create_tracks_graph(features, colors, matches_thresholded,
        #                                             data.config)
        # logger.info('Creating tracks graph using pruned thresholded matches')
        # tracks_graph_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_pruned_thresholded,
        #                                             data.config)
        logger.info('Creating tracks graph using closest images thresholded matches')
        tracks_graph_closest_images_thresholded = matching.create_tracks_graph(features, colors, matches_closest_images_thresholded,
                                                    data.config)

        logger.info('Creating tracks graph using distance thresholded matches')
        tracks_graph_distance_thresholded = matching.create_tracks_graph(features, colors, matches_distance_thresholded,
                                                    data.config)

        logger.info('Creating tracks graph using distance ratio thresholded matches')
        tracks_graph_distance_ratio_thresholded = matching.create_tracks_graph(features, colors, matches_distance_ratio_thresholded,
                                                    data.config)
        
        logger.info('Creating tracks graph using MST and adaptive distance thresholded matches')
        tracks_graph_mst_adaptive_distance_thresholded = matching.create_tracks_graph(features, colors, matches_mst_adaptive_distance_thresholded,
                                                    data.config)
        
        # tracks_graph_distance_ratio_thresholded = self.create_distance_ratio_thresholded_tracks_graph(data, options=options)

        # logger.info('Creating tracks graph using distance with sequences pruned thresholded matches')
        # tracks_graph_distance_w_seq_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_distance_w_seq_pruned_thresholded,
        #                                             data.config)
        # logger.info('Creating tracks graph using all weighted matches')
        # tracks_graph_all_weighted = matching.create_tracks_graph(features, colors, matches_all_weighted,
        #                                             data.config)
        # logger.info('Creating tracks graph using thresholded weighted matches')
        # tracks_graph_thresholded_weighted = matching.create_tracks_graph(features, colors, matches_thresholded_weighted,
        #                                             data.config)
        # logger.info('Creating tracks graph using pruned thresholded weighted matches')
        # tracks_graph_pruned_thresholded_weighted = matching.create_tracks_graph(features, colors, matches_pruned_thresholded_weighted,
        #                                             data.config)

        # if not config.get('production_mode', True) and data.reconstruction_exists('reconstruction_gt.json'):
        #     logger.info('Creating tracks graph using ground-truth distance pruned rmatches')
        #     tracks_graph_gt_distance_pruned = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned,
        #                                             data.config)
        #     logger.info('Creating tracks graph using ground-truth distance pruned thresholded matches')
        #     tracks_graph_gt_distance_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned_thresholded,
        #                                             data.config)
        if data.reconstruction_exists('reconstruction_gt.json'):
            logger.info('Creating tracks graph using ground-truth matches')
            tracks_graph_gt = matching.create_tracks_graph(features, colors, matches_gt,
                                                        data.config)
        #     logger.info('Creating tracks graph using pruned ground-truth matches')
        #     tracks_graph_gt_pruned = matching.create_tracks_graph(features, colors, matches_gt_pruned,
        #                                                 data.config)
        #     logger.info('Creating tracks graph using selective ground-truth matches')
        #     tracks_graph_gt_selective = matching.create_tracks_graph(features, colors, matches_gt_selective,
        #                                                 data.config)
        # elif data.reconstruction_exists('reconstruction_gt.json'):
        #     logger.info('Creating tracks graph using ground-truth distance pruned rmatches')
        #     tracks_graph_gt_distance_pruned = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned,
        #                                             data.config)
        #     logger.info('Creating tracks graph using ground-truth distance pruned thresholded matches')
        #     tracks_graph_gt_distance_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned_thresholded,
        #                                             data.config)
            
        tracks_end = timer()
        # data.save_tracks_graph(tracks_graph_pruned, 'tracks-pruned-matches.csv')
        # data.save_tracks_graph(tracks_graph_distance_pruned, 'tracks-distance-pruned-matches.csv')
        # data.save_tracks_graph(tracks_graph_distance_w_seq_pruned, 'tracks-distance-w-seq-pruned-matches.csv')
        data.save_tracks_graph(tracks_graph_all, 'tracks-all-matches.csv')
        # data.save_tracks_graph(tracks_graph_thresholded, 'tracks-thresholded-matches.csv')
        # data.save_tracks_graph(tracks_graph_pruned_thresholded, 'tracks-pruned-thresholded-matches.csv')
        
        # data.save_tracks_graph(tracks_graph_distance_thresholded, 'tracks-distance-thresholded-matches-{}.csv'.format(data.config['distance_threshold_value']))
        data.save_tracks_graph(tracks_graph_closest_images_thresholded, 'tracks-mdstc-closest-images-mdstv-{}-mkcip-{}-mkcimin-{}-mkcimax-{}-ust-{}.csv'.format( \
            data.config['mds_threshold_value'], data.config['mds_k_closest_images_percentage'], data.config['mds_k_closest_images_min'], data.config['mds_k_closest_images_max'], \
            data.config['use_soft_tracks']
            )
        )

        data.save_tracks_graph(tracks_graph_distance_thresholded, 'tracks-mdstc-distance-mdstv-{}-mkcip-{}-mkcimin-{}-mkcimax-{}-ust-{}.csv'.format( \
            data.config['mds_threshold_value'], data.config['mds_k_closest_images_percentage'], data.config['mds_k_closest_images_min'], data.config['mds_k_closest_images_max'], \
            data.config['use_soft_tracks']
            )
        )

        data.save_tracks_graph(tracks_graph_distance_ratio_thresholded, 'tracks-mdstc-distance-ratio-mdstv-{}-mkcip-{}-mkcimin-{}-mkcimax-{}-ust-{}.csv'.format( \
            data.config['mds_threshold_value'], data.config['mds_k_closest_images_percentage'], data.config['mds_k_closest_images_min'], data.config['mds_k_closest_images_max'], \
            data.config['use_soft_tracks']
            )
        )

        data.save_tracks_graph(tracks_graph_mst_adaptive_distance_thresholded, 'tracks-mdstc-mst-adaptive-distance-mdstv-{}-mkcip-{}-mkcimin-{}-mkcimax-{}-ust-{}.csv'.format( \
            data.config['mds_threshold_value'], data.config['mds_k_closest_images_percentage'], data.config['mds_k_closest_images_min'], data.config['mds_k_closest_images_max'], \
            data.config['use_soft_tracks']
            )
        )

        # data.save_tracks_graph(tracks_graph_distance_w_seq_pruned_thresholded, 'tracks-distance-w-seq-pruned-thresholded-matches.csv')
        # data.save_tracks_graph(tracks_graph_all_weighted, 'tracks-all-weighted-matches.csv')
        # data.save_tracks_graph(tracks_graph_thresholded_weighted, 'tracks-thresholded-weighted-matches.csv')
        # data.save_tracks_graph(tracks_graph_pruned_thresholded_weighted, 'tracks-pruned-thresholded-weighted-matches.csv')
        # if not config.get('production_mode', True) and data.reconstruction_exists('reconstruction_gt.json'):
        #     data.save_tracks_graph(tracks_graph_gt_distance_pruned, 'tracks-gt-distance-pruned-matches.csv')
        #     data.save_tracks_graph(tracks_graph_gt_distance_pruned_thresholded, 'tracks-gt-distance-pruned-thresholded-matches.csv')
        if data.reconstruction_exists('reconstruction_gt.json'):
            data.save_tracks_graph(tracks_graph_gt, 'tracks-gt-matches.csv')
        #     data.save_tracks_graph(tracks_graph_gt_pruned, 'tracks-gt-matches-pruned.csv')
        #     data.save_tracks_graph(tracks_graph_gt_selective, 'tracks-gt-matches-selective.csv')
        # elif data.reconstruction_exists('reconstruction_gt.json'):
        #     data.save_tracks_graph(tracks_graph_gt_distance_pruned, 'tracks-gt-distance-pruned-matches.csv')
        #     data.save_tracks_graph(tracks_graph_gt_distance_pruned_thresholded, 'tracks-gt-distance-pruned-thresholded-matches.csv')

        end = timer()

        with open(data.profile_log(), 'a') as fout:
            fout.write('create_tracks_classifier: {0}\n'.format(end - start))

        self.write_report(data,
                          tracks_graph_all,
                          features_end - start,
                          matches_end - features_end,
                          tracks_end - matches_end)

    def load_features(self, data):
        logging.info('reading features')
        features = {}
        colors = {}
        for im in sorted(data.all_feature_maps()):
            p, f, c = data.load_features(im)
            features[im] = p[:, :2]
            colors[im] = c
        return features, colors

    # Modified original
    def load_matches(self, data, options):
        matches = {}
        robust_matching_min_match = data.config['robust_matching_min_match']
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1_matches[im2] >= robust_matching_min_match:
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_pruned_matches(self, data, spl, options):
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                        im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def get_top_k(self, data, closest_images_top_k_option):
        k = 10
        if closest_images_top_k_option == 'A':
            k = min(len(sorted(data.all_feature_maps())) - 1, max(20, len(sorted(data.all_feature_maps())) * 0.2))
        elif closest_images_top_k_option == 'B':
            k = min(len(sorted(data.all_feature_maps())) - 1, max(20, len(sorted(data.all_feature_maps())) * 0.1))
        elif closest_images_top_k_option == 'C':
            k = min(len(sorted(data.all_feature_maps())) - 1, max(10, len(sorted(data.all_feature_maps())) * 0.2))
        elif closest_images_top_k_option == 'D':
            k = min(len(sorted(data.all_feature_maps())) - 1, max(10, len(sorted(data.all_feature_maps())) * 0.1))
        elif closest_images_top_k_option == 'E':
            k = min(len(sorted(data.all_feature_maps())) - 1, len(sorted(data.all_feature_maps())) * 0.2)
        elif closest_images_top_k_option == 'F':
            k = min(len(sorted(data.all_feature_maps())) - 1, len(sorted(data.all_feature_maps())) * 0.1)
        elif closest_images_top_k_option == 'G':
            k = 20
        elif closest_images_top_k_option == 'H':
            k = 10
        return k

    def load_distance_pruned_matches(self, data, options):
        matches = {}
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        im_closest_images = data.load_closest_images('rm-cost')
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im1 in im_closest_images and im2 in im_closest_images[im1] \
                    and im_closest_images[im1].index(im2) <= closest_images_top_k:
                    #and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                    #    im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_distance_w_seq_pruned_matches(self, data, options):
        matches = {}
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        seq_cost_factor = 1.0
        im_closest_images = data.load_closest_images('rm-seq-cost-{}'.format(seq_cost_factor))
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im1 in im_closest_images and im2 in im_closest_images[im1] \
                    and im_closest_images[im1].index(im2) <= closest_images_top_k:
                    #and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                    #    im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_gt_distance_pruned_matches(self, data, options):
        matches = {}
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        im_closest_images = data.load_closest_images('gt')
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im1 in im_closest_images and im2 in im_closest_images[im1] \
                    and im_closest_images[im1].index(im2) <= closest_images_top_k:
                    #and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                    #    im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_all_matches(self, data, options):
        matches = {}
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                matches[im1, im2] = im1_matches[im2]
        return matches

    def load_thresholded_matches(self, data, options):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        image_matching_classifier_range = data.config.get('image_matching_classifier_range')
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if len(im1_matches[im2]) < image_matching_classifier_range[0]:
                    continue
                elif len(im1_matches[im2]) > image_matching_classifier_range[-1]:
                    matches[im1, im2] = im1_matches[im2]
                else:
                    if im1 in im_matching_results and im2 in im_matching_results[im1] \
                        and im_matching_results[im1][im2]['score'] >= image_matching_classifier_threshold:
                        matches[im1, im2] = im1_matches[im2]
                    elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                        and im_matching_results[im2][im1]['score'] >= image_matching_classifier_threshold:
                        matches[im1, im2] = im1_matches[im2]
        return matches

    def load_pruned_thresholded_matches(self, data, spl, options):
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] >= image_matching_classifier_threshold \
                    and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                        im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] >= image_matching_classifier_threshold \
                    and (im_matching_results[im2][im1]['shortest_path_length'] == spl or \
                        im_matching_results[im2][im1]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches    

    def load_closest_images_thresholded_matches(self, data, options):
        matches = {}
        robust_matching_min_match = data.config['robust_matching_min_match']
        mds_positions = data.load_mds_positions(label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
        for im1 in sorted(data.all_feature_maps()):
            closest_images = data.load_closest_images(im1, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                #distance_matrix = euclidean_distances([mds_positions[im1], mds_positions[im2]])
                # if len(im1_matches[im2]) >= robust_matching_min_match and distance_matrix[0,1] <= data.config['distance_threshold_value']:
                
                if len(im1_matches[im2]) >= robust_matching_min_match and closest_images.index(im2) <= min(max(data.config['mds_k_closest_images_min'], len(data.all_feature_maps()) * data.config['mds_k_closest_images_percentage']), data.config['mds_k_closest_images_max']):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_mst_adaptive_distance_thresholded_matches(self, data, options):
        im_mapping = {}
        im_reverse_mapping = {}
        mds_pairwise_distances = {}
        matches = {}
        robust_matching_min_match = data.config['robust_matching_min_match']

        mds_positions = data.load_mds_positions(label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
        mds_distance_matrix = euclidean_distances([mds_positions[im] for im in sorted(mds_positions.keys())])
        for i,im in enumerate(sorted(mds_positions.keys())):
            im_mapping[im] = i
            im_reverse_mapping[i] = im

        for i, im1 in enumerate(sorted(mds_positions.keys())):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for j, im2 in enumerate(sorted(mds_positions.keys())):
                if j <= i:
                    continue

                if im1 not in mds_pairwise_distances:
                    mds_pairwise_distances[im1] = {}
                if im2 not in mds_pairwise_distances:
                    mds_pairwise_distances[im2] = {}

                if im2 in im1_matches and len(im1_matches[im2]) >= robust_matching_min_match:
                    mds_pairwise_distances[im1][im2] = mds_distance_matrix[im_mapping[im1], im_mapping[im2]]
                    mds_pairwise_distances[im2][im1] = mds_distance_matrix[im_mapping[im2], im_mapping[im1]]

        G = opensfm.commands.formulate_graphs.formulate_graph([data, sorted(mds_positions.keys()), mds_pairwise_distances, 'mds-distances', 0.000000001])
        G = nx.minimum_spanning_tree(G)
        if options['debug']:
            opensfm.commands.formulate_graphs.draw_graph(G, os.path.join(data.data_path, 'results', 'graph-{}-it-{}-idfv-{}.png'.format('mds-distances', 'NA', 'NA')), highlighted_nodes=[], layout='spring', title=None)

        mean_edge_weight = 1.0 * np.mean([e['weight'] for im1, im2, e in G.edges(data=True)])
        max_edge_weight = 1.0 * max([e['weight'] for im1, im2, e in G.edges(data=True)])
        median_edge_weight = 1.0 * np.median([e['weight'] for im1, im2, e in G.edges(data=True)])

        print ('median weight: {}  mean weight: {}  max weight: {}'.format(median_edge_weight, mean_edge_weight, max_edge_weight))
        # import pdb; pdb.set_trace()
        for im1 in sorted(data.all_feature_maps()):
            # closest_images = data.load_closest_images(im1, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                pairwise_distance_matrix = euclidean_distances([mds_positions[im1], mds_positions[im2]])
                if len(im1_matches[im2]) >= robust_matching_min_match and (pairwise_distance_matrix[0,1] <= float(data.config['mds_threshold_value']) * max_edge_weight or G.has_edge(im1, im2)):
                # if len(im1_matches[im2]) >= robust_matching_min_match and closest_images.index(im2) <= min(max(data.config['mds_k_closest_images_min'], len(data.all_feature_maps()) * data.config['mds_k_closest_images_percentage']), data.config['mds_k_closest_images_max']):
                    matches[im1, im2] = im1_matches[im2]


        # print ('#'*100)
        # print ('#'*100)
        # print ('load_mst_adaptive_distance_thresholded_matches')
        # print ('#'*100)
        # print ('#'*100)
        # for im1, im2 in sorted(matches):
        #     if im1 == '0006.jpg' and im2 == '0012.jpg' or im2 == '0006.jpg' and im1 == '0012.jpg':
        #         print ('*'*100)
        #     if im1 == '0007.jpg' and im2 == '0012.jpg' or im2 == '0007.jpg' and im1 == '0012.jpg':
        #         print ('*'*100)

        #     print ('{} - {} : {}'.format(im1, im2, len(matches[im1,im2])))


        return matches

        # import pdb; pdb.set_trace()
        # import sys; sys.exit(1)

    def load_distance_thresholded_matches(self, data, options):
        matches = {}
        robust_matching_min_match = data.config['robust_matching_min_match']
        mds_positions = data.load_mds_positions(label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
        for im1 in sorted(data.all_feature_maps()):
            # closest_images = data.load_closest_images(im1, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                distance_matrix = euclidean_distances([mds_positions[im1], mds_positions[im2]])
                if len(im1_matches[im2]) >= robust_matching_min_match and distance_matrix[0,1] <= data.config['mds_threshold_value']:
                # if len(im1_matches[im2]) >= robust_matching_min_match and closest_images.index(im2) <= min(max(data.config['mds_k_closest_images_min'], len(data.all_feature_maps()) * data.config['mds_k_closest_images_percentage']), data.config['mds_k_closest_images_max']):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_distance_ratio_thresholded_matches(self, data, options):
        matches = {}
        im_mapping = {}
        im_reverse_mapping = {}
        outlier_matches = {}
        outlier_match_distances = {}
        outlier_match_count = {}
        outlier_percentages = {}
        epsilon = 0.00000001
        graph = data.load_tracks_graph('tracks.csv')

        mds_positions = data.load_mds_positions(label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
        mds_distance_matrix = euclidean_distances([mds_positions[im] for im in sorted(mds_positions.keys())])
        for i,im in enumerate(sorted(mds_positions.keys())):
            im_mapping[im] = i
            im_reverse_mapping[i] = im

        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in sorted(data.all_feature_maps()):
                if im1 == im2:
                    continue
                if im2 not in im1_matches or len(im1_matches[im2]) == 0:
                    try:
                        im2_matches = data.load_matches(im2)
                    except IOError:
                        continue
                    if im1 not in im2_matches or len(im2_matches[im1]) == 0:
                        continue
                common_tracks = set(graph[im1]).intersection(set(graph[im2]))
                for t in common_tracks:
                    track_images = sorted(graph[t].keys())
                    track_im_indices = np.array([im_mapping[im] for im in track_images])
                    im1_distances = mds_distance_matrix[im_mapping[im1], track_im_indices]
                    im2_distances = mds_distance_matrix[im_mapping[im2], track_im_indices]

                    closest_im1_distance = np.min(im1_distances[im1_distances > 0])
                    closest_im2_distance = np.min(im2_distances[im2_distances > 0])
                    im1_im2_distance = mds_distance_matrix[im_mapping[im1], im_mapping[im2]]
                    # if im1 == '0012.jpg' and im2 == '0015.jpg':
                    #     import pdb; pdb.set_trace()
                    if im1 not in outlier_matches:
                        outlier_matches[im1] = {}
                        outlier_match_distances[im1] = {}
                        outlier_match_count[im1] = {}
                    if im2 not in outlier_matches[im1]:
                        outlier_matches[im1][im2] = 0
                        outlier_match_distances[im1][im2] = 0.0
                        outlier_match_count[im1][im2] = 0

                    # graph.remove_node(t)
                    if im2 in im1_matches and len(im1_matches[im2]) > 0:
                        try:
                            match_index = np.where((graph[t][im1]['feature_id'] == im1_matches[im2][:,0]) & (graph[t][im2]['feature_id'] == im1_matches[im2][:,1]))[0]
                        except:
                            import pdb; pdb.set_trace()
                    elif im1 in im2_matches:
                        try:
                            match_index = np.where((graph[t][im2]['feature_id'] == im2_matches[im1][:,0]) & (graph[t][im1]['feature_id'] == im2_matches[im1][:,1]))[0]
                        except:
                            import pdb; pdb.set_trace()

                    if len(match_index) == 0:
                        continue
                    # if im1 == '0006.jpg' and im2 == '0012.jpg':
                    #     import pdb; pdb.set_trace()
                    if (1.0 * im1_im2_distance / min(closest_im1_distance, closest_im2_distance)) > data.config['mds_threshold_value']:
                        outlier_matches[im1][im2] += 1

                    outlier_match_distances[im1][im2] += im1_im2_distance
                    outlier_match_count[im1][im2] += 1
        # import pdb; pdb.set_trace()

        for i, im1 in enumerate(sorted(data.all_feature_maps())):
            # if im1 == '0012.jpg':
            #     print (outlier_matches['0012.jpg']['0015.jpg'])
            #     import pdb; pdb.set_trace()
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for j, im2 in enumerate(sorted(data.all_feature_maps())):
                # if im1 == '0012.jpg' and im2 == '0015.jpg':
                #     import pdb; pdb.set_trace()

                if j <= i:
                    continue


                if im2 not in im1_matches:
                    try:
                        im2_matches = data.load_matches(im2)
                    except IOError:
                        continue
                    im1_im2_matches = len(im2_matches[im1])
                else:
                    im1_im2_matches = len(im1_matches[im2])
                # try:
                
                if im1 not in outlier_percentages:
                    outlier_percentages[im1] = {}
                if im2 not in outlier_percentages:
                    outlier_percentages[im2] = {}

                # if im1 == '0012.jpg' and im2 == '0015.jpg':
                #     import pdb; pdb.set_trace()
                if im1 in outlier_matches and im2 in outlier_matches[im1]:
                    outlier_percentages[im1][im2] = 1.0 * outlier_matches[im1][im2] / (im1_im2_matches + epsilon)
                    outlier_percentages[im2][im1] = 1.0 * outlier_matches[im1][im2] / (im1_im2_matches + epsilon)

                    print ('Outliers: {} / {}     :     {} / {} = {}          |          {} / {} = {}'.format(im1, im2, outlier_matches[im1][im2], im1_im2_matches, outlier_percentages[im1][im2], outlier_match_distances[im1][im2], outlier_match_count[im1][im2], 1.0*outlier_match_distances[im1][im2] / (outlier_match_count[im1][im2] + epsilon) ))
                # except:
                    # import pdb; pdb.set_trace()                    

        for im1 in sorted(data.all_feature_maps()):
            # closest_images = data.load_closest_images(im1, label='{}-PCA_n_components-{}-MDS_n_components-{}-edge_threshold-{}-lmds-{}-it-{}-idfv-{}-ust-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['edge_threshold'], options['lmds'], options['iteration'], options['iteration_distance_filter_value'], options['use_soft_tracks']))
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                # distance_matrix = euclidean_distances([mds_positions[im1], mds_positions[im2]])
                # if len(im1_matches[im2]) >= robust_matching_min_match and distance_matrix[0,1] <= data.config['mds_threshold_value']:
                if im1 not in outlier_percentages or im2 not in outlier_percentages[im1] or outlier_percentages[im1][im2] <= 0.4:
                # if 1.0*outlier_match_distances[im1][im2] / outlier_match_count[im1][im2] < 0.5:
                    # if im1 == '0012.jpg' and im2 == '0015.jpg':
                    #     import pdb; pdb.set_trace()
                # if len(im1_matches[im2]) >= robust_matching_min_match and closest_images.index(im2) <= min(max(data.config['mds_k_closest_images_min'], len(data.all_feature_maps()) * data.config['mds_k_closest_images_percentage']), data.config['mds_k_closest_images_max']):
                    matches[im1, im2] = im1_matches[im2]
        
        for im1, im2 in sorted(matches):
            if im1 == '0006.jpg' and im2 == '0012.jpg' or im2 == '0006.jpg' and im1 == '0012.jpg':
                print ('*'*100)
            if im1 == '0007.jpg' and im2 == '0012.jpg' or im2 == '0007.jpg' and im1 == '0012.jpg':
                print ('*'*100)

            print ('{} - {} : {}'.format(im1, im2, len(matches[im1,im2])))

        # import pdb; pdb.set_trace()
        return matches
        # import pdb; pdb.set_trace()
        # tracks, images = matching.tracks_and_images(graph)
        # logger.debug('Good tracks: {}'.format(len(tracks)))
        # return graph

    def load_distance_w_seq_pruned_thresholded_matches(self, data, options):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        seq_cost_factor = 1.0
        im_closest_images = data.load_closest_images('rm-seq-cost-{}'.format(seq_cost_factor))
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] >= image_matching_classifier_threshold \
                    and im1 in im_closest_images and im2 in im_closest_images[im1] \
                    and im_closest_images[im1].index(im2) <= closest_images_top_k:
                    #and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                    #    im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] >= image_matching_classifier_threshold \
                    and im2 in im_closest_images and im1 in im_closest_images[im2] \
                    and im_closest_images[im2].index(im1) <= closest_images_top_k:
                    #and (im_matching_results[im2][im1]['shortest_path_length'] == spl or \
                    #    im_matching_results[im2][im1]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_gt_distance_pruned_thresholded_matches(self, data, options):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        im_closest_images = data.load_closest_images('gt')
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] >= image_matching_classifier_threshold \
                    and im1 in im_closest_images and im2 in im_closest_images[im1] \
                    and im_closest_images[im1].index(im2) <= closest_images_top_k:
                    #and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                    #    im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] >= image_matching_classifier_threshold \
                    and im2 in im_closest_images and im1 in im_closest_images[im2] \
                    and im_closest_images[im2].index(im1) <= closest_images_top_k:
                    #and (im_matching_results[im2][im1]['shortest_path_length'] == spl or \
                    #    im_matching_results[im2][im1]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_all_weighted_matches(self, data, options):
        matches = {}
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_weighted_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                matches[im1, im2] = im1_matches[im2]
        return matches

    def load_thresholded_weighted_matches(self, data, options):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_weighted_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] >= image_matching_classifier_threshold:
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] >= image_matching_classifier_threshold:
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_pruned_thresholded_weighted_matches(self, data, spl, options):
        # Pruning refers to shortest path length (=2)
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        im_matching_results = data.load_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_weighted_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] >= image_matching_classifier_threshold \
                    and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                        im_matching_results[im1][im2]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] >= image_matching_classifier_threshold \
                    and (im_matching_results[im2][im1]['shortest_path_length'] == spl or \
                        im_matching_results[im2][im1]['num_rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_gt_matches(self, data, options):
        matches = {}
        im_matching_results = data.load_groundtruth_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] == 1.0:
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] == 1.0:
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_selective_gt_matches(self, data, options):
        matches = {}
        gt_matches_selective_threshold = data.config.get('gt_matches_selective_threshold')
        im_matching_results = data.load_groundtruth_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                im1_matches = data.load_matches(im1)
                _, _, im1_all_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_all_matches:
                if im2 in im1_matches and len(im1_matches[im2]) < gt_matches_selective_threshold:
                    matches[im1, im2] = im1_matches[im2]
                else:
                    if im1 in im_matching_results and im2 in im_matching_results[im1] \
                        and im_matching_results[im1][im2]['score'] == 1.0 \
                        and im_matching_results[im1][im2]['rmatches'] >= gt_matches_selective_threshold:
                        matches[im1, im2] = im1_all_matches[im2]
                    elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                        and im_matching_results[im2][im1]['score'] == 1.0 \
                        and im_matching_results[im2][im1]['rmatches'] >= gt_matches_selective_threshold:
                        matches[im1, im2] = im1_all_matches[im2]
        return matches

    def load_gt_pruned_matches(self, data, spl, options):
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        im_matching_results = data.load_groundtruth_image_matching_results(options['robust_matches_threshold'])
        for im1 in sorted(data.all_feature_maps()):
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1 in im_matching_results and im2 in im_matching_results[im1] \
                    and im_matching_results[im1][im2]['score'] == 1.0 \
                    and (im_matching_results[im1][im2]['shortest_path_length'] == spl or \
                        im_matching_results[im1][im2]['rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
                elif im2 in im_matching_results and im1 in im_matching_results[im2] \
                    and im_matching_results[im2][im1]['score'] == 1.0 \
                    and (im_matching_results[im2][im1]['shortest_path_length'] == spl or \
                        im_matching_results[im2][im1]['rmatches'] >= shortest_path_rmatches_threshold):
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def write_report(self, data, graph,
                     features_time, matches_time, tracks_time):
        tracks, images = matching.tracks_and_images(graph)
        image_graph = bipartite.weighted_projected_graph(graph, images)
        view_graph = []
        for im1 in sorted(data.all_feature_maps()):
            for im2 in sorted(data.all_feature_maps()):
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
        data.save_report(io.json_dumps(report), 'tracks-classifier.json')
