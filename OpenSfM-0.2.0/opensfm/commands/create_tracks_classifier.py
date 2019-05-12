import logging
from timeit import default_timer as timer

from networkx.algorithms import bipartite

from opensfm import dataset
from opensfm import io
from opensfm import matching

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

        matches_pruned = self.load_pruned_matches(data, spl=2)
        matches_distance_pruned = self.load_distance_pruned_matches(data)
        matches_distance_w_seq_pruned = self.load_distance_w_seq_pruned_matches(data)
        matches_all = self.load_all_matches(data)
        matches_thresholded = self.load_thresholded_matches(data)
        matches_pruned_thresholded = self.load_pruned_thresholded_matches(data, spl=2)
        matches_distance_pruned_thresholded = self.load_distance_pruned_thresholded_matches(data)
        matches_distance_w_seq_pruned_thresholded = self.load_distance_w_seq_pruned_thresholded_matches(data)
        matches_all_weighted = self.load_all_weighted_matches(data)
        matches_thresholded_weighted = self.load_thresholded_weighted_matches(data)
        matches_pruned_thresholded_weighted = self.load_pruned_thresholded_weighted_matches(data, spl=2)
        if not config.get('production_mode', True) and data.reconstruction_exists('reconstruction_gt.json'):
            matches_gt_distance_pruned = self.load_gt_distance_pruned_matches(data)
            matches_gt_distance_pruned_thresholded = self.load_gt_distance_pruned_thresholded_matches(data)
            matches_gt = self.load_gt_matches(data)
            matches_gt_pruned = self.load_gt_pruned_matches(data, spl=2)
            matches_gt_selective = self.load_selective_gt_matches(data)
        elif data.reconstruction_exists('reconstruction_gt.json'):
            matches_gt_distance_pruned = self.load_gt_distance_pruned_matches(data)
            matches_gt_distance_pruned_thresholded = self.load_gt_distance_pruned_thresholded_matches(data)

        matches_end = timer()
        logger.info('Creating tracks graph using pruned rmatches')
        tracks_graph_pruned = matching.create_tracks_graph(features, colors, matches_pruned,
                                                    data.config)
        logger.info('Creating tracks graph using distance pruned rmatches')
        tracks_graph_distance_pruned = matching.create_tracks_graph(features, colors, matches_distance_pruned,
                                                    data.config)
        logger.info('Creating tracks graph using distance with sequences pruned rmatches')
        tracks_graph_distance_w_seq_pruned = matching.create_tracks_graph(features, colors, matches_distance_w_seq_pruned,
                                                    data.config)
        logger.info('Creating tracks graph using all matches')
        tracks_graph_all = matching.create_tracks_graph(features, colors, matches_all,
                                                    data.config)
        logger.info('Creating tracks graph using thresholded matches')
        tracks_graph_thresholded = matching.create_tracks_graph(features, colors, matches_thresholded,
                                                    data.config)
        logger.info('Creating tracks graph using pruned thresholded matches')
        tracks_graph_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_pruned_thresholded,
                                                    data.config)
        logger.info('Creating tracks graph using distance pruned thresholded matches')
        tracks_graph_distance_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_distance_pruned_thresholded,
                                                    data.config)
        logger.info('Creating tracks graph using distance with sequences pruned thresholded matches')
        tracks_graph_distance_w_seq_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_distance_w_seq_pruned_thresholded,
                                                    data.config)
        logger.info('Creating tracks graph using all weighted matches')
        tracks_graph_all_weighted = matching.create_tracks_graph(features, colors, matches_all_weighted,
                                                    data.config)
        logger.info('Creating tracks graph using thresholded weighted matches')
        tracks_graph_thresholded_weighted = matching.create_tracks_graph(features, colors, matches_thresholded_weighted,
                                                    data.config)
        logger.info('Creating tracks graph using pruned thresholded weighted matches')
        tracks_graph_pruned_thresholded_weighted = matching.create_tracks_graph(features, colors, matches_pruned_thresholded_weighted,
                                                    data.config)

        if not config.get('production_mode', True) and data.reconstruction_exists('reconstruction_gt.json'):
            logger.info('Creating tracks graph using ground-truth distance pruned rmatches')
            tracks_graph_gt_distance_pruned = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned,
                                                    data.config)
            logger.info('Creating tracks graph using ground-truth distance pruned thresholded matches')
            tracks_graph_gt_distance_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned_thresholded,
                                                    data.config)
            logger.info('Creating tracks graph using ground-truth matches')
            tracks_graph_gt = matching.create_tracks_graph(features, colors, matches_gt,
                                                        data.config)
            logger.info('Creating tracks graph using pruned ground-truth matches')
            tracks_graph_gt_pruned = matching.create_tracks_graph(features, colors, matches_gt_pruned,
                                                        data.config)
            logger.info('Creating tracks graph using selective ground-truth matches')
            tracks_graph_gt_selective = matching.create_tracks_graph(features, colors, matches_gt_selective,
                                                        data.config)
        elif data.reconstruction_exists('reconstruction_gt.json'):
            logger.info('Creating tracks graph using ground-truth distance pruned rmatches')
            tracks_graph_gt_distance_pruned = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned,
                                                    data.config)
            logger.info('Creating tracks graph using ground-truth distance pruned thresholded matches')
            tracks_graph_gt_distance_pruned_thresholded = matching.create_tracks_graph(features, colors, matches_gt_distance_pruned_thresholded,
                                                    data.config)
            
        tracks_end = timer()
        data.save_tracks_graph(tracks_graph_pruned, 'tracks-pruned-matches.csv')
        data.save_tracks_graph(tracks_graph_distance_pruned, 'tracks-distance-pruned-matches.csv')
        data.save_tracks_graph(tracks_graph_distance_w_seq_pruned, 'tracks-distance-w-seq-pruned-matches.csv')
        data.save_tracks_graph(tracks_graph_all, 'tracks-all-matches.csv')
        data.save_tracks_graph(tracks_graph_thresholded, 'tracks-thresholded-matches.csv')
        data.save_tracks_graph(tracks_graph_pruned_thresholded, 'tracks-pruned-thresholded-matches.csv')
        data.save_tracks_graph(tracks_graph_distance_pruned_thresholded, 'tracks-distance-pruned-thresholded-matches.csv')
        data.save_tracks_graph(tracks_graph_distance_w_seq_pruned_thresholded, 'tracks-distance-w-seq-pruned-thresholded-matches.csv')
        data.save_tracks_graph(tracks_graph_all_weighted, 'tracks-all-weighted-matches.csv')
        data.save_tracks_graph(tracks_graph_thresholded_weighted, 'tracks-thresholded-weighted-matches.csv')
        data.save_tracks_graph(tracks_graph_pruned_thresholded_weighted, 'tracks-pruned-thresholded-weighted-matches.csv')
        if not config.get('production_mode', True) and data.reconstruction_exists('reconstruction_gt.json'):
            data.save_tracks_graph(tracks_graph_gt_distance_pruned, 'tracks-gt-distance-pruned-matches.csv')
            data.save_tracks_graph(tracks_graph_gt_distance_pruned_thresholded, 'tracks-gt-distance-pruned-thresholded-matches.csv')
            data.save_tracks_graph(tracks_graph_gt, 'tracks-gt-matches.csv')
            data.save_tracks_graph(tracks_graph_gt_pruned, 'tracks-gt-matches-pruned.csv')
            data.save_tracks_graph(tracks_graph_gt_selective, 'tracks-gt-matches-selective.csv')
        elif data.reconstruction_exists('reconstruction_gt.json'):
            data.save_tracks_graph(tracks_graph_gt_distance_pruned, 'tracks-gt-distance-pruned-matches.csv')
            data.save_tracks_graph(tracks_graph_gt_distance_pruned_thresholded, 'tracks-gt-distance-pruned-thresholded-matches.csv')

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
        for im in data.images():
            p, f, c = data.load_features(im)
            features[im] = p[:, :2]
            colors[im] = c
        return features, colors

    # Modified original
    def load_matches(self, data):
        matches = {}
        robust_matching_min_match = data.config['robust_matching_min_match']
        for im1 in data.images():
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                if im1_matches[im2] >= robust_matching_min_match:
                    matches[im1, im2] = im1_matches[im2]
        return matches

    def load_pruned_matches(self, data, spl):
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        im_matching_results = data.load_image_matching_results()
        for im1 in data.images():
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
            k = min(len(data.images()) - 1, max(20, len(data.images()) * 0.2))
        elif closest_images_top_k_option == 'B':
            k = min(len(data.images()) - 1, max(20, len(data.images()) * 0.1))
        elif closest_images_top_k_option == 'C':
            k = min(len(data.images()) - 1, max(10, len(data.images()) * 0.2))
        elif closest_images_top_k_option == 'D':
            k = min(len(data.images()) - 1, max(10, len(data.images()) * 0.1))
        elif closest_images_top_k_option == 'E':
            k = min(len(data.images()) - 1, len(data.images()) * 0.2)
        elif closest_images_top_k_option == 'F':
            k = min(len(data.images()) - 1, len(data.images()) * 0.1)
        elif closest_images_top_k_option == 'G':
            k = 20
        elif closest_images_top_k_option == 'H':
            k = 10
        return k

    def load_distance_pruned_matches(self, data):
        matches = {}
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results()
        im_closest_images = data.load_closest_images('rm-cost')
        for im1 in data.images():
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

    def load_distance_w_seq_pruned_matches(self, data):
        matches = {}
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results()
        seq_cost_factor = 1.0
        im_closest_images = data.load_closest_images('rm-seq-cost-{}'.format(seq_cost_factor))
        for im1 in data.images():
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

    def load_gt_distance_pruned_matches(self, data):
        matches = {}
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results()
        im_closest_images = data.load_closest_images('gt')
        for im1 in data.images():
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

    def load_all_matches(self, data):
        matches = {}
        for im1 in data.images():
            try:
                _, _, im1_matches = data.load_all_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                matches[im1, im2] = im1_matches[im2]
        return matches

    def load_thresholded_matches(self, data):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        image_matching_classifier_range = data.config.get('image_matching_classifier_range')
        im_matching_results = data.load_image_matching_results()
        for im1 in data.images():
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

    def load_pruned_thresholded_matches(self, data, spl):
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        im_matching_results = data.load_image_matching_results()
        for im1 in data.images():
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

    def load_distance_pruned_thresholded_matches(self, data):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results()
        im_closest_images = data.load_closest_images('rm-cost')
        for im1 in data.images():
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

    def load_distance_w_seq_pruned_thresholded_matches(self, data):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results()
        seq_cost_factor = 1.0
        im_closest_images = data.load_closest_images('rm-seq-cost-{}'.format(seq_cost_factor))
        for im1 in data.images():
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

    def load_gt_distance_pruned_thresholded_matches(self, data):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        closest_images_top_k = self.get_top_k(data, data.config['closest_images_top_k'])
        im_matching_results = data.load_image_matching_results()
        im_closest_images = data.load_closest_images('gt')
        for im1 in data.images():
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

    def load_all_weighted_matches(self, data):
        matches = {}
        for im1 in data.images():
            try:
                im1_matches = data.load_weighted_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
                matches[im1, im2] = im1_matches[im2]
        return matches

    def load_thresholded_weighted_matches(self, data):
        matches = {}
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        im_matching_results = data.load_image_matching_results()
        for im1 in data.images():
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

    def load_pruned_thresholded_weighted_matches(self, data, spl):
        # Pruning refers to shortest path length (=2)
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        image_matching_classifier_threshold = data.config.get('image_matching_classifier_threshold')
        im_matching_results = data.load_image_matching_results()
        for im1 in data.images():
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

    def load_gt_matches(self, data):
        matches = {}
        image_matching_classifier_range = data.config.get('image_matching_classifier_range')
        im_matching_results = data.load_groundtruth_image_matching_results(image_matching_classifier_range)
        for im1 in data.images():
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

    def load_selective_gt_matches(self, data):
        matches = {}
        image_matching_classifier_range = data.config.get('image_matching_classifier_range')
        gt_matches_selective_threshold = data.config.get('gt_matches_selective_threshold')
        im_matching_results = data.load_groundtruth_image_matching_results(image_matching_classifier_range)
        for im1 in data.images():
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

    def load_gt_pruned_matches(self, data, spl):
        matches = {}
        shortest_path_rmatches_threshold = data.config['shortest_path_rmatches_threshold']
        image_matching_classifier_range = data.config.get('image_matching_classifier_range')
        im_matching_results = data.load_groundtruth_image_matching_results(image_matching_classifier_range)
        for im1 in data.images():
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
        data.save_report(io.json_dumps(report), 'tracks-classifier.json')
