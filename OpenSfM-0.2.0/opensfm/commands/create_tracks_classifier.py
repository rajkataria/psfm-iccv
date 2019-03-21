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

        start = timer()
        features, colors = self.load_features(data)
        features_end = timer()

        matches_pruned = self.load_pruned_matches(data, spl=2)
        matches_all = self.load_all_matches(data)
        matches_thresholded = self.load_thresholded_matches(data)
        matches_pruned_thresholded = self.load_pruned_thresholded_matches(data, spl=2)
        matches_all_weighted = self.load_all_weighted_matches(data)
        matches_thresholded_weighted = self.load_thresholded_weighted_matches(data)
        matches_pruned_thresholded_weighted = self.load_pruned_thresholded_weighted_matches(data, spl=2)
        if data.reconstruction_exists('reconstruction_gt.json'):
            matches_gt = self.load_gt_matches(data)
            matches_gt_pruned = self.load_gt_pruned_matches(data, spl=2)
            matches_gt_selective = self.load_selective_gt_matches(data)

        matches_end = timer()
        logger.info('Creating tracks graph using pruned rmatches')
        tracks_graph_pruned = matching.create_tracks_graph(features, colors, matches_pruned,
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
        logger.info('Creating tracks graph using all weighted matches')
        tracks_graph_all_weighted = matching.create_tracks_graph(features, colors, matches_all_weighted,
                                                    data.config)
        logger.info('Creating tracks graph using thresholded weighted matches')
        tracks_graph_thresholded_weighted = matching.create_tracks_graph(features, colors, matches_thresholded_weighted,
                                                    data.config)
        logger.info('Creating tracks graph using pruned thresholded weighted matches')
        tracks_graph_pruned_thresholded_weighted = matching.create_tracks_graph(features, colors, matches_pruned_thresholded_weighted,
                                                    data.config)

        if data.reconstruction_exists('reconstruction_gt.json'):
            logger.info('Creating tracks graph using ground-truth matches')
            tracks_graph_gt = matching.create_tracks_graph(features, colors, matches_gt,
                                                        data.config)
            logger.info('Creating tracks graph using pruned ground-truth matches')
            tracks_graph_gt_pruned = matching.create_tracks_graph(features, colors, matches_gt_pruned,
                                                        data.config)
            logger.info('Creating tracks graph using selective ground-truth matches')
            tracks_graph_gt_selective = matching.create_tracks_graph(features, colors, matches_gt_selective,
                                                        data.config)
        tracks_end = timer()
        data.save_tracks_graph(tracks_graph_pruned, 'tracks-pruned-matches.csv')
        data.save_tracks_graph(tracks_graph_all, 'tracks-all-matches.csv')
        data.save_tracks_graph(tracks_graph_thresholded, 'tracks-thresholded-matches.csv')
        data.save_tracks_graph(tracks_graph_pruned_thresholded, 'tracks-pruned-thresholded-matches.csv')
        data.save_tracks_graph(tracks_graph_all_weighted, 'tracks-all-weighted-matches.csv')
        data.save_tracks_graph(tracks_graph_thresholded_weighted, 'tracks-thresholded-weighted-matches.csv')
        data.save_tracks_graph(tracks_graph_pruned_thresholded_weighted, 'tracks-pruned-thresholded-weighted-matches.csv')
        if data.reconstruction_exists('reconstruction_gt.json'):
            data.save_tracks_graph(tracks_graph_gt, 'tracks-gt-matches.csv')
            data.save_tracks_graph(tracks_graph_gt_pruned, 'tracks-gt-matches-pruned.csv')
            data.save_tracks_graph(tracks_graph_gt_selective, 'tracks-gt-matches-selective.csv')
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

    # Original
    def load_matches(self, data):
        matches = {}
        for im1 in data.images():
            try:
                im1_matches = data.load_matches(im1)
            except IOError:
                continue
            for im2 in im1_matches:
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
        im_matching_results = data.load_image_matching_results()
        for im1 in data.images():
            try:
                _, _, im1_matches = data.load_all_matches(im1)
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
        image_matching_classifier_thresholds = data.config.get('image_matching_classifier_thresholds')
        im_matching_results = data.load_groundtruth_image_matching_results(image_matching_classifier_thresholds)
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
        image_matching_classifier_thresholds = data.config.get('image_matching_classifier_thresholds')
        gt_matches_selective_threshold = data.config.get('gt_matches_selective_threshold')
        im_matching_results = data.load_groundtruth_image_matching_results(image_matching_classifier_thresholds)
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
        image_matching_classifier_thresholds = data.config.get('image_matching_classifier_thresholds')
        im_matching_results = data.load_groundtruth_image_matching_results(image_matching_classifier_thresholds)
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
