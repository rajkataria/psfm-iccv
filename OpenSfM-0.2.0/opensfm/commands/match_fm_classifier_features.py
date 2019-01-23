import logging
from itertools import combinations
from timeit import default_timer as timer

import numpy as np
import scipy.spatial as spatial

from opensfm import dataset
from opensfm import geo
from opensfm import io
from opensfm import log
from opensfm import matching
from opensfm import classifier
from opensfm.context import parallel_map
from opensfm.commands import match_features


logger = logging.getLogger(__name__)


class Command:
    name = 'match_fm_classifier_features'
    help = 'Match features (returned by feature matching classifier) between image pairs'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        images = data.images()
        exifs = {im: data.load_exif(im) for im in images}
        if data.config.get('image_matcher_type', False) == 'VOCAB_TREE':
            pairs, preport = match_features.match_candidates_from_vocab_tree(images, exifs, data)
        elif data.config.get('image_matcher_type', False) == 'BRUTEFORCE':
            pairs, preport = match_features.match_candidates_bruteforce(images, exifs, data)
        else:
            pairs, preport = match_features.match_candidates_from_metadata(images, exifs, data)

        num_pairs = sum(len(c) for c in pairs.values())
        logger.info('Matching {} image pairs using matches from feature matching classifier (same # as lowe\'s threshold matches)'.format(num_pairs))

        ctx = Context()
        ctx.data = data
        ctx.cameras = ctx.data.load_camera_models()
        ctx.exifs = exifs
        ctx.p_pre, ctx.f_pre = match_features.load_preemptive_features(data)
        args = list(match_features.match_arguments(pairs, ctx))

        start = timer()
        processes = ctx.data.config['processes']
        parallel_map(match, args, processes)
        end = timer()
        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('match_fm_classifier_features: {0}\n'.format(end - start))
        self.write_report(data, preport, pairs, end - start)

    def write_report(self, data, preport, pairs, wall_time):
        pair_list = []
        for im1, others in pairs.items():
            for im2 in others:
                pair_list.append((im1, im2))

        report = {
            "wall_time": wall_time,
            "num_pairs": len(pair_list),
            "pairs": pair_list,
        }
        report.update(preport)
        data.save_report(io.json_dumps(report), 'matches_fm_classifier.json')


class Context:
    pass

def match(args):
    """Compute all matches for a single image"""
    log.setup()

    im1, candidates, i, n, ctx = args
    logger.info('Matching {}  -  {} / {}'.format(im1, i + 1, n))

    config = ctx.data.config
    robust_matching_min_match = config['robust_matching_min_match']
    lowes_ratio = config['lowes_ratio']

    im1_all_robust_matches = {}
    im1_valid_rmatches = {}
    im1_T = {}
    im1_F = {}
    im1_valid_inliers = {}

    im1_fmr = ctx.data.load_feature_matching_results(im1)
    p1, f1, c1 = ctx.data.load_features(im1)

    for im2 in candidates:
        # robust matching
        t_robust_matching = timer()
        camera1 = ctx.cameras[ctx.exifs[im1]['camera']]
        camera2 = ctx.cameras[ctx.exifs[im2]['camera']]

        unthresholded_matches = im1_fmr[im2]
        classified_matches = np.concatenate(( \
            np.array(unthresholded_matches['indices1']).reshape((-1,1)),
            np.array(unthresholded_matches['indices2']).reshape((-1,1)),
            np.array(unthresholded_matches['distances1']).reshape((-1,1)),
            np.array(unthresholded_matches['distances2']).reshape((-1,1)),
            np.array(unthresholded_matches['scores']).reshape((-1,1))
        ), axis=1)

        lowes_matches_indices = np.where((classified_matches[:,2] <= lowes_ratio) & (classified_matches[:,3] <= lowes_ratio))[0]
        relevant_feature_matching_indices = np.argsort(classified_matches[:,4])[::-1][:len(lowes_matches_indices)]
        thresholded_matches = classified_matches[relevant_feature_matching_indices, :]
        p2, f2, c2 = ctx.data.load_features(im2)
        rmatches, T, F, validity = classifier.robust_match_fundamental_weighted(p1, p2, thresholded_matches, config)

        im1_all_robust_matches[im2] = thresholded_matches
        im1_valid_rmatches[im2] = 1
        im1_T[im2] = T.tolist()
        im1_F[im2] = F.tolist()
        im1_valid_inliers[im2] = validity

        logger.debug("Full matching {0} / {1}, time: {2}s".format(
            len(rmatches), len(thresholded_matches), timer() - t_robust_matching))

    ctx.data.save_weighted_matches(im1, im1_all_robust_matches)