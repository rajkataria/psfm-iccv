import logging
from itertools import combinations
from timeit import default_timer as timer

import numpy as np
import pyopengv
import scipy.spatial as spatial
from sklearn.externals import joblib

from opensfm import dataset
from opensfm import geo
from opensfm import io
from opensfm import log
from opensfm import classifier
from opensfm.context import parallel_map
from multiprocessing import Pool


logger = logging.getLogger(__name__)

class Command:
    name = 'classify_features'
    help = 'Classify features - Used to determine relative pose (weighted ransac) and in weighted resctioning'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        start = timer()
        data = dataset.DataSet(args.dataset)
        images = data.images()
        exifs = {im: data.load_exif(im) for im in images}
        config = data.config
        processes = config['processes']
        ctx = Context()
        ctx.data = data
        ctx.cameras = ctx.data.load_camera_models()
        ctx.exifs = exifs
        
        logger.info('Loading features...')
        logger.info('Classifying feature matches...')
        args = []
        num_pairs = 0
        im_features = {}
        feature_matching_classifier_name = config.get('feature_matching_classifier')
        regr = joblib.load(feature_matching_classifier_name)
        for im1 in images:
            if im1 not in im_features:
                im_features[im1] = data.load_features(im1) # p1, f1, c1
            im1_unthresholded_matches = data.load_unthresholded_matches(im1)
            for im2 in im1_unthresholded_matches:
                if im2 not in im_features:
                    im_features[im2] = data.load_features(im2) # p1, f1, c1

                p1, f1, c1 = im_features[im1]
                p2, f2, c2 = im_features[im2]

                indices1 = im1_unthresholded_matches[im2][:, 0].astype(int)
                indices2 = im1_unthresholded_matches[im2][:, 1].astype(int)

                sizes1 = p1[indices1][:, 2].copy()
                angles1 = p1[indices1][:, 3].copy()
                sizes2 = p2[indices2][:, 2].copy()
                angles2 = p2[indices2][:, 3].copy()
                dists1 = im1_unthresholded_matches[im2][:, 2]
                dists2 = im1_unthresholded_matches[im2][:, 3]

                args.append([ \
                    np.array([im1, im2]), \
                    indices1, indices2, \
                    dists1, dists2, \
                    sizes1, sizes2, \
                    angles1, angles2, \
                    np.zeros((len(indices1),1)), \
                    False, regr, \
                    { 'classifier': 'BDT', 'max_depth': -1, 'n_estimators': -1} \
                    ])
                num_pairs = num_pairs + 1
        logger.info('Classifying {} total image pairs...'.format(num_pairs))
        
        p_results = []
        results = {}

        p = Pool(processes)
        if processes == 1:
            for arg in args:
                p_results.append(classifier.classify_boosted_dts_feature_match(arg))
        else:
            p_results = p.map(classifier.classify_boosted_dts_feature_match, args)
            p.close()

        for r in p_results:
            fns, indices1, indices2, distances1, distances2, _, scores = r
            im1, im2 = fns
            if im1 not in results:
                results[im1] = {}
            if im2 not in results:
                results[im2] = {}

            results[im1][im2] = {'im1': im1, 'im2': im2, 'indices1': indices1.tolist(), 'indices2': indices2.tolist(), \
                'scores': scores.tolist(), 'distances1': distances1.tolist(), 'distances2': distances2.tolist()}
            results[im2][im1] = {'im1': im2, 'im2': im1, 'indices1': indices2.tolist(), 'indices2': indices1.tolist(), \
                'scores': scores.tolist(), 'distances1': distances2.tolist(), 'distances2': distances1.tolist()}

            num_pairs = num_pairs + 1
        
        for im in results:
            data.save_feature_matching_results(im, results[im])
        end = timer()
        report = {
            "total_wall_time": end - start,
        }

        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('classify_features: {0}\n'.format(end - start))
        self.write_report(data, report)

    def write_report(self, data, report):
        data.save_report(io.json_dumps(report), 'classify_features.json')

class Context:
    pass
