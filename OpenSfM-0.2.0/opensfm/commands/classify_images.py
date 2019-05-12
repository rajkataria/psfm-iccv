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
    name = 'classify_images'
    help = 'Classify images for image matching'

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
        transformations = data.load_transformations()
        photometric_errors = data.load_photometric_errors()
        # triplet_pairwise_errors = data.load_triplet_pairwise_errors()
        consistency_errors = data.load_consistency_errors(cutoff=3, edge_threshold=15)
        shortest_paths = data.load_shortest_paths()
        spatial_entropies = data.load_spatial_entropies()
        color_histograms = data.load_color_histograms()
        nbvs = data.load_nbvs()
        vt_ranks, vt_scores = data.load_vocab_ranks_and_scores()
        sequence_scores_mean, sequence_scores_min, sequence_scores_max, sequence_distance_scores = \
            data.sequence_rank_adapter()
        lccs = data.load_lccs()

        logger.info('Classifying images...')
        args = []
        num_pairs = 0
        image_matching_classifier_name = config.get('image_matching_classifier')
        image_matching_classifier_thresholds = config.get('image_matching_classifier_thresholds')
        regr = joblib.load(image_matching_classifier_name)
        for im1 in images:
            im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
            for im2 in im1_all_robust_matches:
                rmatches = im1_all_robust_matches[im2]
                matches = im1_all_matches[im2]
                if len(rmatches) == 0:
                    continue

                T = transformations[im1][im2]
                R = np.array(T['rotation'])
                se = spatial_entropies[im1][im2]
                pe = np.array(photometric_errors[im1][im2]['histogram'])
                pe_percentage = np.array(photometric_errors[im1][im2]['polygon_area_percentage'])
                # te = np.array(triplet_pairwise_errors[im1][im2]['histogram'])
                te = np.array(consistency_errors[im1][im2]['histogram'])
                colmap = nbvs[im1][im2]

                chist_im1 = np.array([color_histograms[im1]['histogram']])
                chist_im2 = np.array([color_histograms[im2]['histogram']])
                vt_rank_percentage_im1_im2 = np.array([100.0 * vt_ranks[im1][im2] / len(data.images())])
                vt_rank_percentage_im2_im1 = np.array([100.0 * vt_ranks[im2][im1] / len(data.images())])
                sq_scores_mean, sq_scores_min, sq_scores_max, sq_distance_scores = \
                    np.array(sequence_scores_mean[im1][im2]), np.array(sequence_scores_min[im1][im2]), \
                    np.array(sequence_scores_max[im1][im2]), np.array(sequence_distance_scores[im1][im2])

                timedelta = 0.0
                feature_matching_scores = 0.0
                feature_matching_rmatch_scores = 0.0

                args.append([ \
                    data.data_path,
                    np.array([im1, im2]), \
                    R[0][0], R[0][1], R[0][2], R[1][0], R[1][1], R[1][2], R[2][0], R[2][1], R[2][2], \
                    np.array([len(rmatches)]), np.array([len(matches)]), \
                    np.array([se['entropy_im1_8']]), np.array([se['entropy_im2_8']]), np.array([se['entropy_im1_16']]), np.array([se['entropy_im2_16']]), \
                    np.array([pe]), pe_percentage, \
                    colmap['nbvs_im1'], colmap['nbvs_im2'], \
                    # feature_matching_scores, feature_matching_rmatch_scores, \
                    te, chist_im1, chist_im2, \
                    vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
                    sq_scores_mean, sq_scores_min, sq_scores_max, sq_distance_scores, \
                    np.array([lccs[im1][15]]), np.array([lccs[im2][15]]), np.array([min(lccs[im1][15],lccs[im2][15])]), np.array([max(lccs[im1][15],lccs[im2][15])]), \
                    np.array([lccs[im1][20]]), np.array([lccs[im2][20]]), np.array([min(lccs[im1][20],lccs[im2][20])]), np.array([max(lccs[im1][20],lccs[im2][20])]), \
                    np.array([lccs[im1][25]]), np.array([lccs[im2][25]]), np.array([min(lccs[im1][25],lccs[im2][25])]), np.array([max(lccs[im1][25],lccs[im2][25])]), \
                    np.array([lccs[im1][30]]), np.array([lccs[im2][30]]), np.array([min(lccs[im1][30],lccs[im2][30])]), np.array([max(lccs[im1][30],lccs[im2][30])]), \
                    np.array([lccs[im1][35]]), np.array([lccs[im2][35]]), np.array([min(lccs[im1][35],lccs[im2][35])]), np.array([max(lccs[im1][35],lccs[im2][35])]), \
                    np.array([lccs[im1][40]]), np.array([lccs[im2][40]]), np.array([min(lccs[im1][40],lccs[im2][40])]), np.array([max(lccs[im1][40],lccs[im2][40])]), \
                    np.array([len(shortest_paths[im1][im2]["shortest_path"])]), \
                    [0], [0], \
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
                p_results.append(classifier.classify_boosted_dts_image_match(arg))
        else:
            p_results = p.map(classifier.classify_boosted_dts_image_match, args)
            p.close()

        for r in p_results:
            fns, num_rmatches, _, score, shortest_path_length, _ = r
            im1, im2 = fns

            # if num_rmatches < image_matching_classifier_thresholds[0]:
            #     score = [0.0]
            # elif num_rmatches > image_matching_classifier_thresholds[-1]:
            #     score = [1.0]

            if im1 not in results:
                results[im1] = {}
            if im2 not in results:
                results[im2] = {}

            results[im1][im2] = {'im1': im1, 'im2': im2, 'score': score[0], 'num_rmatches': num_rmatches[0], 'shortest_path_length': shortest_path_length[0]}
            results[im2][im1] = {'im1': im2, 'im2': im1, 'score': score[0], 'num_rmatches': num_rmatches[0], 'shortest_path_length': shortest_path_length[0]}
            num_pairs = num_pairs + 1

        # print results
        data.save_image_matching_results(results)


        end = timer()

        report = {
            # "num_pairs": num_pairs,
            # "transformations_wall_time": e_transformations - s_transformations,
            # "triplet_errors_wall_time": e_triplets - s_triplets,
            # "spatial_entropies_wall_time": e_spatial_entropies - s_spatial_entropies,
            # "color_histograms_wall_time": e_color_histograms - s_color_histograms,
            # "photometric_errors_wall_time": e_photometric_errors - s_photometric_errors,
            # "transformations_wall_time": e_transformations - s_transformations,
            # "nbvs_wall_time": e_nbvs - s_nbvs,
            "total_wall_time": end - start,
        }

        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('classify_images: {0}\n'.format(end - start))
        self.write_report(data, report)

    def write_report(self, data, report):
        data.save_report(io.json_dumps(report), 'classify_images.json')

class Context:
    pass
