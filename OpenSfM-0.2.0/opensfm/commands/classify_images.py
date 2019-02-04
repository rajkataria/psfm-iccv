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
        triplet_pairwise_errors = data.load_triplet_pairwise_errors()
        spatial_entropies = data.load_spatial_entropies()
        color_histograms = data.load_color_histograms()
        nbvs = data.load_nbvs()
        vt_ranks, vt_scores = data.load_vocab_ranks_and_scores()

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
                te = np.array(triplet_pairwise_errors[im1][im2]['histogram'])
                colmap = nbvs[im1][im2]
                # chist = np.concatenate((
                #     np.array(color_histograms[im1]['histogram']).reshape((-1,1)), np.array(color_histograms[im2]['histogram']).reshape((-1,1))
                #     ))
                chist_im1 = np.array([color_histograms[im1]['histogram']])
                chist_im2 = np.array([color_histograms[im2]['histogram']])
                vt_rank_percentage_im1_im2 = np.array([100.0 * vt_ranks[im1][im2] / len(data.images())])
                vt_rank_percentage_im2_im1 = np.array([100.0 * vt_ranks[im2][im1] / len(data.images())])
                timedelta = 0.0
                feature_matching_scores = 0.0
                feature_matching_rmatch_scores = 0.0

                # print '#'*200
                # print '#'*25 + ' {} - {} : {} '.format(im1, im2, len(rmatches)) + '#'*25
                # print '#'*25 + ' transformations ' + '#'*25
                
                # print '#'*25 + ' photometric_errors ' + '#'*25
                # print photometric_errors[im1][im2]
                # print '#'*25 + ' triplet_pairwise_errors ' + '#'*25
                # print triplet_pairwise_errors[im1][im2]
                # print '#'*25 + ' spatial_entropies ' + '#'*25
                # print spatial_entropies[im1][im2]

                # print '#'*25 + ' color_histograms ' + '#'*25
                # print color_histograms[im1]
                # print color_histograms[im2]
                # print '#'*25 + ' nbvs ' + '#'*25
                # print nbvs[im1][im2]

                # regr, R33s, rmatches, matches, \
                # spatial_entropy_1_8x8, spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, \
                # photometric_histogram, polygon_area_percentage, \
                # feature_matching_scores, feature_matching_rmatch_scores, \
                # triplet_histogram, nbvs, timedelta, color_histograms = arg
                
                # classify_boosted_dts_image_match(
                #     fns, R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, num_rmatches, num_matches, spatial_entropy_1_8x8, \
                #     spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, pe_histogram, pe_polygon_area_percentage, \
                #     nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, labels, \
                #     train=False, regr=None, options={}
                #     ):
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
                    [0], \
                    False, regr, \
                    { 'classifier': 'BDT', 'max_depth': -1, 'n_estimators': -1} \
                    ])
                # args.append([ \
                #     im1, im2, regr, T['rotation'][2][2], len(rmatches), len(matches), \
                #     se['entropy_im1_8'], se['entropy_im2_8'], se['entropy_im1_16'], se['entropy_im2_16'], \
                #     pe, pe_percentage, feature_matching_scores, feature_matching_rmatch_scores, \
                #     te, colmap, timedelta, chist \
                #     ])
                num_pairs = num_pairs + 1
        logger.info('Classifying {} total image pairs...'.format(num_pairs))
        # import sys
        # sys.exit(1)
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
            fns, num_rmatches, _, score, _ = r
            im1, im2 = fns

            if num_rmatches < image_matching_classifier_thresholds[0]:
                score = [0.0]
            elif num_rmatches > image_matching_classifier_thresholds[-1]:
                score = [1.0]

            if im1 not in results:
                results[im1] = {}
            if im2 not in results:
                results[im2] = {}

            results[im1][im2] = {'im1': im1, 'im2': im2, 'score': score[0], 'num_rmatches': num_rmatches[0]}
            results[im2][im1] = {'im1': im2, 'im2': im1, 'score': score[0], 'num_rmatches': num_rmatches[0]}
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
