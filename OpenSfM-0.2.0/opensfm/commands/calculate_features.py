import logging
from itertools import combinations
from timeit import default_timer as timer

import numpy as np
import pyopengv
import scipy.spatial as spatial

from opensfm import dataset
from opensfm import geo
from opensfm import io
from opensfm import log
from opensfm import classifier
from opensfm.context import parallel_map
from multiprocessing import Pool


logger = logging.getLogger(__name__)

class Command:
    name = 'calculate_features'
    help = 'Calculate features for image matching'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        start = timer()
        data = dataset.DataSet(args.dataset)
        images = data.images()
        exifs = {im: data.load_exif(im) for im in images}
        ctx = Context()
        ctx.data = data
        ctx.cameras = ctx.data.load_camera_models()
        ctx.exifs = exifs
        
        # DELETE THE NEXT 3 LINES (SEQ IS BEING CALCULATED LATER)
        # classifier.calculate_triplet_errors(ctx)
        # classifier.calculate_sequence_ranks(ctx)
        # _, num_pairs = classifier.calculate_transformations(ctx)
        # classifier.create_feature_matching_dataset(ctx)
        # classifier.calculate_photometric_errors(ctx)
        # classifier.create_image_matching_dataset(ctx)
        # import sys; sys.exit(1)

        s_transformations = timer()
        _, num_pairs = classifier.calculate_transformations(ctx)
        e_transformations = timer()

        s_photometric_errors = timer()
        classifier.calculate_photometric_errors(ctx)
        e_photometric_errors = timer()

        s_triplets = timer()
        classifier.calculate_triplet_errors(ctx)
        e_triplets = timer()

        s_seq = timer()
        classifier.calculate_sequence_ranks(ctx)
        e_seq = timer()

        s_spatial_entropies = timer()
        classifier.calculate_spatial_entropies(ctx)
        e_spatial_entropies = timer()

        s_color_histograms = timer()
        classifier.calculate_color_histograms(ctx)
        e_color_histograms = timer()

        s_nbvs = timer()
        classifier.calculate_nbvs(ctx)
        e_nbvs = timer()

        s_feature_matching_dataset = timer()
        classifier.create_feature_matching_dataset(ctx)
        e_feature_matching_dataset = timer()

        s_image_matching_dataset = timer()
        classifier.create_image_matching_dataset(ctx)
        e_image_matching_dataset = timer()

        end = timer()

        report = {
            "num_pairs": num_pairs,
            "transformations_wall_time": e_transformations - s_transformations,
            "triplet_errors_wall_time": e_triplets - s_triplets,
            "spatial_entropies_wall_time": e_spatial_entropies - s_spatial_entropies,
            "color_histograms_wall_time": e_color_histograms - s_color_histograms,
            "photometric_errors_wall_time": e_photometric_errors - s_photometric_errors,
            "transformations_wall_time": e_transformations - s_transformations,
            "nbvs_wall_time": e_nbvs - s_nbvs,
            "feature_matching_dataset_wall_time": e_feature_matching_dataset - s_feature_matching_dataset,
            "image_matching_dataset_wall_time": e_image_matching_dataset - s_image_matching_dataset,
            "total_wall_time": end - start,
        }

        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('calculate_features: {0}\n'.format(end - start))
        self.write_report(data, report)

    def write_report(self, data, report):
        data.save_report(io.json_dumps(report), 'calculate_features.json')

class Context:
    pass
