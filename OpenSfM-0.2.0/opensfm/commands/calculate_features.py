import logging
from itertools import combinations
from timeit import default_timer as timer

import numpy as np
import pyopengv
import scipy.spatial as spatial
import opensfm 

from opensfm import dataset
from opensfm import geo
from opensfm import io
from opensfm import log
from opensfm import classifier
from opensfm import matching
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


        # classifier.calculate_consistency_errors(ctx)
        # classifier.create_image_matching_dataset(ctx)
        # import sys; sys.exit(1)

        # classifier.calculate_spatial_entropies(ctx)
        # # classifier.calculate_gamma_adjusted_images(ctx)
        # # classifier.calculate_photometric_errors(ctx)
        # classifier.create_image_matching_dataset(ctx)
        # import sys; sys.exit(1)

        
        # grid_size = 224
        # for i,im1 in enumerate(sorted(data.images())):
        #     for j,im2 in enumerate(sorted(data.images())):
        #         if j <= i:
        #             continue
        #         classifier.perform_gamma_adjustment(data, im1, im2, grid_size)


        _, num_pairs = classifier.calculate_transformations(ctx)
        classifier.create_feature_matching_dataset(ctx)
        classifier.calculate_resized_images(ctx)
        classifier.calculate_spatial_entropies(ctx)
        classifier.calculate_gamma_adjusted_images(ctx)
        classifier.calculate_photometric_errors(ctx)
        classifier.output_image_keypoints(ctx)
        classifier.calculate_sequence_ranks(ctx)
        classifier.calculate_consistency_errors(ctx)
        classifier.calculate_nbvs(ctx)
        for sequence_cost_factor in [1.0]:
            ctx.sequence_cost_factor = sequence_cost_factor
            classifier.calculate_shortest_paths(ctx)
            classifier.infer_positions(ctx)
            classifier.infer_cleaner_positions(ctx)
        classifier.calculate_color_histograms(ctx)
        classifier.calculate_lccs(ctx)
        classifier.create_image_matching_dataset(ctx)
        
        import sys; sys.exit(1)
        

        feature_for_other_pipelines = False
        if feature_for_other_pipelines:
            _, num_pairs = classifier.calculate_transformations(ctx)
            classifier.calculate_spatial_entropies(ctx)
            classifier.calculate_photometric_errors(ctx)
            classifier.output_image_keypoints(ctx)
            classifier.calculate_sequence_ranks(ctx)
            for sequence_cost_factor in [1.0]:
                ctx.sequence_cost_factor = sequence_cost_factor
                classifier.calculate_shortest_paths(ctx)
            return


        # DELETE THE NEXT 3 LINES (SEQ IS BEING CALCULATED LATER)
        # classifier.calculate_triplet_errors(ctx)
        # classifier.calculate_sequence_ranks(ctx)
        # _, num_pairs = classifier.calculate_transformations(ctx)
        # classifier.create_feature_matching_dataset(ctx)
        # classifier.calculate_photometric_errors(ctx)
        # classifier.calculate_nbvs(ctx)
        # classifier.calculate_lccs(ctx)
        
        # classifier.calculate_spatial_entropies(ctx)
        
        # classifier.create_feature_matching_dataset(ctx)
        # classifier.calculate_shortest_paths(ctx)
        # classifier.calculate_consistency_errors(ctx)
        # classifier.calculate_shortest_paths(ctx)
        # classifier.calculate_nbvs(ctx)
        
        # classifier.create_feature_matching_dataset(ctx)
        # classifier.create_image_matching_dataset(ctx)
        # classifier.output_image_keypoints(ctx)




        # _, num_pairs = classifier.calculate_transformations(ctx)
        # classifier.calculate_spatial_entropies(ctx)
        # classifier.calculate_photometric_errors(ctx)
        # classifier.calculate_sequence_ranks(ctx)
        # classifier.calculate_consistency_errors(ctx)
        # classifier.output_image_keypoints(ctx)
        
        # for sequence_cost_factor in [0.25, 1.0, 5.0, 10.0]:
        # for sequence_cost_factor in [0.25, 5.0, 10.0]:
        for sequence_cost_factor in [1.0]:
            ctx.sequence_cost_factor = sequence_cost_factor
            classifier.calculate_shortest_paths(ctx)
            classifier.infer_positions(ctx)
            # classifier.infer_cleaner_positions(ctx)


        # for sequence_cost_factor in [1.0]:
        # # for sequence_cost_factor in [0.25, 1.0, 5.0, 10.0]:
        #     ctx.sequence_cost_factor = sequence_cost_factor
        #     classifier.infer_positions(ctx)
        #     classifier.infer_cleaner_positions(ctx)
            
        
        # features, colors = opensfm.commands.create_tracks.load_features(data)
        # matches = self.load_all_matches(data)
        # tracks_graph = matching.create_tracks_graph(features, colors, matches, data.config)
        # data.save_tracks_graph(tracks_graph)
        # classifier.create_tracks_map(ctx)
        # classifier.calculate_nbvs(ctx)
        classifier.create_image_matching_dataset(ctx)
        import sys; sys.exit(1)

        s_transformations = timer()
        _, num_pairs = classifier.calculate_transformations(ctx)
        e_transformations = timer()

        s_photometric_errors = timer()
        classifier.calculate_photometric_errors(ctx)
        e_photometric_errors = timer()

        s_image_keypts = timer()
        classifier.output_image_keypoints(ctx)
        e_image_keypts = timer()

        s_consistency = timer()
        classifier.calculate_consistency_errors(ctx)
        e_consistency = timer()

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

        s_lccs = timer()
        classifier.calculate_lccs(ctx)
        e_lccs = timer()

        s_shortest_paths = timer()
        classifier.calculate_shortest_paths(ctx)
        e_shortest_paths = timer()

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
            "consistency_errors_wall_time": e_consistency - s_consistency,
            "spatial_entropies_wall_time": e_spatial_entropies - s_spatial_entropies,
            "color_histograms_wall_time": e_color_histograms - s_color_histograms,
            "photometric_errors_wall_time": e_photometric_errors - s_photometric_errors,
            "transformations_wall_time": e_transformations - s_transformations,
            "nbvs_wall_time": e_nbvs - s_nbvs,
            "lccs_wall_time": e_lccs - s_lccs,
            "feature_matching_dataset_wall_time": e_feature_matching_dataset - s_feature_matching_dataset,
            "image_matching_dataset_wall_time": e_image_matching_dataset - s_image_matching_dataset,
            "total_wall_time": end - start,
        }

        with open(ctx.data.profile_log(), 'a') as fout:
            fout.write('calculate_features: {0}\n'.format(end - start))
        self.write_report(data, report)

    def write_report(self, data, report):
        data.save_report(io.json_dumps(report), 'calculate_features.json')

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

class Context:
    pass
