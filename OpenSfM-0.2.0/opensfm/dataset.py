# -*- coding: utf-8 -*-

import os
import glob
import json
import pickle
import pyquaternion
import gzip

import cv2
import numpy as np
import networkx as nx
import scipy
import six

from opensfm import io
from opensfm import config
from opensfm import context
from pyquaternion import Quaternion 
from PIL import Image

class DataSet:
    """
    Dataset representing directory with images, extracted , feature descriptors (SURF, SIFT), etc.

    Methods to retrieve *base directory* for data file(s) have suffix ``_path``, methods to retrieve path of specified
    data file have suffix ``_file``.
    """
    def __init__(self, data_path):
        """
        Create dataset instance. Empty directories (for EXIF, matches, etc) will be created if they don't exist
        already.

        :param data_path: Path to directory containing dataset
        """
        self.data_path = data_path

        self._load_config()
        self._save_config(self.config)

        # Load list of images.
        image_list_file = os.path.join(self.data_path, 'image_list.txt')
        if os.path.isfile(image_list_file):
            with io.open_rt(image_list_file) as fin:
                lines = fin.read().splitlines()
            self.set_image_list(lines)
        else:
            self.set_image_path(os.path.join(self.data_path, 'images'))

        # Load list of masks if they exist.
        mask_list_file = os.path.join(self.data_path, 'mask_list.txt')
        if os.path.isfile(mask_list_file):
            with open(mask_list_file) as fin:
                lines = fin.read().splitlines()
            self.set_mask_list(lines)
        else:
            self.set_mask_path(os.path.join(self.data_path, 'masks'))

    def _load_config(self):
        config_file = os.path.join(self.data_path, 'config.yaml')
        self.config = config.load_config(config_file)

    def _save_config(self, config_):
        config_file = os.path.join(self.data_path, 'config.yaml')
        config.save_config(config_file, config_)

    def images(self):
        """Return list of file names of all images in this dataset"""
        return self.image_list

    def __image_file(self, image):
        """
        Return path of image with given name
        :param image: Image file name (**with extension**)
        """
        return self.image_files[image]

    def load_image(self, image):
        return open(self.__image_file(image), 'rb')

    def image_as_array(self, image):
        """Return image pixels as 3-dimensional numpy array (R G B order)"""
        return io.imread(self.__image_file(image))

    def _undistorted_image_path(self):
        return os.path.join(self.data_path, 'undistorted')

    def _undistorted_image_file(self, image):
        """Path of undistorted version of an image."""
        return os.path.join(self._undistorted_image_path(), image + '.jpg')

    def undistorted_image_as_array(self, image):
        """Undistorted image pixels as 3-dimensional numpy array (R G B order)"""
        return io.imread(self._undistorted_image_file(image))

    def save_undistorted_image(self, image, array):
        io.mkdir_p(self._undistorted_image_path())
        cv2.imwrite(self._undistorted_image_file(image), array[:, :, ::-1])

    def __processed_image_file(self, image, image1_pair, image2_pair):
        return os.path.join(self.__processed_image_path(), '{}---{}-{}-processed.png'.format(image, image1_pair, image2_pair))

    def __resized_image_file(self, image):
        return os.path.join(self.__resized_image_path(), image) # extension is part of the file name

    def __blurred_image_file(self, image, kernel_size):
        return os.path.join(self.__blurred_image_path(), '{}-{}'.format(kernel_size, image) ) # extension is part of the file name

    def masks(self):
        """Return list of file names of all masks in this dataset"""
        return self.mask_list

    def mask_as_array(self, image):
        """Given an image, returns the associated mask as an array if it exists, otherwise returns None"""
        mask_name = image + '.png'
        if mask_name in self.masks():
            mask_path = self.mask_files[mask_name]
            mask = cv2.imread(mask_path)
            if len(mask.shape) == 3:
                mask = mask.max(axis=2)
        else:
            mask = None
        return mask

    def _depthmap_path(self):
        return os.path.join(self.data_path, 'depthmaps')

    def _depthmap_file(self, image, suffix):
        """Path to the depthmap file"""
        return os.path.join(self._depthmap_path(), image + '.' + suffix)

    def raw_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'raw.npz'))

    def save_raw_depthmap(self, image, depth, plane, score, nghbr, nghbrs):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'raw.npz')
        np.savez_compressed(filepath, depth=depth, plane=plane, score=score, nghbr=nghbr, nghbrs=nghbrs)

    def load_raw_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'raw.npz'))
        return o['depth'], o['plane'], o['score'], o['nghbr'], o['nghbrs']

    def clean_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'clean.npz'))

    def save_clean_depthmap(self, image, depth, plane, score):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'clean.npz')
        np.savez_compressed(filepath, depth=depth, plane=plane, score=score)

    def load_clean_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'clean.npz'))
        return o['depth'], o['plane'], o['score']

    def pruned_depthmap_exists(self, image):
        return os.path.isfile(self._depthmap_file(image, 'pruned.npz'))

    def save_pruned_depthmap(self, image, points, normals, colors):
        io.mkdir_p(self._depthmap_path())
        filepath = self._depthmap_file(image, 'pruned.npz')
        np.savez_compressed(filepath,
                            points=points, normals=normals, colors=colors)

    def load_pruned_depthmap(self, image):
        o = np.load(self._depthmap_file(image, 'pruned.npz'))
        return o['points'], o['normals'], o['colors']

    @staticmethod
    def __is_image_file(filename):
        return filename.split('.')[-1].lower() in {'jpg', 'jpeg', 'png', 'tif', 'tiff', 'pgm', 'pnm', 'gif'}

    def set_image_path(self, path):
        """Set image path and find all images in there"""
        self.image_list = []
        self.image_files = {}
        if os.path.exists(path):
            for name in os.listdir(path):
                name = six.text_type(name)
                if self.__is_image_file(name):
                    self.image_list.append(name)
                    self.image_files[name] = os.path.join(path, name)

    def set_image_list(self, image_list):
            self.image_list = []
            self.image_files = {}
            for line in image_list:
                path = os.path.join(self.data_path, line)
                name = os.path.basename(path)
                self.image_list.append(name)
                self.image_files[name] = path

    @staticmethod
    def __is_mask_file(filename):
        return DataSet.__is_image_file(filename)

    def set_mask_path(self, path):
        """Set mask path and find all masks in there"""
        self.mask_list = []
        self.mask_files = {}
        if os.path.exists(path):
            for name in os.listdir(path):
                if self.__is_mask_file(name):
                    self.mask_list.append(name)
                    self.mask_files[name] = os.path.join(path, name)

    def set_mask_list(self, mask_list):
            self.mask_list = []
            self.mask_files = {}
            for line in mask_list:
                path = os.path.join(self.data_path, line)
                name = os.path.basename(path)
                self.mask_list.append(name)
                self.mask_files[name] = path

    def __exif_path(self):
        """Return path of extracted exif directory"""
        return os.path.join(self.data_path, 'exif')

    def __exif_file(self, image):
        """
        Return path of exif information for given image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self.__exif_path(), image + '.exif')

    def load_exif(self, image):
        """
        Return extracted exif information, as dictionary, usually with fields:

        ================  =====  ===================================
        Field             Type   Description
        ================  =====  ===================================
        width             int    Width of image, in pixels
        height            int    Height of image, in pixels
        focal_prior       float  Focal length (real) / sensor width
        ================  =====  ===================================

        :param image: Image name, with extension (i.e. 123.jpg)
        """
        with io.open_rt(self.__exif_file(image)) as fin:
            return json.load(fin)

    def save_exif(self, image, data):
        io.mkdir_p(self.__exif_path())
        with io.open_wt(self.__exif_file(image)) as fout:
            io.json_dump(data, fout)

    def feature_type(self):
        """Return the type of local features (e.g. AKAZE, SURF, SIFT)"""
        feature_name = self.config['feature_type'].lower()
        if self.config['feature_root']:
            feature_name = 'root_' + feature_name
        return feature_name

    def __feature_path(self):
        """Return path of feature descriptors and FLANN indices directory"""
        return os.path.join(self.data_path, "features")

    def __feature_file(self, image):
        """
        Return path of feature file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self.__feature_path(), image + '.npz')

    def __save_features(self, filepath, image, points, descriptors, colors=None):
        io.mkdir_p(self.__feature_path())
        feature_type = self.config['feature_type']
        if ((feature_type == 'AKAZE' and self.config['akaze_descriptor'] in ['MLDB_UPRIGHT', 'MLDB'])
                or (feature_type == 'HAHOG' and self.config['hahog_normalize_to_uchar'])
                or (feature_type == 'ORB')):
            feature_data_type = np.uint8
        else:
            feature_data_type = np.float32
        np.savez_compressed(filepath,
                            points=points.astype(np.float32),
                            descriptors=descriptors.astype(feature_data_type),
                            colors=colors)

    def features_exist(self, image):
        return os.path.isfile(self.__feature_file(image))

    def load_features(self, image):
        feature_type = self.config['feature_type']
        s = np.load(self.__feature_file(image))
        if feature_type == 'HAHOG' and self.config['hahog_normalize_to_uchar']:
            descriptors = s['descriptors'].astype(np.float32)
        else:
            descriptors = s['descriptors']
        return s['points'], descriptors, s['colors'].astype(float)

    def save_features(self, image, points, descriptors, colors):
        self.__save_features(self.__feature_file(image), image, points, descriptors, colors)

    def feature_index_exists(self, image):
        return os.path.isfile(self.__feature_index_file(image))

    def __feature_index_file(self, image):
        """
        Return path of FLANN index file for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self.__feature_path(), image + '.flann')

    def load_feature_index(self, image, features):
        index = context.flann_Index()
        index.load(features, self.__feature_index_file(image))
        return index

    def save_feature_index(self, image, index):
        index.save(self.__feature_index_file(image))

    def __preemptive_features_file(self, image):
        """
        Return path of preemptive feature file (a short list of the full feature file)
        for specified image
        :param image: Image name, with extension (i.e. 123.jpg)
        """
        return os.path.join(self.__feature_path(), image + '_preemptive' + '.npz')

    def load_preemtive_features(self, image):
        s = np.load(self.__preemptive_features_file(image))
        return s['points'], s['descriptors']

    def save_preemptive_features(self, image, points, descriptors):
        self.__save_features(self.__preemptive_features_file(image), image, points, descriptors)

    def __matches_path(self):
        """Return path of matches directory"""
        return os.path.join(self.data_path, 'matches')

    def __rmatches_secondary_path(self):
        """Return path of matches directory"""
        return os.path.join(self.data_path, 'rmatches_secondary')

    def __all_matches_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'all_matches')

    def __weighted_matches_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'weighted_matches')

    def __unthresholded_matches_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'unthresholded_matches')

    def __pairwise_results_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'pairwise_results')

    def __classifier_features_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'classifier_features')

    def __classifier_features_transformation_path(self):
        return os.path.join(self.__classifier_features_path(), 'transformations')
    
    def __classifier_features_spatial_entropies_path(self):
        return os.path.join(self.__classifier_features_path(), 'spatial_entropies')

    def __classifier_features_nbvs_path(self):
        return os.path.join(self.__classifier_features_path(), 'nbvs')

    def __classifier_features_color_histogram_path(self):
        return os.path.join(self.__classifier_features_path(), 'color_histograms')

    def __classifier_features_shortest_paths_path(self):
        return os.path.join(self.__classifier_features_path(), 'shortest_paths')

    def __classifier_features_closest_images_path(self):
        return os.path.join(self.__classifier_features_path(), 'closest_images')

    def __feature_matching_results_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'feature_matching_results')
    
    def __classifier_dataset_path(self):
        return os.path.join(self.data_path, 'classifier_dataset')

    def __yan_path(self):
        return os.path.join(self.data_path, 'yan')

    def __results_path(self):
        return os.path.join(self.data_path, 'results')

    def __processed_image_path(self):
        return os.path.join(self.data_path, 'images-resized-processed')

    def __resized_image_path(self):
        return os.path.join(self.data_path, 'images-resized')

    def __blurred_image_path(self):
        return os.path.join(self.data_path, 'images-blurred')

    def __classifier_dataset_unthresholded_matches_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_matches')
    
    def __classifier_dataset_unthresholded_inliers_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_inliers')

    def __classifier_dataset_unthresholded_outliers_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_outliers')

    def __classifier_dataset_unthresholded_features_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_features')

    def __classifier_features_match_map_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.__classifier_features_path(), 'match_maps')

    def __classifier_features_feature_map_path(self):
        return os.path.join(self.__classifier_features_path(), 'feature_maps')

    def __classifier_features_track_map_path(self):
        return os.path.join(self.__classifier_features_path(), 'track_maps')

    def __classifier_features_photometric_errors_map_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.__classifier_features_path(), 'pe_maps')

    def __classifier_features_consistency_errors_path(self):
        return os.path.join(self.__classifier_features_path(), 'consistency_errors')

    def __classifier_features_lccs_path(self):
        return os.path.join(self.__classifier_features_path(), 'lccs')
    # def __classifier_features_photometric_errors_triangle_transformations_path(self):
    #     return os.path.join(self.__classifier_features_path(), 'pe_triangle_transformations_preprocessed')

    def __classifier_features_sequence_ranks_path(self):
        return os.path.join(self.__classifier_features_path(), 'sequence_ranks')

    def __classifier_features_graph_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.__classifier_features_path(), 'graphs')

    def __match_map_file(self, image):
        """File for matches for an image"""
        return os.path.join(self.__classifier_features_match_map_path(), '{}.png'.format(image))

    def __feature_map_file(self, image):
        """File for matches for an image"""
        return os.path.join(self.__classifier_features_feature_map_path(), '{}.png'.format(image))

    def __track_map_file(self, image):
        """File for matches for an image"""
        return os.path.join(self.__classifier_features_track_map_path(), '{}.png'.format(image))

    def __photometric_errors_map_file(self, image):
        """File for matches for an image"""
        return os.path.join(self.__classifier_features_photometric_errors_map_path(), '{}.png'.format(image))
    
    # def __photometric_error_triangle_transformations_file(self, im1, im2):
    #     return os.path.join(self.__classifier_features_photometric_errors_triangle_transformations_path(), '{}--{}.json'.format(im1, im2))

    def __matches_file(self, image):
        """File for matches for an image"""
        return os.path.join(self.__matches_path(), '{}_matches.pkl.gz'.format(image))

    def __rmatches_secondary_file(self, image):
        return os.path.join(self.__rmatches_secondary_path(), '{}_rmatches_secondary.pkl.gz'.format(image))

    def __graph_file(self, graph_label, edge_threshold):
        return os.path.join(self.__classifier_features_graph_path(), 'graph-{}-t-{}.gpickle'.format(graph_label, edge_threshold))

    def __all_matches_file(self, image):
        """File for all matches for an image"""
        return os.path.join(self.__all_matches_path(), '{}_matches.pkl.gz'.format(image))

    def __weighted_matches_file(self, image):
        """File for all matches for an image"""
        return os.path.join(self.__weighted_matches_path(), '{}_matches.pkl.gz'.format(image))

    def __unthresholded_matches_file(self, image):
        """File for all matches for an image"""
        return os.path.join(self.__unthresholded_matches_path(), '{}_matches.pkl.gz'.format(image))

    def __matches_flags_file(self, image):
        """File for matches flags for an image"""
        return os.path.join(self.__all_matches_path(), '{}_matches_flags.pkl.gz'.format(image))

    def __all_robust_matches_file(self, image):
        """File for all matches for an image"""
        return os.path.join(self.__all_matches_path(), '{}_robust_matches.pkl.gz'.format(image))

    def __transformations_file(self, image):
        """File for transformations for an image (w.r.t. other images)"""
        return os.path.join(self.__pairwise_results_path(), '{}_transformations.pkl.gz'.format(image))

    def __fundamentals_file(self, image):
        """File for fundamental matrices for an image (w.r.t. other images)"""
        return os.path.join(self.__pairwise_results_path(), '{}_fundamentals.pkl.gz'.format(image))

    def __valid_inliers_file(self, image):
        """File for flags indicating valid inliers for an image"""
        return os.path.join(self.__pairwise_results_path(), '{}_valid_inliers.pkl.gz'.format(image))

    def __calibration_flags_file(self, image):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__pairwise_results_path(), '{}_calibration_flags.pkl.gz'.format(image))

    def __feature_transformations_file(self, im, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_transformation_path(), '{}_transformations.{}'.format(im, ext))

    def __feature_lccs_file(self, im, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_lccs_path(), '{}_lccs.{}'.format(im, ext))

    def __feature_triplet_errors_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'triplet_errors.{}'.format(ext))

    def __feature_triplet_pairwise_errors_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'triplet_pairwise_errors.{}'.format(ext))

    def __feature_consistency_errors_file(self, im, cutoff, edge_threshold, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_consistency_errors_path(), '{}_consistency_errors_{}_t-{}.{}'.format(im, cutoff, edge_threshold, ext))

    def __feature_sequence_ranks_file(self, im, ext='pkl.gz'):
        return os.path.join(self.__classifier_features_sequence_ranks_path(), '{}_sequence_ranks.{}'.format(im, ext))

    def __feature_spatial_entropies_file(self, im, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_spatial_entropies_path(), '{}_spatial_entropies.{}'.format(im, ext))

    def __feature_shortest_paths_file(self, im, label, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_shortest_paths_path(), '{}_shortest_paths-{}.{}'.format(im, label, ext))

    def __feature_color_histogram_file(self, im, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_color_histogram_path(), '{}_color_histogram.{}'.format(im, ext))

    # def __feature_photometric_errors_file(self, ext='pkl.gz'):
    #     """File for flags indicating whether calibrated robust matching occured"""
    #     return os.path.join(self.__classifier_features_path(), 'pe_preprocessed_unfiltered.{}'.format(ext))

    # def __feature_secondary_motion_results_file(self, ext='pkl.gz'):
    #     """File for flags indicating whether calibrated robust matching occured"""
    #     return os.path.join(self.__classifier_features_path(), 'secondary_motion_results.{}'.format(ext))

    def __feature_nbvs_file(self, im, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_nbvs_path(), '{}_nbvs.{}'.format(im, ext))

    def __feature_closest_images(self, im, label=None, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        if label is not None:
            return os.path.join(self.__classifier_features_closest_images_path(), '{}_closest_images-{}.{}'.format(im, label, ext))
        return os.path.join(self.__classifier_features_closest_images_path(), '{}_closest_images.{}'.format(im, ext))

    def __feature_image_matching_results_file(self, ext='pkl.gz', suffix=15):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'image_matching_results_{}.{}'.format(suffix, ext))
        # return os.path.join(self.__classifier_features_path(), 'image_matching_results.{}'.format(ext))

    def __feature_matching_results_file(self, image, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__feature_matching_results_path(), '{}_fmr.{}'.format(image, ext))

    def __unthresholded_matches_file(self, image):
        return os.path.join(self.__classifier_dataset_unthresholded_matches_path(), '{}_matches.pkl.gz'.format(image))

    def __unthresholded_inliers_file(self, image):
        return os.path.join(self.__classifier_dataset_unthresholded_inliers_path(), '{}_inliers.pkl.gz'.format(image))

    def __unthresholded_outliers_file(self, image):
        return os.path.join(self.__classifier_dataset_unthresholded_outliers_path(), '{}_outliers.pkl.gz'.format(image))

    def __unthresholded_features_file(self, image):
        return os.path.join(self.__classifier_dataset_unthresholded_features_path(), '{}_features.pkl.gz'.format(image))

    def __feature_matching_dataset_file(self, suffix):
        return os.path.join(self.__classifier_dataset_path(), 'feature_matching_dataset_{}.csv'.format(suffix))

    def __image_matching_dataset_file(self, suffix):
        return os.path.join(self.__classifier_dataset_path(), 'image_matching_dataset_{}.csv'.format(suffix))

    def __ate_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'ate_results.{}'.format(ext))

    def __rpe_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'rpe_results.{}'.format(ext))

    def __match_graph_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'match_graph_results.{}'.format(ext))

    def __reconstruction_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'reconstruction_results.{}'.format(ext))

    def __resectioning_order_file(self, run):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'resectioning_order-{}'.format(run))

    def __resectioning_order_attempted_file(self, run):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'resectioning_order_attempted-{}'.format(run))

    def __resectioning_order_common_tracks_file(self, run):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'resectioning_order_common_tracks-{}'.format(run))

    def __iconic_image_list_file(self, ext):
        return os.path.join(self.__yan_path(), 'iconic_images.{}'.format(ext))

    def __non_iconic_image_list_file(self, ext):
        return os.path.join(self.__yan_path(), 'non_iconic_images.{}'.format(ext))

    def graph_exists(self, graph_label, edge_threshold):
        return os.path.isfile(self.__graph_file(graph_label, edge_threshold))

    def load_graph(self, graph_label, edge_threshold):
        return nx.read_gpickle(self.__graph_file(graph_label, edge_threshold))

    def save_graph(self, G, graph_label, edge_threshold):
        io.mkdir_p(self.__classifier_features_graph_path())
        nx.write_gpickle(G, self.__graph_file(graph_label, edge_threshold))

    def save_shortest_paths(self, im, shortest_paths, label):
        io.mkdir_p(self.__classifier_features_shortest_paths_path())
        with gzip.open(self.__feature_shortest_paths_file(im, label=label, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(shortest_paths, fout)
        with open(self.__feature_shortest_paths_file(im, label=label, ext='json'), 'w') as fout:
            json.dump(shortest_paths, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_shortest_paths(self, im, label):
        with gzip.open(self.__feature_shortest_paths_file(im, label=label), 'rb') as fin:
            shortest_paths = pickle.load(fin)
        return shortest_paths
    
    def shortest_paths_exists(self, im, label):
        return os.path.isfile(self.__feature_shortest_paths_file(im, label=label))

    def matches_exists(self, image):
        return os.path.isfile(self.__matches_file(image))

    def load_matches(self, image):
        with gzip.open(self.__matches_file(image), 'rb') as fin:
            matches = pickle.load(fin)
        return matches

    def rmatches_secondary_exists(self, image):
        return os.path.isfile(self.__rmatches_secondary_file(image))

    def load_rmatches_secondary(self, image):
        with gzip.open(self.__rmatches_secondary_file(image), 'rb') as fin:
            rmatches_secondary = pickle.load(fin)
        return rmatches_secondary

    def load_weighted_matches(self, image):
        with gzip.open(self.__weighted_matches_file(image), 'rb') as fin:
            matches = pickle.load(fin)
        return matches

    def load_all_matches(self, image):
        try:
            with gzip.open(self.__all_matches_file(image), 'rb') as fin:
                matches = pickle.load(fin)
        except:
            matches = None

        try:
            with gzip.open(self.__matches_flags_file(image), 'rb') as fin:
                flags = pickle.load(fin)
        except:
            flags = None

        try:
            with gzip.open(self.__all_robust_matches_file(image), 'rb') as fin:
                robust_matches = pickle.load(fin)
        except:
            robust_matches = None

        return matches, flags, robust_matches

    def all_matches_exists(self, image):
        return os.path.isfile(self.__all_matches_file(image)) and \
            os.path.isfile(self.__matches_flags_file(image)) and \
            os.path.isfile(self.__all_robust_matches_file(image))

    def load_pairwise_results(self, image):
        with gzip.open(self.__transformations_file(image), 'rb') as fin:
            Ts = pickle.load(fin)
        with gzip.open(self.__fundamentals_file(image), 'rb') as fin:
            Fs = pickle.load(fin)
        with gzip.open(self.__valid_inliers_file(image), 'rb') as fin:
            valid_inliers = pickle.load(fin)
        with gzip.open(self.__calibration_flags_file(image), 'rb') as fin:
            calibration_flags = pickle.load(fin)
        return Ts, Fs, valid_inliers, calibration_flags

    def save_matches(self, image, matches):
        io.mkdir_p(self.__matches_path())
        with gzip.open(self.__matches_file(image), 'wb') as fout:
            pickle.dump(matches, fout)

    def save_rmatches_secondary(self, image, rmatches_secondary):
        io.mkdir_p(self.__rmatches_secondary_path())
        with gzip.open(self.__rmatches_secondary_file(image), 'wb') as fout:
            pickle.dump(rmatches_secondary, fout)

    def save_weighted_matches(self, image, matches):
        io.mkdir_p(self.__weighted_matches_path())
        with gzip.open(self.__weighted_matches_file(image), 'wb') as fout:
            pickle.dump(matches, fout)

    def save_all_matches(self, image, matches, flags, robust_matches):
        io.mkdir_p(self.__all_matches_path())
        with gzip.open(self.__all_matches_file(image), 'wb') as fout:
            pickle.dump(matches, fout)
        with gzip.open(self.__matches_flags_file(image), 'wb') as fout:
            pickle.dump(flags, fout)
        with gzip.open(self.__all_robust_matches_file(image), 'wb') as fout:
            pickle.dump(robust_matches, fout)
    
    def save_pairwise_results(self, image, Ts, Fs, valid_inliers, calibration_flags):
        io.mkdir_p(self.__pairwise_results_path())
        with gzip.open(self.__transformations_file(image), 'wb') as fout:
            pickle.dump(Ts, fout)
        with gzip.open(self.__fundamentals_file(image), 'wb') as fout:
            pickle.dump(Fs, fout)
        with gzip.open(self.__valid_inliers_file(image), 'wb') as fout:
            pickle.dump(valid_inliers, fout)
        with gzip.open(self.__calibration_flags_file(image), 'wb') as fout:
            pickle.dump(calibration_flags, fout)

    def transformations_exists(self, im):
        return os.path.isfile(self.__feature_transformations_file(im))

    def save_transformations(self, im, transformations):
        io.mkdir_p(self.__classifier_features_transformation_path())
        with gzip.open(self.__feature_transformations_file(im, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(transformations, fout)
        with open(self.__feature_transformations_file(im, ext='json'), 'w') as fout:
            json.dump(transformations, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_transformations(self, im):
        with gzip.open(self.__feature_transformations_file(im), 'rb') as fin:
            transformations = pickle.load(fin)
        return transformations

    def save_lccs(self, im, lccs):
        io.mkdir_p(self.__classifier_features_lccs_path())
        with gzip.open(self.__feature_lccs_file(im, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(lccs, fout)
        with open(self.__feature_lccs_file(im, ext='json'), 'w') as fout:
            json.dump(lccs, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_lccs(self, im):
        with gzip.open(self.__feature_lccs_file(im), 'rb') as fin:
            lccs = pickle.load(fin)
        return lccs

    def lccs_exists(self, im):
        return os.path.isfile(self.__feature_lccs_file(im))

    def save_triplet_errors(self, triplet_errors):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_triplet_errors_file('pkl.gz'), 'wb') as fout:
            pickle.dump(triplet_errors, fout)
        with open(self.__feature_triplet_errors_file('json'), 'w') as fout:
            json.dump(triplet_errors, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def triplet_errors_exists(self):
        return os.path.isfile(self.__feature_triplet_errors_file()) and \
            os.path.isfile(self.__feature_triplet_pairwise_errors_file())

    def save_triplet_pairwise_errors(self, triplet_pairwise_errors):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_triplet_pairwise_errors_file('pkl.gz'), 'wb') as fout:
            pickle.dump(triplet_pairwise_errors, fout)
        with open(self.__feature_triplet_pairwise_errors_file('json'), 'w') as fout:
            json.dump(triplet_pairwise_errors, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_triplet_pairwise_errors(self):
        with gzip.open(self.__feature_triplet_pairwise_errors_file(), 'rb') as fin:
            triplet_pairwise_errors = pickle.load(fin)
        return triplet_pairwise_errors

    def consistency_errors_exists(self, im, cutoff, edge_threshold):
        return os.path.isfile(self.__feature_consistency_errors_file(im, cutoff=cutoff, edge_threshold=edge_threshold, ext='pkl.gz')) and \
            os.path.isfile(self.__feature_consistency_errors_file(im, cutoff=cutoff, edge_threshold=edge_threshold, ext='json'))
        # return os.path.isfile(self.__feature_consistency_errors_file(cutoff=cutoff, edge_threshold=edge_threshold, ext='pkl.gz'))

    def save_consistency_errors(self, im, consistency_errors, cutoff, edge_threshold):
        io.mkdir_p(self.__classifier_features_consistency_errors_path())
        with gzip.open(self.__feature_consistency_errors_file(im, cutoff=cutoff, edge_threshold=edge_threshold, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(consistency_errors, fout)
        with open(self.__feature_consistency_errors_file(im, cutoff=cutoff, edge_threshold=edge_threshold, ext='json'), 'w') as fout:
            json.dump(consistency_errors, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_consistency_errors(self, im, cutoff, edge_threshold):
        with gzip.open(self.__feature_consistency_errors_file(im, cutoff=cutoff, edge_threshold=edge_threshold, ext='pkl.gz'), 'rb') as fin:
            consistency_errors = pickle.load(fin)
        return consistency_errors

    def save_sequence_ranks(self, im, sequence_ranks):
        io.mkdir_p(self.__classifier_features_sequence_ranks_path())
        with gzip.open(self.__feature_sequence_ranks_file(im, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(sequence_ranks, fout)
        with open(self.__feature_sequence_ranks_file(im, ext='json'), 'w') as fout:
            json.dump(sequence_ranks, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_sequence_ranks(self, im):
        with gzip.open(self.__feature_sequence_ranks_file(im), 'rb') as fin:
            sequence_ranks = pickle.load(fin)
        return sequence_ranks

    def sequence_ranks_exists(self, im):
        return os.path.isfile(self.__feature_sequence_ranks_file(im))

    def save_spatial_entropies(self, im, spatial_entropies):
        io.mkdir_p(self.__classifier_features_spatial_entropies_path())
        with gzip.open(self.__feature_spatial_entropies_file(im, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(spatial_entropies, fout)
        with open(self.__feature_spatial_entropies_file(im, ext='json'), 'w') as fout:
            json.dump(spatial_entropies, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def spatial_entropies_exists(self, im):
        return os.path.isfile(self.__feature_spatial_entropies_file(im))

    def load_spatial_entropies(self, im):
        with gzip.open(self.__feature_spatial_entropies_file(im), 'rb') as fin:
            spatial_entropies = pickle.load(fin)
        return spatial_entropies

    def save_match_map(self, image, match_map):
        io.mkdir_p(self.__classifier_features_match_map_path())
        scipy.misc.imsave(self.__match_map_file(image), match_map)
        # resized_match_map = cv2.resize(match_map, (224, 224))
        # cv2.imwrite(self.__match_map_file(image), resized_match_map)

    def save_feature_map(self, image, feature_map):
        io.mkdir_p(self.__classifier_features_feature_map_path())
        scipy.misc.imsave(self.__feature_map_file(image), feature_map)
        # resized_match_map = cv2.resize(match_map, (224, 224))
        # cv2.imwrite(self.__match_map_file(image), resized_match_map)
    
    def all_feature_maps(self):
        # list all images based on feature maps (local machine doesn't have images, but will have feature maps)
        images = []
        for f in glob.glob(self.__classifier_features_feature_map_path() + '/*.png'):
            images.append(os.path.basename(f).split('---')[1][:-4])
        return images

    def save_track_map(self, image, track_map):
        io.mkdir_p(self.__classifier_features_track_map_path())
        scipy.misc.imsave(self.__track_map_file(image), track_map)

    def save_color_histogram(self, im, color_histogram):
        io.mkdir_p(self.__classifier_features_color_histogram_path())
        with gzip.open(self.__feature_color_histogram_file(im, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(color_histogram, fout)
        with open(self.__feature_color_histogram_file(im, ext='json'), 'w') as fout:
            json.dump(color_histogram, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def color_histogram_exists(self, im):
        return os.path.isfile(self.__feature_color_histogram_file(im))

    def load_color_histogram(self, im):
        with gzip.open(self.__feature_color_histogram_file(im), 'rb') as fin:
            color_histogram = pickle.load(fin)
        return color_histogram

    # def save_secondary_motion_results(self, results):
    #     io.mkdir_p(self.__classifier_features_path())
    #     with gzip.open(self.__feature_secondary_motion_results_file('pkl.gz'), 'wb') as fout:
    #         pickle.dump(results, fout)
    #     with open(self.__feature_secondary_motion_results_file('json'), 'w') as fout:
    #         json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    # def save_photometric_errors(self, photometric_errors):
    #     io.mkdir_p(self.__classifier_features_path())
    #     with gzip.open(self.__feature_photometric_errors_file('pkl.gz'), 'wb') as fout:
    #         pickle.dump(photometric_errors, fout)
    #     with open(self.__feature_photometric_errors_file('json'), 'w') as fout:
    #         json.dump(photometric_errors, fout, sort_keys=True, indent=4, separators=(',', ': '))

    # def load_photometric_errors(self):
    #     with gzip.open(self.__feature_photometric_errors_file(), 'rb') as fin:
    #         photometric_errors = pickle.load(fin)
    #     return photometric_errors

    def save_photometric_errors_map(self, image, photometric_errors_map, size=None):
        io.mkdir_p(self.__classifier_features_photometric_errors_map_path())
        # scipy.misc.imsave(self.__photometric_errors_map_file(image), photometric_errors_map)

        if size is not None:
            resized_photometric_errors_map = cv2.resize(photometric_errors_map, (size, size))
        else:
            resized_photometric_errors_map = photometric_errors_map
        cv2.imwrite(self.__photometric_errors_map_file(image), resized_photometric_errors_map)

    def load_photometric_errors_map(self, image, grayscale=False):
        if grayscale:
            photometric_errors_map = cv2.imread(self.__photometric_errors_map_file(image), cv2.IMREAD_GRAYSCALE)
        else:
            photometric_errors_map = cv2.imread(self.__photometric_errors_map_file(image))
        return photometric_errors_map

    def photometric_errors_map_exists(self, image):
        return os.path.isfile(self.__photometric_errors_map_file(image))    

    # def photometric_errors_exists(self):
    #     return os.path.isfile(self.__feature_photometric_errors_file())

    # def secondary_motion_results_exists(self):
    #     return os.path.isfile(self.__feature_secondary_motion_results_file())

    def save_nbvs(self, im1, nbvs):
        io.mkdir_p(self.__classifier_features_nbvs_path())
        with gzip.open(self.__feature_nbvs_file(im1, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(nbvs, fout)
        with open(self.__feature_nbvs_file(im1, ext='json'), 'w') as fout:
            json.dump(nbvs, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def nbvs_exists(self, im1):
        return os.path.isfile(self.__feature_nbvs_file(im1))

    def load_nbvs(self, im1):
        with gzip.open(self.__feature_nbvs_file(im1), 'rb') as fin:
            nbvs = pickle.load(fin)
        return nbvs

    def save_closest_images(self, im, closest_images, label=None):
        io.mkdir_p(self.__classifier_features_closest_images_path())
        with gzip.open(self.__feature_closest_images(im, label=label, ext='pkl.gz'), 'wb') as fout:
            pickle.dump(closest_images, fout)
        with open(self.__feature_closest_images(im, label=label, ext='json'), 'w') as fout:
            json.dump(closest_images, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_closest_images(self, im, label=None):
        with gzip.open(self.__feature_closest_images(im, label=label, ext='pkl.gz'), 'rb') as fin:
            closest_images = pickle.load(fin)
        return closest_images

    def closest_images_exists(self, im, label=None):
        return os.path.isfile(self.__feature_closest_images(im, label=label, ext='pkl.gz'))        

    def save_image_matching_results(self, results, robust_matches_threshold, classifier):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_image_matching_results_file(ext='pkl.gz', suffix='{}-{}'.format(robust_matches_threshold, classifier)), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__feature_image_matching_results_file(ext='json', suffix='{}-{}'.format(robust_matches_threshold, classifier)), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def save_feature_matching_results(self, image, results):
        io.mkdir_p(self.__feature_matching_results_path())
        with gzip.open(self.__feature_matching_results_file(image, 'pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__feature_matching_results_file(image, 'json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_image_matching_results(self, robust_matches_threshold, classifier):
        with gzip.open(self.__feature_image_matching_results_file(ext='pkl.gz', suffix='{}-{}'.format(robust_matches_threshold, classifier)), 'rb') as fin:
            results = pickle.load(fin)
        return results

    def load_feature_matching_results(self, image):
        with gzip.open(self.__feature_matching_results_file(image), 'rb') as fin:
            results = pickle.load(fin)
        return results

    def save_iconic_image_list(self, image_list):
        io.mkdir_p(self.__yan_path())
        with open(self.__iconic_image_list_file('json'), 'w') as fout:
            json.dump(image_list, fout, sort_keys=True, indent=4, separators=(',', ': '))
    
    def load_iconic_image_list(self):
        with open(self.__iconic_image_list_file('json'), 'r') as fin:
            image_list = json.load(fin)
        return image_list

    def iconic_image_list_exists(self):
        return os.path.isfile(self.__iconic_image_list_file('json'))

    def save_non_iconic_image_list(self, image_list):
        io.mkdir_p(self.__yan_path())
        with open(self.__non_iconic_image_list_file('json'), 'w') as fout:
            json.dump(image_list, fout, sort_keys=True, indent=4, separators=(',', ': '))
    
    def load_non_iconic_image_list(self):
        with open(self.__non_iconic_image_list_file('json'), 'r') as fin:
            image_list = json.load(fin)
        return image_list

    def non_iconic_image_list_exists(self):
        return os.path.isfile(self.__non_iconic_image_list_file('json'))

    def load_groundtruth_image_matching_results(self, image_matching_classifier_thresholds):
        # Load all path lengths
        fns, [R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, num_rmatches, num_matches, spatial_entropy_1_8x8, \
            spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, pe_histogram, pe_polygon_area_percentage, \
            nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
            sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, \
            lcc_im1_15, lcc_im2_15, min_lcc_15, max_lcc_15, \
            lcc_im1_20, lcc_im2_20, min_lcc_20, max_lcc_20, \
            lcc_im1_25, lcc_im2_25, min_lcc_25, max_lcc_25, \
            lcc_im1_30, lcc_im2_30, min_lcc_30, max_lcc_30, \
            lcc_im1_35, lcc_im2_35, min_lcc_35, max_lcc_35, \
            lcc_im1_40, lcc_im2_40, min_lcc_40, max_lcc_40, \
            shortest_path_length, \
            num_gt_inliers, labels] \
            = self.load_image_matching_dataset(robust_matches_threshold=15)

        gt_results = {}
        for idx, _ in enumerate(fns[:,0]):
            im1 = fns[idx,0]
            im2 = fns[idx,1]
            if labels[idx] >= 1.0:
                if im1 not in gt_results:
                    gt_results[im1] = {}
                if im2 not in gt_results:
                    gt_results[im2] = {}

                gt_results[im1][im2] = {"im1": im1, "im2": im2, "score": 1.0, "rmatches": num_rmatches[idx], 'shortest_path_length': shortest_path_length[idx]}
                gt_results[im2][im1] = {"im1": im2, "im2": im1, "score": 1.0, "rmatches": num_rmatches[idx], 'shortest_path_length': shortest_path_length[idx]}
        return gt_results

    def save_unthresholded_matches(self, image, matches):
        io.mkdir_p(self.__classifier_dataset_unthresholded_matches_path())
        with gzip.open(self.__unthresholded_matches_file(image), 'wb') as fout:
            pickle.dump(matches, fout)

    def unthresholded_matches_exists(self, image):
        return os.path.isfile(self.__unthresholded_matches_file(image))

    def load_unthresholded_matches(self, image):
        with gzip.open(self.__unthresholded_matches_file(image), 'rb') as fin:
            matches = pickle.load(fin)
        return matches

    def save_unthresholded_inliers(self, image, inliers):
        io.mkdir_p(self.__classifier_dataset_unthresholded_inliers_path())
        with gzip.open(self.__unthresholded_inliers_file(image), 'wb') as fout:
            pickle.dump(inliers, fout)

    def load_unthresholded_inliers(self, image):
        with gzip.open(self.__unthresholded_inliers_file(image), 'rb') as fin:
            inliers = pickle.load(fin)
        return inliers

    def unthresholded_inliers_exists(self, image):
        return os.path.isfile(self.__unthresholded_inliers_file(image))

    def save_unthresholded_outliers(self, image, outliers):
        io.mkdir_p(self.__classifier_dataset_unthresholded_outliers_path())
        with gzip.open(self.__unthresholded_outliers_file(image), 'wb') as fout:
            pickle.dump(outliers, fout)

    def load_unthresholded_outliers(self, image):
        with gzip.open(self.__unthresholded_outliers_file(image), 'rb') as fin:
            outliers = pickle.load(fin)
        return outliers

    def unthresholded_outliers_exists(self, image):
        return os.path.isfile(self.__unthresholded_outliers_file(image))

    def save_unthresholded_features(self, image, features):
        io.mkdir_p(self.__classifier_dataset_unthresholded_features_path())
        with gzip.open(self.__unthresholded_features_file(image), 'wb') as fout:
            pickle.dump(features, fout)

    def load_unthresholded_features(self, image):
        with gzip.open(self.__unthresholded_features_file(image), 'rb') as fin:
            features = pickle.load(fin)
        return features

    def unthresholded_features_exists(self, image):
        return os.path.isfile(self.__unthresholded_features_file(image))

    def load_general_dataset(self, dataset_fn, load_file_names=True):
        with open(dataset_fn, 'r') as fin:
            data = fin.readlines()
            fns = []
            for i,datum in enumerate(data):
                if i == 0: # header row
                    # Initialize variables
                    header = datum.split(',')
                    # Skip image 1 and image 2 names for columns and header row for rows
                    data_formatted = np.zeros((len(data)-1, len(header)-2)).astype(np.float)
                    continue

                data_split = datum.split(',')
                for di, d in enumerate(data_split):
                    if di == 0 or di == 1:
                        if len(fns) < i:
                            fns.append([None,None])
                        fns[i-1][di] = data_split[di].strip()
                        continue
                    data_formatted[i-1,di-2] = float(d)
        if load_file_names:
            return np.array(fns), data_formatted
        return data_formatted

    def save_feature_matching_dataset(self, lowes_threshold):
        with open(self.__feature_matching_dataset_file(suffix=lowes_threshold), 'w') as fout:
            fout.write('image 1,image 2, index 1, index 2, lowe\'s ratio 1, lowe\'s ratio 2, max reprojection error, size 1, angle 1, size 2, angle 2, reproj error 1, reproj error 2, label\n')
            for im1 in sorted(self.images()):
                # if im1 != 'DSC_1761.JPG':
                #     continue
                if not self.unthresholded_matches_exists(im1):
                    continue

                try:
                    im1_unthresholded_matches = self.load_unthresholded_matches(im1)
                    im1_unthresholded_inliers = self.load_unthresholded_inliers(im1)
                    im1_unthresholded_outliers = self.load_unthresholded_outliers(im1)
                    im1_unthresholded_features = self.load_unthresholded_features(im1)
                except:
                    continue

                for im2 in im1_unthresholded_matches:
                    # if im2 != 'DSC_1804.JPG':
                    #     continue
                    for i,m in enumerate(im1_unthresholded_matches[im2]):
                        if im2 not in im1_unthresholded_features:
                            continue
                        features = im1_unthresholded_features[im2][i]
                        if m in im1_unthresholded_inliers[im2]:
                            label = 1
                        elif m in im1_unthresholded_outliers[im2]:
                            label = -1
                        else:
                            label = 0

                        max_lowes_ratio = max(float(m[2]), float(m[3]))
                        if self.config['matcher_type'] == 'FLANN':
                            if max_lowes_ratio > lowes_threshold**2:
                                continue
                        else:
                            if max_lowes_ratio > lowes_threshold:
                                continue
                        # image 1,image 2, index 1, index 2, lowe\'s ratio 1, lowe\'s ratio 2, max reprojection error, size 1, angle 1, size 2, angle 2, label
                        fout.write(
                            str(im1) + ', ' + \
                            str(im2) + ', ' + \
                            str(int(m[0])) + ', ' + \
                            str(int(m[1])) + ', ' + \
                            str(round(float(m[2]), 3)) + ', ' + \
                            str(round(float(m[3]), 3)) + ', ' + \
                            str(round(max(float(features[0]), float(features[1])), 6)) + ', ' + \
                            str(float(features[4])) + ', ' + \
                            str(float(features[5])) + ', ' + \
                            str(float(features[6])) + ', ' + \
                            str(float(features[7])) + ', ' + \
                            str(float(features[0])) + ', ' + \
                            str(float(features[1])) + ', ' + \
                            str(label) + '\n')

    def load_feature_matching_dataset(self, lowes_threshold):
        fns, data = self.load_general_dataset(self.__feature_matching_dataset_file(suffix=lowes_threshold))
        indices_1 = data[:,0]
        indices_2 = data[:,1]
        lowes_ratio_1 = data[:,2]
        lowes_ratio_2 = data[:,3]
        errors = data[:,4]
        size1 = data[:,5]
        angle1 = data[:,6]
        size2 = data[:,7]
        angle2 = data[:,8]
        rerr1 = data[:,9]
        rerr2 = data[:,10]
        labels = data[:,11]
        return fns, [indices_1, indices_2, lowes_ratio_1, lowes_ratio_2, errors, size1, size2, angle1, angle2, rerr1, \
            rerr2, labels]


    def sequence_rank_adapter(self, options={}):
        # sequence_ranks = self.load_sequence_ranks()
        sequence_rank_scores_mean = {}
        sequence_rank_scores_min = {}
        sequence_rank_scores_max = {}
        sequence_distance_scores = {}
        total_images = len(self.images())

        for im1 in sorted(self.images()):
            if self.sequence_ranks_exists(im1):
                im1_sequence_ranks = self.load_sequence_ranks(im1)
            else:
                im1_sequence_ranks = {}
                for i in self.images():
                    im1_sequence_ranks[i] = {'rank': 0, 'distance': 0}

            if im1 not in sequence_rank_scores_mean:
                sequence_rank_scores_mean[im1] = {}
                sequence_rank_scores_min[im1] = {}
                sequence_rank_scores_max[im1] = {}
                sequence_distance_scores[im1] = {}

            for im2 in im1_sequence_ranks:
                if self.sequence_ranks_exists(im2):
                    im2_sequence_ranks = self.load_sequence_ranks(im2)
                else:
                    im2_sequence_ranks = {}
                    for i in self.images():
                        im2_sequence_ranks[i] = {'rank': 0, 'distance': 0}
                    

                sequence_distance_scores[im1][im2] = \
                    (total_images - im1_sequence_ranks[im2]['distance']) / total_images

                sequence_rank_scores_mean[im1][im2] = \
                    0.5 * (total_images - im1_sequence_ranks[im2]['rank']) / total_images + \
                    0.5 * (total_images - im2_sequence_ranks[im1]['rank']) / total_images
                sequence_rank_scores_min[im1][im2] = min(\
                    (total_images - im1_sequence_ranks[im2]['rank']) / total_images,
                    (total_images - im2_sequence_ranks[im1]['rank']) / total_images
                    )
                sequence_rank_scores_max[im1][im2] = max(\
                    (total_images - im1_sequence_ranks[im2]['rank']) / total_images,
                    (total_images - im2_sequence_ranks[im1]['rank']) / total_images
                    )

        return sequence_rank_scores_mean, sequence_rank_scores_min, sequence_rank_scores_max, sequence_distance_scores

    def save_image_matching_dataset(self, robust_matches_threshold):
        write_header = True
        lowes_threshold = 0.8
        vt_ranks, vt_scores = self.load_vocab_ranks_and_scores()
        sequence_scores_mean, sequence_scores_min, sequence_scores_max, sequence_distance_scores = \
            self.sequence_rank_adapter()

        counter = 0
        with open(self.__image_matching_dataset_file(suffix=robust_matches_threshold), 'w') as fout:
            
            for im1 in sorted(self.images()):
                if not self.transformations_exists(im1):
                    continue

                im_transformations = self.load_transformations(im1)
                im_spatial_entropies = self.load_spatial_entropies(im1)
                im_all_matches, _, im_all_rmatches = self.load_all_matches(im1)

                if self.consistency_errors_exists(im1, cutoff=3, edge_threshold=15):
                    im_consistency_errors = self.load_consistency_errors(im1, cutoff=3, edge_threshold=15)
                else:
                    im_consistency_errors = {}
                    for i in im_all_rmatches:
                        im_consistency_errors[i] = {'histogram-cumsum': np.zeros((80,)).tolist()}

                if self.nbvs_exists(im1):
                    im_nbvs = self.load_nbvs(im1)
                else:
                    im_nbvs = {}
                    for i in im_all_rmatches:
                        im_nbvs[i] = {'nbvs_im1': 0, 'nbvs_im2': 0}

                if self.shortest_paths_exists(im1, 'rm-cost'):
                    im_shortest_paths = self.load_shortest_paths(im1, 'rm-cost')
                else:
                    im_shortest_paths = {}
                    for i in im_all_rmatches:
                        im_shortest_paths[i] = {'shortest_path': []}
                if self.color_histogram_exists(im1):
                    color_histogram_im1 = self.load_color_histogram(im1)
                else:
                    color_histogram_im1 = {'histogram': np.zeros((384,)).tolist()}
                
                if self.lccs_exists(im1):
                    lccs_im1 = self.load_lccs(im1)
                else:
                    lccs_im1 = {}
                    for t in [15, 20, 25, 30, 35, 40]:
                        lccs_im1[t] = 0.0

                if self.closest_images_exists(im1, 'rm-cost-lmds-False'):
                    im1_closest_images = self.load_closest_images(im1, 'rm-cost-lmds-False')
                else:
                    im1_closest_images = []

                if self.closest_images_exists(im1, 'gt-lmds-False'):
                    im1_closest_images_gt = self.load_closest_images(im1, 'gt-lmds-False')
                else:
                    im1_closest_images_gt = []
                
                if self.unthresholded_inliers_exists(im1):
                    im_unthresholded_inliers = self.load_unthresholded_inliers(im1)
                else:
                    continue
                    
                for im2 in im_all_rmatches:#transformations[im1]:
                    if im2 not in im_transformations:
                        continue
                    # print ('{} / {}'.format(im1, im2))
                    # te_histogram = np.array(triplet_pairwise_errors[im1][im2]['histogram-cumsum'])
                    if self.color_histogram_exists(im2):
                        color_histogram_im2 = self.load_color_histogram(im2)
                    else:
                        color_histogram_im2 = {'histogram': np.zeros((384,)).tolist()}

                    if self.lccs_exists(im2):
                        lccs_im2 = self.load_lccs(im2)
                    else:
                        lccs_im2 = {}
                        for t in [15, 20, 25, 30, 35, 40]:
                            lccs_im2[t] = 0.0

                    if self.closest_images_exists(im2, 'rm-cost-lmds-False'):
                        im2_closest_images = self.load_closest_images(im2, 'rm-cost-lmds-False')
                    else:
                        im2_closest_images = []

                    if self.closest_images_exists(im2, 'gt-lmds-False'):
                        im2_closest_images_gt = self.load_closest_images(im2, 'gt-lmds-False')
                    else:
                        im2_closest_images_gt = []
                    
                    if im2 in im_consistency_errors:
                        te_histogram = np.array(im_consistency_errors[im2]['histogram-cumsum'])
                    else:
                        te_histogram = np.zeros((80,))    

                    # mu, sigma = scipy.stats.norm.fit(te_histogram)
                    # te_histogram = np.zeros((len(te_histogram),))
                    # te_histogram[0] = mu
                    # te_histogram[1] = sigma

                    # pe_histogram = np.array(photometric_errors[im1][im2]['histogram-cumsum'])
                    # mu, sigma = scipy.stats.norm.fit(pe_histogram)
                    # pe_histogram = np.zeros((len(pe_histogram),))
                    # pe_histogram[0] = mu
                    # pe_histogram[1] = sigma
                    # if False:
                    #     pe_histogram = np.array(photometric_errors[im1][im2]['histogram-cumsum'])
                    pe_histogram = np.zeros((51,))
                    # mu, sigma = scipy.stats.norm.fit(pe_histogram)
                    # pe_histogram = np.zeros((len(pe_histogram),))
                    # pe_histogram[0] = mu
                    # pe_histogram[1] = sigma

                    R = np.around(np.array(im_transformations[im2]['rotation']), decimals=2)
                    se = im_spatial_entropies[im2]
                    # pe_histogram = ','.join(map(str, np.around(np.array(photometric_errors[im1][im2]['histogram-cumsum']), decimals=2)))
                    pe_histogram = ','.join(map(str, np.around(pe_histogram, decimals=2)))

                    if False:
                        pe_polygon_area_percentage = photometric_errors[im1][im2]['polygon_area_percentage']
                    pe_polygon_area_percentage = 0.0
                    
                    nbvs_im1 = im_nbvs[im2]['nbvs_im1']
                    nbvs_im2 = im_nbvs[im2]['nbvs_im2']
                    te_histogram = ','.join(map(str, np.around(te_histogram, decimals=2)))
                    # if False:
                    ch_im1 = ','.join(map(str, np.around(np.array(color_histogram_im1['histogram']), decimals=2)))
                    ch_im2 = ','.join(map(str, np.around(np.array(color_histogram_im2['histogram']), decimals=2)))
                    # ch_im1 = ','.join(map(str, np.around(np.zeros((384,)), decimals=2)))
                    # ch_im2 = ','.join(map(str, np.around(np.zeros((384,)), decimals=2)))

                    vt_rank_percentage_im1_im2 = 100.0 * vt_ranks[im1][im2] / len(self.images())
                    vt_rank_percentage_im2_im1 = 100.0 * vt_ranks[im2][im1] / len(self.images())
                    
                    if im2 not in im1_closest_images:
                        mds_rank_percentage_im1_im2 = 99.99
                    else:
                        mds_rank_percentage_im1_im2 = 100.0 * im1_closest_images.index(im2) / len(self.images())
                        
                    if im1 not in im2_closest_images:
                        mds_rank_percentage_im2_im1 = 99.99
                    else:
                        mds_rank_percentage_im2_im1 = 100.0 * im2_closest_images.index(im1) / len(self.images())

                    if im2 not in im1_closest_images_gt:
                        distance_rank_percentage_im1_im2_gt = 99.99
                    else:
                        distance_rank_percentage_im1_im2_gt = 100.0 * im1_closest_images_gt.index(im2) / len(self.images())

                    if im1 not in im2_closest_images_gt:
                        distance_rank_percentage_im2_im1_gt = 99.99
                    else:
                        distance_rank_percentage_im2_im1_gt = 100.0 * im2_closest_images_gt.index(im1) / len(self.images())


                    if im2 not in im_unthresholded_inliers or \
                        im_unthresholded_inliers[im2] is None:
                        label = -1
                        num_rmatches = 0
                        num_matches = 0
                        num_thresholded_gt_inliers = 0
                    else:
                        max_lowes_thresholds = np.maximum(im_unthresholded_inliers[im2][:,2], im_unthresholded_inliers[im2][:,3])

                        if self.config['matcher_type'] == 'FLANN':
                            num_thresholded_gt_inliers = len(np.where(max_lowes_thresholds <= lowes_threshold**2)[0])
                        else:
                            num_thresholded_gt_inliers = len(np.where(max_lowes_thresholds <= lowes_threshold)[0])

                        if num_thresholded_gt_inliers < robust_matches_threshold:
                            label = -1
                            num_rmatches = len(im_all_rmatches[im2])
                            num_matches = len(im_all_matches[im2])
                        else:
                            label = 1
                            num_rmatches = len(im_all_rmatches[im2])
                            num_matches = len(im_all_matches[im2])

                    # print ('*'*100)
                    # print ('num_matches: {} / num_rmatches: {} / num_thresholded_gt_inliers: {} / num_unthresholded_gt_inliers: {}'.format(num_matches, num_rmatches, num_thresholded_gt_inliers, len(im_unthresholded_inliers[im2])))
                    # print (max_lowes_thresholds)
                    # print ('*'*100)
                    # counter = counter + 1
                    # if counter > 10:
                    #     import sys; sys.exit(1)
                    if write_header:
                        fout.write('image 1, image 2,\
                            R11, R12, R13, R21, R22, R23, R31, R32, R33,\
                            # of rmatches, # of matches,\
                            spatial entropy 1 8x8, spatial entropy 2 8x8, spatial entropy 1 16x16, spatial entropy 2 16x16,\
                            photometric error histogram {} photometric error area percentage,\
                            colmap score im1, colmap score im2,\
                            triplet error histogram {}\
                            color histogram im1 {} color histogram im2 {}\
                            vt rank percentage im1-im2, vt rank percentage im2-im1,\
                            sq mean, sq min, sq max, sq distance score, \
                            lcc im1 15, lcc im2 15, min lcc 15, max lcc 15, \
                            lcc im1 20, lcc im2 20, min lcc 20, max lcc 20, \
                            lcc im1 25, lcc im2 25, min lcc 25, max lcc 25, \
                            lcc im1 30, lcc im2 30, min lcc 30, max lcc 30, \
                            lcc im1 35, lcc im2 35, min lcc 35, max lcc 35, \
                            lcc im1 40, lcc im2 40, min lcc 40, max lcc 40, \
                            path length, \
                            mds rank percentage im1-im2, mds rank percentage im2-im1,\
                            distance rank percentage im1-im2 gt, distance rank percentage im2-im1 gt,\
                            # of gt inliers, label\n'.format(\
                                ','*len(pe_histogram.split(',')), \
                                ','*len(te_histogram.split(',')), \
                                ','*len(ch_im1.split(',')), \
                                ','*len(ch_im2.split(','))
                                )
                            )
                        write_header = False

                    # import sys; sys.exit(1)
                    # 2 +
                    # 9 +
                    # 2 + 
                    # 4 +
                    # 51 + 1 +
                    # 2 + 
                    # 80 + 
                    # 384 + 
                    # 384 + 
                    # 2 + 
                    # 1 
                    # = 922
                    fout.write(
                        '{}, {}, \
                        {}, {}, {}, {}, {}, {}, {}, {}, {}, \
                        {}, {}, \
                        {}, {}, {}, {}, \
                        {}, {}, \
                        {}, {}, \
                        {}, \
                        {}, {}, \
                        {}, {}, \
                        {}, {}, {}, \
                        {}, \
                        {}, {}, {}, {}, \
                        {}, {}, {}, {}, \
                        {}, {}, {}, {}, \
                        {}, {}, {}, {}, \
                        {}, {}, {}, {}, \
                        {}, {}, {}, {}, \
                        {}, \
                        {}, {}, \
                        {}, {}, \
                        {}, {}\n'.format( \
                        im1, im2, \
                        R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2], \
                        num_rmatches, num_matches, \
                        se['entropy_im1_8'], se['entropy_im2_8'], se['entropy_im1_16'], se['entropy_im2_16'], \
                        pe_histogram, pe_polygon_area_percentage, \
                        nbvs_im1, nbvs_im2, \
                        te_histogram, \
                        ch_im1, ch_im2, \
                        vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
                        sequence_scores_mean[im1][im2], sequence_scores_min[im1][im2], sequence_scores_max[im1][im2], \
                        sequence_distance_scores[im1][im2], \
                        lccs_im1[15], lccs_im2[15], min(lccs_im1[15],lccs_im2[15]), max(lccs_im1[15],lccs_im2[15]), \
                        lccs_im1[20], lccs_im2[20], min(lccs_im1[20],lccs_im2[20]), max(lccs_im1[20],lccs_im2[20]), \
                        lccs_im1[25], lccs_im2[25], min(lccs_im1[25],lccs_im2[25]), max(lccs_im1[25],lccs_im2[25]), \
                        lccs_im1[30], lccs_im2[30], min(lccs_im1[30],lccs_im2[30]), max(lccs_im1[30],lccs_im2[30]), \
                        lccs_im1[35], lccs_im2[35], min(lccs_im1[35],lccs_im2[35]), max(lccs_im1[35],lccs_im2[35]), \
                        lccs_im1[40], lccs_im2[40], min(lccs_im1[40],lccs_im2[40]), max(lccs_im1[40],lccs_im2[40]), \
                        len(im_shortest_paths[im2]["shortest_path"]), \
                        mds_rank_percentage_im1_im2, mds_rank_percentage_im2_im1, \
                        distance_rank_percentage_im1_im2_gt, distance_rank_percentage_im2_im1_gt, \
                        num_thresholded_gt_inliers, label))

    def image_matching_dataset_exists(self, robust_matches_threshold):
        return os.path.isfile(self.__image_matching_dataset_file(suffix=robust_matches_threshold))

    def load_image_matching_dataset(self, robust_matches_threshold, rmatches_min_threshold=0, rmatches_max_threshold=10000, spl=10000):
        fns, data = self.load_general_dataset(self.__image_matching_dataset_file(suffix=robust_matches_threshold))
        R11s = data[:,0]
        R12s = data[:,1]
        R13s = data[:,2]
        R21s = data[:,3]
        R22s = data[:,4]
        R23s = data[:,5]
        R31s = data[:,6]
        R32s = data[:,7]
        R33s = data[:,8]
        num_rmatches = data[:,9]
        num_matches = data[:,10]
        spatial_entropy_1_8x8 = data[:,11]
        spatial_entropy_2_8x8 = data[:,12]
        spatial_entropy_1_16x16 = data[:,13]
        spatial_entropy_2_16x16 = data[:,14]
        pe_histogram = data[:,15:66] # 51 dimensional vector
        pe_polygon_area_percentage = data[:,66]
        nbvs_im1 = data[:,67]
        nbvs_im2 = data[:,68]
        te_histogram = data[:,69:149] # 81 dimensional vector
        ch_im1 = data[:,149:533] # 384 dimensional vector
        ch_im2 = data[:,533:917] # 384 dimensional vector
        vt_rank_percentage_im1_im2 = data[:,917]
        vt_rank_percentage_im2_im1 = data[:,918]
        sequence_scores_mean = data[:,919]
        sequence_scores_min = data[:,920]
        sequence_scores_max = data[:,921]
        sequence_distance_scores = data[:,922]
        lcc_im1_15 = data[:,923]
        lcc_im2_15 = data[:,924]
        min_lcc_15 = data[:,925]
        max_lcc_15 = data[:,926]
        lcc_im1_20 = data[:,927]
        lcc_im2_20 = data[:,928]
        min_lcc_20 = data[:,929]
        max_lcc_20 = data[:,930]
        lcc_im1_25 = data[:,931]
        lcc_im2_25 = data[:,932]
        min_lcc_25 = data[:,933]
        max_lcc_25 = data[:,934]
        lcc_im1_30 = data[:,935]
        lcc_im2_30 = data[:,936]
        min_lcc_30 = data[:,937]
        max_lcc_30 = data[:,938]
        lcc_im1_35 = data[:,939]
        lcc_im2_35 = data[:,940]
        min_lcc_35 = data[:,941]
        max_lcc_35 = data[:,942]
        lcc_im1_40 = data[:,943]
        lcc_im2_40 = data[:,944]
        min_lcc_40 = data[:,945]
        max_lcc_40 = data[:,946]
        shortest_path_length = data[:,947]
        mds_rank_percentage_im1_im2 = data[:,948]
        mds_rank_percentage_im2_im1 = data[:,949]
        distance_rank_percentage_im1_im2_gt = data[:,950]
        distance_rank_percentage_im2_im1_gt = data[:,951]
        gt_inliers = data[:,952]
        labels = data[:,953]

        ri = np.where( \
            (num_rmatches >= rmatches_min_threshold) & \
            (num_rmatches <= rmatches_max_threshold) & \
            (shortest_path_length <= spl)
        )[0]
        
        return fns[ri], [R11s[ri], R12s[ri], R13s[ri], R21s[ri], R22s[ri], R23s[ri], R31s[ri], R32s[ri], R33s[ri], \
          num_rmatches[ri], num_matches[ri], \
          spatial_entropy_1_8x8[ri], spatial_entropy_2_8x8[ri], spatial_entropy_1_16x16[ri], spatial_entropy_2_16x16[ri], \
          pe_histogram[ri], pe_polygon_area_percentage[ri], \
          nbvs_im1[ri], nbvs_im2[ri], \
          te_histogram[ri], \
          ch_im1[ri], ch_im2[ri], \
          vt_rank_percentage_im1_im2[ri], vt_rank_percentage_im2_im1[ri], \
          sequence_scores_mean[ri], sequence_scores_min[ri], sequence_scores_max[ri], sequence_distance_scores[ri], \
          lcc_im1_15[ri], lcc_im2_15[ri], min_lcc_15[ri], max_lcc_15[ri], \
          lcc_im1_20[ri], lcc_im2_20[ri], min_lcc_20[ri], max_lcc_20[ri], \
          lcc_im1_25[ri], lcc_im2_25[ri], min_lcc_25[ri], max_lcc_25[ri], \
          lcc_im1_30[ri], lcc_im2_30[ri], min_lcc_30[ri], max_lcc_30[ri], \
          lcc_im1_35[ri], lcc_im2_35[ri], min_lcc_35[ri], max_lcc_35[ri], \
          lcc_im1_40[ri], lcc_im2_40[ri], min_lcc_40[ri], max_lcc_40[ri], \
          shortest_path_length[ri], \
          mds_rank_percentage_im1_im2[ri], mds_rank_percentage_im2_im1[ri], \
          distance_rank_percentage_im1_im2_gt[ri], distance_rank_percentage_im2_im1_gt[ri], \
          gt_inliers[ri], labels[ri]]

    def save_tum_format(self, reconstruction, suffix):
        io.mkdir_p(self.__results_path())
        for i, r in enumerate(reconstruction):
            with open(self.__reconstruction_tum_file('reconstruction-{}-{}.txt'.format(i, suffix)), 'w') as f:
                f.write('# ground truth trajectory\n')
                f.write('# file: \'\'\n')
                f.write('# timestamp tx ty tz qx qy qz qw\n')

                for timestamp, s in enumerate(sorted(r.shots.keys())):
                    # print ('{} / {}'.format(timestamp, s))
                    q = Quaternion(matrix=r.shots[s].pose.get_rotation_matrix().T)
                    qw,qx,qy,qz = q
                    tx, ty, tz = r.shots[s].pose.get_origin()
                    # if suffix == 'gt':
                    #     f.write('{} {} {} {} {} {} {} {}\n'.format(timestamp, round(tx,4), round(tz,4), round(ty,4), \
                    #         round(qx,4), round(qz,4), round(qy,4), round(qw,4)))
                    # else:
                    f.write('{} {} {} {} {} {} {} {}\n'.format(timestamp, round(tx,4), round(ty,4), round(tz,4), \
                        round(qx,4), round(qy,4), round(qz,4), round(qw,4)))
                    

    def save_ate_results(self, results):
        io.mkdir_p(self.__results_path())
        with gzip.open(self.__ate_results_file('pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__ate_results_file('json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_ate_results(self):
        with gzip.open(self.__ate_results_file('pkl.gz'), 'rb') as fin:
            results = pickle.load(fin)
        return results

    def save_rpe_results(self, results):
        io.mkdir_p(self.__results_path())
        with gzip.open(self.__rpe_results_file('pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__rpe_results_file('json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_rpe_results(self):
        with gzip.open(self.__rpe_results_file('pkl.gz'), 'rb') as fin:
            results = pickle.load(fin)
        return results

    def save_match_graph_results(self, results):
        io.mkdir_p(self.__results_path())
        with gzip.open(self.__match_graph_results_file('pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__match_graph_results_file('json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def save_resectioning_order(self, resectioning_order, run):
        io.mkdir_p(self.__results_path())
        with open(self.__resectioning_order_file(run), 'w') as fout:
            json.dump(resectioning_order, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def save_resectioning_order_attempted(self, resectioning_order_attempted, run):
        io.mkdir_p(self.__results_path())
        with open(self.__resectioning_order_attempted_file(run), 'w') as fout:
            json.dump(resectioning_order_attempted, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def save_resectioning_order_common_tracks(self, resectioning_order_common_tracks, run):
        io.mkdir_p(self.__results_path())
        with open(self.__resectioning_order_common_tracks_file(run), 'w') as fout:
            json.dump(resectioning_order_common_tracks, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def save_reconstruction_results(self, results):
        io.mkdir_p(self.__results_path())
        with gzip.open(self.__reconstruction_results_file('pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__reconstruction_results_file('json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def find_matches(self, im1, im2):
        if self.matches_exists(im1):
            im1_matches = self.load_matches(im1)
            if im2 in im1_matches:
                return im1_matches[im2]
        if self.matches_exists(im2):
            im2_matches = self.load_matches(im2)
            if im1 in im2_matches:
                if len(im2_matches[im1]):
                    return im2_matches[im1][:, [1, 0]]
        return []

    def __tracks_graph_file(self, filename=None):
        """Return path of tracks file"""
        return os.path.join(self.data_path, filename or 'tracks.csv')

    def tracks_graph_exists(self, filename=None):
        return os.path.isfile(self.__tracks_graph_file(filename))

    def load_tracks_graph(self, filename=None):
        """Return graph (networkx data structure) of tracks"""
        with open(self.__tracks_graph_file(filename)) as fin:
            return load_tracks_graph(fin)

    def save_tracks_graph(self, graph, filename=None):
        with io.open_wt(self.__tracks_graph_file(filename)) as fout:
            save_tracks_graph(fout, graph)

    def load_undistorted_tracks_graph(self):
        return self.load_tracks_graph('undistorted_tracks.csv')

    def save_undistorted_tracks_graph(self, graph):
        return self.save_tracks_graph(graph, 'undistorted_tracks.csv')

    def __reconstruction_file(self, filename):
        """Return path of reconstruction file"""
        return os.path.join(self.data_path, filename or 'reconstruction.json')

    def __reconstruction_tum_file(self, filename):
        """Return path of reconstruction file"""
        return os.path.join(self.__results_path(), filename)

    def reconstruction_exists(self, filename=None):
        return os.path.isfile(self.__reconstruction_file(filename))

    def load_reconstruction(self, filename=None):
        with open(self.__reconstruction_file(filename)) as fin:
            reconstructions = io.reconstructions_from_json(json.load(fin))
        return reconstructions

    def save_reconstruction(self, reconstruction, filename=None, minify=False):
        with io.open_wt(self.__reconstruction_file(filename)) as fout:
            io.json_dump(io.reconstructions_to_json(reconstruction), fout, minify)

    def load_undistorted_reconstruction(self):
        return self.load_reconstruction(
            filename='undistorted_reconstruction.json')

    def save_undistorted_reconstruction(self, reconstruction):
        return self.save_reconstruction(
            reconstruction, filename='undistorted_reconstruction.json')

    def load_vocab_ranks_and_scores(self):
        im_scores = {}
        im_ranks = {}
        images = self.images()
        with open(os.path.join(self.data_path, 'vocab_out','match.out'),'r') as f:
            i1_prev = None
            for r, datum in enumerate(f.readlines()):
                i1, i2, score = datum.split(' ')
                i1 = int(i1)
                i2 = int(i2)
                score = float(score)
                if r == 0 or i1 != i1_prev:
                    rank = 1
                if images[i1] not in im_scores:
                    im_scores[images[i1]] = {images[i2]: 0.0}
                    im_ranks[images[i1]] = {images[i2]: 0}

                im_scores[images[int(i1)]][images[int(i2)]] = score
                im_ranks[images[int(i1)]][images[int(i2)]] = rank

                rank = rank + 1
                i1_prev = i1
        return im_ranks, im_scores

    def __reference_lla_path(self):
        return os.path.join(self.data_path, 'reference_lla.json')

    def invent_reference_lla(self, images=None):
        lat, lon, alt = 0.0, 0.0, 0.0
        wlat, wlon, walt = 0.0, 0.0, 0.0
        if images is None: images = self.images()
        for image in images:
            d = self.load_exif(image)
            if 'gps' in d and 'latitude' in d['gps'] and 'longitude' in d['gps']:
                w = 1.0 / d['gps'].get('dop', 15)
                lat += w * d['gps']['latitude']
                lon += w * d['gps']['longitude']
                wlat += w
                wlon += w
                if 'altitude' in d['gps']:
                    alt += w * d['gps']['altitude']
                    walt += w
        if wlat: lat /= wlat
        if wlon: lon /= wlon
        if walt: alt /= walt
        reference = {'latitude': lat, 'longitude': lon, 'altitude': 0}  # Set altitude manually.
        self.save_reference_lla(reference)
        return reference

    def save_reference_lla(self, reference):
        with io.open_wt(self.__reference_lla_path()) as fout:
            io.json_dump(reference, fout)

    def load_reference_lla(self):
        with io.open_rt(self.__reference_lla_path()) as fin:
            return io.json_load(fin)

    def reference_lla_exists(self):
        return os.path.isfile(self.__reference_lla_path())

    def __camera_models_file(self):
        """Return path of camera model file"""
        return os.path.join(self.data_path, 'camera_models.json')

    def load_camera_models(self):
        """Return camera models data"""
        with io.open_rt(self.__camera_models_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def save_camera_models(self, camera_models):
        """Save camera models data"""
        with io.open_wt(self.__camera_models_file()) as fout:
            obj = io.cameras_to_json(camera_models)
            io.json_dump(obj, fout)

    def __camera_models_overrides_file(self):
        """Path to the camera model overrides file."""
        return os.path.join(self.data_path, 'camera_models_overrides.json')

    def camera_models_overrides_exists(self):
        """Check if camera overrides file exists."""
        return os.path.isfile(self.__camera_models_overrides_file())

    def load_camera_models_overrides(self):
        """Load camera models overrides data."""
        with io.open_rt(self.__camera_models_overrides_file()) as fin:
            obj = json.load(fin)
            return io.cameras_from_json(obj)

    def __exif_overrides_file(self):
        """Path to the EXIF overrides file."""
        return os.path.join(self.data_path, 'exif_overrides.json')

    def exif_overrides_exists(self):
        """Check if EXIF overrides file exists."""
        return os.path.isfile(self.__exif_overrides_file())

    def load_exif_overrides(self):
        """Load EXIF overrides data."""
        with io.open_rt(self.__exif_overrides_file()) as fin:
            return json.load(fin)

    def profile_log(self):
        "Filename where to write timings."
        return os.path.join(self.data_path, 'profile.log')

    def _report_path(self):
        return os.path.join(self.data_path, 'reports')

    def load_report(self, path):
        """Load a report file as a string."""
        with open(os.path.join(self._report_path(), path)) as fin:
            return fin.read()

    def save_report(self, report_str, path):
        """Save report string to a file."""
        filepath = os.path.join(self._report_path(), path)
        io.mkdir_p(os.path.dirname(filepath))
        with io.open_wt(filepath) as fout:
            return fout.write(report_str)

    def __navigation_graph_file(self):
        "Return the path of the navigation graph."
        return os.path.join(self.data_path, 'navigation_graph.json')

    def save_navigation_graph(self, navigation_graphs):
        with io.open_wt(self.__navigation_graph_file()) as fout:
            io.json_dump(navigation_graphs, fout)

    def __ply_file(self, filename):
        return os.path.join(self.data_path, filename or 'reconstruction.ply')

    def save_ply(self, reconstruction, filename=None,
                 no_cameras=False, no_points=False):
        """Save a reconstruction in PLY format."""
        ply = io.reconstruction_to_ply(reconstruction, no_cameras, no_points)
        with io.open_wt(self.__ply_file(filename)) as fout:
            fout.write(ply)

    def __ground_control_points_file(self):
        return os.path.join(self.data_path, 'gcp_list.txt')

    def ground_control_points_exist(self):
        return os.path.isfile(self.__ground_control_points_file())

    def load_ground_control_points(self):
        """Load ground control points.

        It uses reference_lla to convert the coordinates
        to topocentric reference frame.
        """
        exif = {image: self.load_exif(image) for image in self.images()}

        with open(self.__ground_control_points_file()) as fin:
            return io.read_ground_control_points_list(
                fin, self.load_reference_lla(), exif)

    def save_processed_image(self, im1_fn, im2_fn, image, grid_size=None):
        io.mkdir_p(self.__processed_image_path())
        if grid_size is None:
            cv2.imwrite(self.__processed_image_file(im1_fn, min(im1_fn, im2_fn), max(im1_fn, im2_fn)), image)
        else:
            cv2.imwrite(self.__processed_image_file(im1_fn, min(im1_fn, im2_fn), max(im1_fn, im2_fn)), cv2.resize(image, (grid_size, grid_size)))

        return self.__processed_image_file(im1_fn, min(im1_fn, im2_fn), max(im1_fn, im2_fn))

    def load_processed_image(self, im1_fn, im2_fn):
        image = cv2.imread(self.__processed_image_file(im1_fn, min(im1_fn, im2_fn), max(im1_fn, im2_fn)), cv2.IMREAD_COLOR)
        with open(os.path.join(self.__resized_image_file(im1_fn) + '.json'), 'r') as fin:
            metadata = json.load(fin)
            
        return self.__processed_image_file(im1_fn, min(im1_fn, im2_fn), max(im1_fn, im2_fn)), image, metadata

    def processed_image_exists(self, im1_fn, im2_fn):
        return os.path.isfile(self.__processed_image_file(im1_fn, min(im1_fn, im2_fn), max(im1_fn, im2_fn)))

    def save_resized_image(self, im_fn, image, grid_size=None):
        io.mkdir_p(self.__resized_image_path())
        if grid_size is None:
            cv2.imwrite(self.__resized_image_file(im_fn), image)
        else:
            cv2.imwrite(self.__resized_image_file(im_fn), cv2.resize(image, (grid_size, grid_size)))
        
        # Save metadata of the original file along with the resized file
        with open(os.path.join(self.__resized_image_file(im_fn) + '.json'), 'w') as fout:
            metadata = {'height': image.shape[0], 'width': image.shape[1]}
            json.dump(metadata, fout, sort_keys=True, indent=4, separators=(',', ': '))

        return metadata

    def save_blurred_image(self, im_fn, image, grid_size=None, kernel_size=None):
        io.mkdir_p(self.__blurred_image_path())
        if grid_size is None:
            cv2.imwrite(self.__blurred_image_file(im_fn, kernel_size=kernel_size), image)
        else:
            cv2.imwrite(self.__blurred_image_file(im_fn, kernel_size=kernel_size), cv2.resize(image, (grid_size, grid_size)))

    def load_resized_image(self, im_fn):
        image = cv2.imread(self.__resized_image_file(im_fn), cv2.IMREAD_COLOR)
        with open(os.path.join(self.__resized_image_file(im_fn) + '.json'), 'r') as fin:
            metadata = json.load(fin)
        return image, self.__resized_image_file(im_fn), metadata

    def load_blurred_image(self, im_fn, kernel_size=3):
        image = cv2.imread(self.__blurred_image_file(im_fn, kernel_size=kernel_size), cv2.IMREAD_COLOR)
        # still load metadata from the resized image
        with open(os.path.join(self.__resized_image_file(im_fn) + '.json'), 'r') as fin:
            metadata = json.load(fin)
        return image, self.__blurred_image_file(im_fn, kernel_size=kernel_size), metadata

    def resized_image_exists(self, im_fn):
        return os.path.isfile(self.__resized_image_file(im_fn))

    def blurred_image_exists(self, im_fn, kernel_size=3):
        return os.path.isfile(self.__blurred_image_file(im_fn, kernel_size=kernel_size))

    # def photometric_error_triangle_transformations_exists(self, im1, im2):
    #     return os.path.isfile(self.__photometric_error_triangle_transformations_file(im1, im2) + '.json')

    # def load_photometric_error_triangle_transformations(self, im1, im2):
    #     with open(self.__photometric_error_triangle_transformations_file(im1, im2) + '.json', 'r') as fin:
    #         result = json.load(fin)

    #     result['Ms'] = np.array(result['Ms'])
    #     result['triangle_pts_img1'] = np.array(result['triangle_pts_img1'])
    #     result['triangle_pts_img2'] = np.array(result['triangle_pts_img2'])
    #     return result

    # def save_photometric_error_triangle_transformations(self, im1, im2, triangles_data):
    #     io.mkdir_p(self.__classifier_features_photometric_errors_triangle_transformations_path())
        
    #     triangles_data['Ms'] = triangles_data['Ms'].tolist()
    #     triangles_data['triangle_pts_img1'] = triangles_data['triangle_pts_img1'].tolist()
    #     triangles_data['triangle_pts_img2'] = triangles_data['triangle_pts_img2'].tolist()

    #     with open(os.path.join(self.__photometric_error_triangle_transformations_file(im1, im2) + '.json'), 'w') as fout:
    #         json.dump(triangles_data, fout, sort_keys=True, indent=4, separators=(',', ': '))


def load_tracks_graph(fileobj):
    g = nx.Graph()
    for line in fileobj:
        image, track, observation, x, y, R, G, B = line.split('\t')
        g.add_node(image, bipartite=0)
        g.add_node(track, bipartite=1)
        g.add_edge(
            image, track,
            feature=(float(x), float(y)),
            feature_id=int(observation),
            feature_color=(float(R), float(G), float(B)))
    return g


def save_tracks_graph(fileobj, graph):
    for node, data in graph.nodes(data=True):
        if data['bipartite'] == 0:
            image = node
            for track, data in graph[image].items():
                x, y = data['feature']
                fid = data['feature_id']
                r, g, b = data['feature_color']
                fileobj.write(u'%s\t%s\t%d\t%g\t%g\t%g\t%g\t%g\n' % (
                    str(image), str(track), fid, x, y, r, g, b))
