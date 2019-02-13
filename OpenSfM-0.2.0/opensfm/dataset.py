# -*- coding: utf-8 -*-

import os
import json
import pickle
import pyquaternion
import gzip

import cv2
import numpy as np
import networkx as nx
import six

from opensfm import io
from opensfm import config
from opensfm import context
from pyquaternion import Quaternion 

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

    def __feature_matching_results_path(self):
        """Return path of all matches directory"""
        return os.path.join(self.data_path, 'feature_matching_results')
    
    def __classifier_dataset_path(self):
        return os.path.join(self.data_path, 'classifier_dataset')

    def __results_path(self):
        return os.path.join(self.data_path, 'results')

    def __classifier_dataset_unthresholded_matches_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_matches')
    
    def __classifier_dataset_unthresholded_inliers_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_inliers')

    def __classifier_dataset_unthresholded_outliers_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_outliers')

    def __classifier_dataset_unthresholded_features_path(self):
        return os.path.join(self.__classifier_dataset_path(), 'unthresholded_features')

    def __matches_file(self, image):
        """File for matches for an image"""
        return os.path.join(self.__matches_path(), '{}_matches.pkl.gz'.format(image))

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

    def __feature_transformations_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'transformations.{}'.format(ext))

    def __feature_triplet_errors_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'triplet_errors.{}'.format(ext))

    def __feature_triplet_pairwise_errors_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'triplet_pairwise_errors.{}'.format(ext))

    def __feature_sequence_ranks_file(self, ext='pkl.gz'):
        return os.path.join(self.__classifier_features_path(), 'sequence_ranks.{}'.format(ext))

    def __feature_spatial_entropies_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'spatial_entropies.{}'.format(ext))

    def __feature_color_histograms_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'color_histograms.{}'.format(ext))

    def __feature_photometric_errors_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'photometric_errors.{}'.format(ext))

    def __feature_nbvs_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'nbvs.{}'.format(ext))

    def __feature_image_matching_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__classifier_features_path(), 'image_matching_results.{}'.format(ext))

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

    def __match_graph_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'match_graph_results.{}'.format(ext))

    def __reconstruction_results_file(self, ext='pkl.gz'):
        """File for flags indicating whether calibrated robust matching occured"""
        return os.path.join(self.__results_path(), 'reconstruction_results.{}'.format(ext))

    def matches_exists(self, image):
        return os.path.isfile(self.__matches_file(image))

    def load_matches(self, image):
        with gzip.open(self.__matches_file(image), 'rb') as fin:
            matches = pickle.load(fin)
        return matches

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

    def transformations_exists(self):
        return os.path.isfile(self.__feature_transformations_file())

    def save_transformations(self, transformations):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_transformations_file('pkl.gz'), 'wb') as fout:
            pickle.dump(transformations, fout)
        with open(self.__feature_transformations_file('json'), 'w') as fout:
            json.dump(transformations, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_transformations(self):
        with gzip.open(self.__feature_transformations_file(), 'rb') as fin:
            transformations = pickle.load(fin)
        return transformations

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

    def save_sequence_ranks(self, sequence_ranks):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_sequence_ranks_file('pkl.gz'), 'wb') as fout:
            pickle.dump(sequence_ranks, fout)
        with open(self.__feature_sequence_ranks_file('json'), 'w') as fout:
            json.dump(sequence_ranks, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_sequence_ranks(self):
        with gzip.open(self.__feature_sequence_ranks_file(), 'rb') as fin:
            sequence_ranks = pickle.load(fin)
        return sequence_ranks

    def save_spatial_entropies(self, spatial_entropies):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_spatial_entropies_file('pkl.gz'), 'wb') as fout:
            pickle.dump(spatial_entropies, fout)
        with open(self.__feature_spatial_entropies_file('json'), 'w') as fout:
            json.dump(spatial_entropies, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def spatial_entropies_exists(self):
        return os.path.isfile(self.__feature_spatial_entropies_file())

    def load_spatial_entropies(self):
        with gzip.open(self.__feature_spatial_entropies_file(), 'rb') as fin:
            spatial_entropies = pickle.load(fin)
        return spatial_entropies

    def save_color_histograms(self, color_histograms):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_color_histograms_file('pkl.gz'), 'wb') as fout:
            pickle.dump(color_histograms, fout)
        with open(self.__feature_color_histograms_file('json'), 'w') as fout:
            json.dump(color_histograms, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def color_histograms_exists(self):
        return os.path.isfile(self.__feature_color_histograms_file())

    def load_color_histograms(self):
        with gzip.open(self.__feature_color_histograms_file(), 'rb') as fin:
            color_histograms = pickle.load(fin)
        return color_histograms

    def save_photometric_errors(self, photometric_errors):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_photometric_errors_file('pkl.gz'), 'wb') as fout:
            pickle.dump(photometric_errors, fout)
        with open(self.__feature_photometric_errors_file('json'), 'w') as fout:
            json.dump(photometric_errors, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_photometric_errors(self):
        with gzip.open(self.__feature_photometric_errors_file(), 'rb') as fin:
            photometric_errors = pickle.load(fin)
        return photometric_errors

    def photometric_errors_exists(self):
        return os.path.isfile(self.__feature_photometric_errors_file())

    def save_nbvs(self, nbvs):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_nbvs_file('pkl.gz'), 'wb') as fout:
            pickle.dump(nbvs, fout)
        with open(self.__feature_nbvs_file('json'), 'w') as fout:
            json.dump(nbvs, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def nbvs_exists(self):
        return os.path.isfile(self.__feature_nbvs_file())

    def load_nbvs(self):
        with gzip.open(self.__feature_nbvs_file(), 'rb') as fin:
            nbvs = pickle.load(fin)
        return nbvs

    def save_image_matching_results(self, results):
        io.mkdir_p(self.__classifier_features_path())
        with gzip.open(self.__feature_image_matching_results_file('pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__feature_image_matching_results_file('json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def save_feature_matching_results(self, image, results):
        io.mkdir_p(self.__feature_matching_results_path())
        with gzip.open(self.__feature_matching_results_file(image, 'pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__feature_matching_results_file(image, 'json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

    def load_image_matching_results(self):
        with gzip.open(self.__feature_image_matching_results_file(), 'rb') as fin:
            results = pickle.load(fin)
        return results

    def load_feature_matching_results(self, image):
        with gzip.open(self.__feature_matching_results_file(image), 'rb') as fin:
            results = pickle.load(fin)
        return results
    
    def load_groundtruth_image_matching_results(self):
        fns, [R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, num_rmatches, num_matches, spatial_entropy_1_8x8, \
            spatial_entropy_2_8x8, spatial_entropy_1_16x16, spatial_entropy_2_16x16, pe_histogram, pe_polygon_area_percentage, \
            nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, \
            sq_rank_scores_mean, sq_rank_scores_min, sq_rank_scores_max, sq_distance_scores, num_gt_inliers, labels] \
            = self.load_image_matching_dataset(robust_matches_threshold=15)

        gt_results = {}
        for idx, _ in enumerate(fns[:,0]):
            im1 = fns[idx,0]
            im2 = fns[idx,1]
            if labels[idx] == 1:
                if im1 not in gt_results:
                    gt_results[im1] = {}
                if im2 not in gt_results:
                    gt_results[im2] = {}
                gt_results[im1][im2] = {"im1": im1, "im2": im2, "score": 1.0, "rmatches": num_rmatches[idx]}
                gt_results[im2][im1] = {"im1": im2, "im2": im1, "score": 1.0, "rmatches": num_rmatches[idx]}

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

    def save_unthresholded_outliers(self, image, outliers):
        io.mkdir_p(self.__classifier_dataset_unthresholded_outliers_path())
        with gzip.open(self.__unthresholded_outliers_file(image), 'wb') as fout:
            pickle.dump(outliers, fout)

    def load_unthresholded_outliers(self, image):
        with gzip.open(self.__unthresholded_outliers_file(image), 'rb') as fin:
            outliers = pickle.load(fin)
        return outliers

    def save_unthresholded_features(self, image, features):
        io.mkdir_p(self.__classifier_dataset_unthresholded_features_path())
        with gzip.open(self.__unthresholded_features_file(image), 'wb') as fout:
            pickle.dump(features, fout)

    def load_unthresholded_features(self, image):
        with gzip.open(self.__unthresholded_features_file(image), 'rb') as fin:
            features = pickle.load(fin)
        return features

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
            for im1 in self.images():
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
                        if max_lowes_ratio > lowes_threshold:
                            continue
                        # image 1,image 2, index 1, index 2, max lowe\'s ratio, max reprojection error, size 1, angle 1, size 2, angle 2, label
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
        max_distances = data[:,2]
        errors = data[:,3]
        size1 = data[:,4]
        angle1 = data[:,5]
        size2 = data[:,6]
        angle2 = data[:,7]
        rerr1 = data[:,8]
        rerr2 = data[:,9]
        labels = data[:,10]
        return fns, [indices_1, indices_2, max_distances, errors, size1, size2, angle1, angle2, rerr1, \
            rerr2, labels]


    def sequence_rank_adapter(self, options={}):
        sequence_ranks = self.load_sequence_ranks()
        sequence_rank_scores_mean = {}
        sequence_rank_scores_min = {}
        sequence_rank_scores_max = {}
        sequence_distance_scores = {}
        total_images = len(sequence_ranks.keys())

        for im1 in sequence_ranks:
            if im1 not in sequence_rank_scores_mean:
                sequence_rank_scores_mean[im1] = {}
                sequence_rank_scores_min[im1] = {}
                sequence_rank_scores_max[im1] = {}
                sequence_distance_scores[im1] = {}

            for im2 in sequence_ranks[im1]:
                sequence_distance_scores[im1][im2] = \
                    (total_images - sequence_ranks[im1][im2]['distance']) / total_images

                sequence_rank_scores_mean[im1][im2] = \
                    0.5 * (total_images - sequence_ranks[im1][im2]['rank']) / total_images + \
                    0.5 * (total_images - sequence_ranks[im2][im1]['rank']) / total_images
                sequence_rank_scores_min[im1][im2] = min(\
                    (total_images - sequence_ranks[im1][im2]['rank']) / total_images,
                    (total_images - sequence_ranks[im2][im1]['rank']) / total_images
                    )
                sequence_rank_scores_max[im1][im2] = max(\
                    (total_images - sequence_ranks[im1][im2]['rank']) / total_images,
                    (total_images - sequence_ranks[im2][im1]['rank']) / total_images
                    )

        return sequence_rank_scores_mean, sequence_rank_scores_min, sequence_rank_scores_max, sequence_distance_scores

    def save_image_matching_dataset(self, robust_matches_threshold):
        write_header = True
        lowes_threshold = 0.8
        transformations = self.load_transformations()
        spatial_entropies = self.load_spatial_entropies()
        photometric_errors = self.load_photometric_errors()
        nbvs = self.load_nbvs()
        triplet_pairwise_errors = self.load_triplet_pairwise_errors()
        color_histograms = self.load_color_histograms()
        vt_ranks, vt_scores = self.load_vocab_ranks_and_scores()
        sequence_scores_mean, sequence_scores_min, sequence_scores_max, sequence_distance_scores = \
            self.sequence_rank_adapter()

        counter = 0
        with open(self.__image_matching_dataset_file(suffix=robust_matches_threshold), 'w') as fout:
            
            for im1 in self.images():
                if im1 not in transformations:
                    continue
                im_all_matches, _, im_all_rmatches = self.load_all_matches(im1)
                im_unthresholded_inliers = self.load_unthresholded_inliers(im1)
                for im2 in transformations[im1]:
                    R = np.around(np.array(transformations[im1][im2]['rotation']), decimals=2)
                    se = spatial_entropies[im1][im2]
                    pe_histogram = ','.join(map(str, np.around(np.array(photometric_errors[im1][im2]['histogram-cumsum']), decimals=2)))
                    pe_polygon_area_percentage = photometric_errors[im1][im2]['polygon_area_percentage']
                    nbvs_im1 = nbvs[im1][im2]['nbvs_im1']
                    nbvs_im2 = nbvs[im1][im2]['nbvs_im2']
                    te_histogram = ','.join(map(str, np.around(np.array(triplet_pairwise_errors[im1][im2]['histogram-cumsum']), decimals=2)))
                    ch_im1 = ','.join(map(str, np.around(np.array(color_histograms[im1]['histogram']), decimals=2)))
                    ch_im2 = ','.join(map(str, np.around(np.array(color_histograms[im2]['histogram']), decimals=2)))
                    vt_rank_percentage_im1_im2 = 100.0 * vt_ranks[im1][im2] / len(self.images())
                    vt_rank_percentage_im2_im1 = 100.0 * vt_ranks[im2][im1] / len(self.images())

                    if im2 not in im_unthresholded_inliers or \
                        im_unthresholded_inliers[im2] is None:
                        label = -1
                        num_rmatches = 0
                        num_matches = 0
                        num_thresholded_gt_inliers = 0
                    else:
                        max_lowes_thresholds = np.maximum(im_unthresholded_inliers[im2][:,2], im_unthresholded_inliers[im2][:,3])
                        num_thresholded_gt_inliers = len(np.where(max_lowes_thresholds < lowes_threshold)[0])
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
                        num_thresholded_gt_inliers, label))

    def load_image_matching_dataset(self, robust_matches_threshold, rmatches_min_threshold=0, rmatches_max_threshold=10000):
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
        gt_inliers = data[:,923]
        labels = data[:,924]

        ri = np.where((num_rmatches >= rmatches_min_threshold) & (num_rmatches <= rmatches_max_threshold))[0]
        return fns[ri], [R11s[ri], R12s[ri], R13s[ri], R21s[ri], R22s[ri], R23s[ri], R31s[ri], R32s[ri], R33s[ri], \
          num_rmatches[ri], num_matches[ri], \
          spatial_entropy_1_8x8[ri], spatial_entropy_2_8x8[ri], spatial_entropy_1_16x16[ri], spatial_entropy_2_16x16[ri], \
          pe_histogram[ri], pe_polygon_area_percentage[ri], \
          nbvs_im1[ri], nbvs_im2[ri], \
          te_histogram[ri], \
          ch_im1[ri], ch_im2[ri], \
          vt_rank_percentage_im1_im2[ri], vt_rank_percentage_im2_im1[ri], \
          sequence_scores_mean[ri], sequence_scores_min[ri], sequence_scores_max[ri], sequence_distance_scores[ri], \
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

    def save_match_graph_results(self, results):
        io.mkdir_p(self.__results_path())
        with gzip.open(self.__match_graph_results_file('pkl.gz'), 'wb') as fout:
            pickle.dump(results, fout)
        with open(self.__match_graph_results_file('json'), 'w') as fout:
            json.dump(results, fout, sort_keys=True, indent=4, separators=(',', ': '))

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
