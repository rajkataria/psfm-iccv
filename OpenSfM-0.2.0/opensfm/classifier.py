import cv2
import glob
import json
import logging
import math
import networkx as nx
import numpy as np
import os
import pickle
import pyopengv
import random
import re
import scipy
import opensfm 
import sys
import matplotlib.pyplot as plt

from opensfm import features, multiview
from opensfm import context
from opensfm import types
# from opensfm.commands import formulate_graphs
from multiprocessing import Pool
from sklearn.externals import joblib
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial import Delaunay, ConvexHull
from scipy import interpolate
from shapely.geometry import Polygon, Point, LineString
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from timeit import default_timer as timer

logger = logging.getLogger(__name__)

def weighted_samples(p1, p2, weights, num_samples, cum_sum_weights):
    # Get samples
    samples = np.random.rand(num_samples) * cum_sum_weights[-1]
    sample_indices = np.zeros((num_samples,)).astype(np.int32)
    for i, s in enumerate(samples):
        sample_indices[i] = np.where(s < cum_sum_weights)[0][0]
    return p1[sample_indices], p2[sample_indices]

def getReprojectionError(p1_homo, p2_homo, F):
    line1 = np.matrix(p2_homo) * np.matrix(F)
    line1 /= np.linalg.norm(line1[:,0:2], axis=1)[:, np.newaxis]
    
    line2 = np.matrix(F) * np.matrix(p1_homo).T
    line2 = line2.T
    line2 /= np.linalg.norm(line2[:,0:2], axis=1)[:, np.newaxis]

    result1 = np.matrix(line1) * np.matrix(p1_homo).T
    result2 = np.matrix(line2) * np.matrix(p2_homo).T

    reproj1 = np.abs(np.diag(result1))
    reproj2 = np.abs(np.diag(result2))
    
    return reproj1, reproj2

def weighted_ransac_compute_inliers(F, p1, p2, weights, config):
    if type(F) != np.ndarray:
        return 0.0, 0
    if F[2,2] == 0.0:
        return 0.0, 0

    reproj1, reproj2 = getReprojectionError(p1, p2, F)

    robust_matching_threshold = config.get('robust_matching_threshold', 0.006)
    inlier_indices = np.where((reproj1 < robust_matching_threshold) & (reproj2 < robust_matching_threshold))
    
    return inlier_indices, np.sum(weights[inlier_indices])

def robust_match_fundamental_weighted(p1, p2, matches, config, w=np.array([])):
    '''Computes robust matches by estimating the Fundamental matrix via RANSAC.
    '''
    if len(matches) < 8:
        return [], np.array([]), np.zeros((3,3)), 0

    if len(w) > 0:
        weights = w
    else:
        weights = matches[:, 4].copy()

    relevant_indices = np.isfinite(weights)
    weights = weights[relevant_indices]
    matches = matches[relevant_indices]

    p1 = p1[matches[:, 0].astype(int)][:, :2].copy()
    p2 = p2[matches[:, 1].astype(int)][:, :2].copy()

    p1 = np.concatenate((p1, np.ones((len(p1),1))), axis=1)
    p2 = np.concatenate((p2, np.ones((len(p2),1))), axis=1)

    probability_ = 0.9999
    max_score = 0.0
    num_samples = 8
    iterations_ = 0

    epsilon = 0.00000001
    weights_sum = np.sum(weights)
    weights_cum_sum = np.cumsum(weights)

    weights_length = len(weights)
    FM_8POINT = cv2.FM_8POINT if context.OPENCV3 else cv2.cv.CV_FM_8POINT
    FM_LMEDS = cv2.FM_LMEDS if context.OPENCV3 else cv2.cv.CV_FM_LMEDS

    if len(np.where(weights > 0.0)[0]) < 8:
        return [], np.array([]), np.zeros((3,3)), 0

    for i in xrange(0,1000):
        iterations_ += 1

        p1_, p2_ = weighted_samples(p1, p2, weights, num_samples, weights_cum_sum)
        F_est, mask = cv2.findFundamentalMat(p1_, p2_, FM_8POINT)
        inliers_est, score = weighted_ransac_compute_inliers(F_est, p1, p2, weights, config)
        if score >= max_score:
            max_score = score
            F = F_est
            inliers = inliers_est

            w = max_score / (1.0 * weights_sum)
            p_no_outliers = 1.0 - np.power(w, num_samples)
            p_no_outliers = np.maximum(epsilon, p_no_outliers)
            p_no_outliers = np.minimum(1.0 - epsilon, p_no_outliers)
            k = np.log10(1.0 - probability_) / np.log10(p_no_outliers)
            if i >= k:
                break

    if type(F) != np.ndarray:
        return [], np.array([]), np.zeros((3,3)), 0
    if F[2,2] == 0.0:
        return [], np.array([]), np.zeros((3,3)), 0

    return matches[inliers], np.array([]), F, 1

def calculate_spatial_entropy(image_coordinates, grid_size):
    epsilon = 0.000000000001
    entropy = 0.0
    dmap = np.zeros((grid_size*grid_size,)).astype(np.float)
    dmap_entropy = np.zeros((grid_size*grid_size,)).astype(np.float)
    denormalized_image_coordinates = features.denormalized_image_coordinates(image_coordinates, grid_size, grid_size)
    indx = denormalized_image_coordinates[:,0].astype(np.int32)
    indy = denormalized_image_coordinates[:,1].astype(np.int32)
    for i in xrange(0,len(indx)):
        # print '#'*100
        # print '{} : ({},{}) : ({})'.format(grid_size, indx[i], indy[i], image_coordinates[i,:])
        dmap[indy[i]*grid_size + indx[i]] = 1.0
        dmap_entropy[indy[i]*grid_size + indx[i]] += 1.0
    prob_map = dmap_entropy / np.sum(dmap_entropy) + epsilon
    entropy = np.sum(prob_map * np.log2(prob_map))
    return round(-entropy/np.log2(np.sum(dmap_entropy)),4), dmap.reshape((grid_size, grid_size))

def next_best_view_score(image_coordinates):
  # Based on the paper Structure-from-Motion Revisited - https://demuc.de/papers/schoenberger2016sfm.pdf
  # Get a score based on number of common tracks and spatial distribution of the tracks in the image
  grid_sizes = [2, 4, 8]
  score = 0

  for grid_size in grid_sizes:
    dmap = np.zeros((grid_size*grid_size,1))
    # indx = ((image_coordinates[:,0] + 1)/2 * grid_size).astype(np.int32)
    # indy = ((image_coordinates[:,1] + 1)/2 * grid_size).astype(np.int32)
    denormalized_image_coordinates = features.denormalized_image_coordinates(image_coordinates, grid_size, grid_size)
    indx = denormalized_image_coordinates[:,0].astype(np.int32)
    indy = denormalized_image_coordinates[:,1].astype(np.int32)
    dmap[indy*grid_size + indx] = 1
    score += np.sum(dmap) * grid_size * grid_size
  return score

def classify_boosted_dts_feature_match(arg):
    fns, indices1, indices2, dists1, dists2, size1, size2, angle1, angle2, labels, \
        train, regr, options = arg
    rng = np.random.RandomState(1)
    num_matches = len(dists1)
    X = np.concatenate(( \
            np.maximum(dists1, dists2).reshape((num_matches,1)),
            size1.reshape((num_matches,1)),
            size2.reshape((num_matches,1)),
            size2.reshape((num_matches,1)) - size1.reshape((num_matches,1)), 
            np.absolute(size2.reshape((num_matches,1)) - size1.reshape((num_matches,1))), 
            angle1.reshape((num_matches,1)), 
            angle2.reshape((num_matches,1)),
            angle2.reshape((num_matches,1)) - angle1.reshape((num_matches,1)),
            np.absolute(angle2.reshape((num_matches,1)) - angle1.reshape((num_matches,1))).tolist()
        ),
        axis=1)

    y = labels
    # Fit regression model
    if regr is None:
        regr = GradientBoostingClassifier(max_depth=options['max_depth'], n_estimators=options['n_estimators'], subsample=1.0, random_state=rng)
        regr.fit(X, y)

    y_ = regr.predict_proba(X)[:,1]
    return fns, indices1, indices2, dists1, dists2, regr, y_

def classify_boosted_dts_image_match(arg):
    dsets, fns, R11s, R12s, R13s, R21s, R22s, R23s, R31s, R32s, R33s, num_rmatches, num_matches, spatial_entropy_1_8x8, \
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
        mds_rank_percentage_im1_im2, mds_rank_percentage_im2_im1, \
        distance_rank_percentage_im1_im2_gt, distance_rank_percentage_im2_im1_gt, \
        labels, weights, \
        train, regr, options = arg

    classifier_type = options['classifier']
    epsilon = 0.00000001 * np.ones((len(labels),1))
    rng = np.random.RandomState(1)

    X = np.concatenate((
        R33s.reshape((len(labels),1)),
        num_rmatches.reshape((len(labels),1)),
        num_matches.reshape((len(labels),1)),
        num_matches.reshape((len(labels),1)) / (num_rmatches.reshape((len(labels),1)) + epsilon),
        num_rmatches.reshape((len(labels),1)) / (num_matches.reshape((len(labels),1)) + epsilon),
        np.log(num_rmatches.reshape((len(labels),1)) + epsilon),

        spatial_entropy_1_8x8.reshape((len(labels),1)),
        spatial_entropy_2_8x8.reshape((len(labels),1)),
        spatial_entropy_1_8x8.reshape((len(labels),1))/(spatial_entropy_2_8x8.reshape((len(labels),1)) + epsilon),
        spatial_entropy_2_8x8.reshape((len(labels),1))/(spatial_entropy_1_8x8.reshape((len(labels),1)) + epsilon),

        spatial_entropy_1_16x16.reshape((len(labels),1)),
        spatial_entropy_2_16x16.reshape((len(labels),1)),
        spatial_entropy_1_16x16.reshape((len(labels),1))/(spatial_entropy_2_16x16.reshape((len(labels),1)) + epsilon),
        spatial_entropy_2_16x16.reshape((len(labels),1))/(spatial_entropy_1_16x16.reshape((len(labels),1)) + epsilon),
        
        pe_histogram.reshape((len(labels),-1)),
        pe_polygon_area_percentage.reshape((len(labels),1)),

        nbvs_im1.reshape((len(labels),1)),
        nbvs_im2.reshape((len(labels),1)),
        np.minimum(nbvs_im1, nbvs_im2).reshape((len(labels),1)),
        ((nbvs_im1 + nbvs_im2) / 2.0).reshape((len(labels),1)),

        te_histogram.reshape((len(labels),-1)),
        ch_im1.reshape((len(labels),-1)),
        ch_im2.reshape((len(labels),-1)),
        vt_rank_percentage_im1_im2.reshape((len(labels),-1)),
        vt_rank_percentage_im2_im1.reshape((len(labels),-1)),
        np.log(vt_rank_percentage_im1_im2.reshape((len(labels),1)) + epsilon),
        np.log(vt_rank_percentage_im2_im1.reshape((len(labels),1)) + epsilon),
        np.log(np.maximum(vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1).reshape((len(labels),-1)) + epsilon),
        np.log(np.minimum(vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1).reshape((len(labels),-1)) + epsilon),
        np.log(((vt_rank_percentage_im1_im2 + vt_rank_percentage_im2_im1) / 2.0).reshape((len(labels),1)) + epsilon),
        sq_rank_scores_mean.reshape((len(labels),-1)), 
        sq_rank_scores_min.reshape((len(labels),-1)),
        sq_rank_scores_max.reshape((len(labels),-1)),
        sq_distance_scores.reshape((len(labels),-1)),
        lcc_im1_15.reshape((len(labels),-1)),
        lcc_im2_15.reshape((len(labels),-1)),
        min_lcc_15.reshape((len(labels),-1)),
        max_lcc_15.reshape((len(labels),-1)),
        lcc_im1_20.reshape((len(labels),-1)),
        lcc_im2_20.reshape((len(labels),-1)),
        min_lcc_20.reshape((len(labels),-1)),
        max_lcc_20.reshape((len(labels),-1)),
        lcc_im1_25.reshape((len(labels),-1)),
        lcc_im2_25.reshape((len(labels),-1)),
        min_lcc_25.reshape((len(labels),-1)),
        max_lcc_25.reshape((len(labels),-1)),
        lcc_im1_30.reshape((len(labels),-1)),
        lcc_im2_30.reshape((len(labels),-1)),
        min_lcc_30.reshape((len(labels),-1)),
        max_lcc_30.reshape((len(labels),-1)),
        lcc_im1_35.reshape((len(labels),-1)),
        lcc_im2_35.reshape((len(labels),-1)),
        min_lcc_35.reshape((len(labels),-1)),
        max_lcc_35.reshape((len(labels),-1)),
        lcc_im1_40.reshape((len(labels),-1)),
        lcc_im2_40.reshape((len(labels),-1)),
        min_lcc_40.reshape((len(labels),-1)),
        max_lcc_40.reshape((len(labels),-1)),

        shortest_path_length.reshape((len(labels),-1)),

        mds_rank_percentage_im1_im2.reshape((len(labels),-1)),
        mds_rank_percentage_im2_im1.reshape((len(labels),-1)),
        distance_rank_percentage_im1_im2_gt.reshape((len(labels),-1)),
        distance_rank_percentage_im2_im1_gt.reshape((len(labels),-1)),
        ),
    axis=1)
    
    y = labels

    # Fit regression model
    if regr is None:
        regr = GradientBoostingClassifier(max_depth=options['max_depth'], n_estimators=options['n_estimators'], subsample=1.0, random_state=rng)
        regr.fit(X, y, sample_weight=weights)

    # Predict
    y_ = regr.predict_proba(X)[:,1]
    return fns, num_rmatches, regr, y_, shortest_path_length, labels

def relative_pose(arg):
    im1, im2, p1, p2, cameras, exifs, rmatches, threshold = arg

    p1_ = p1[rmatches[:, 0].astype(int)][:, :2].copy()
    p2_ = p2[rmatches[:, 1].astype(int)][:, :2].copy()
    camera1 = cameras[exifs[im1]['camera']]
    camera2 = cameras[exifs[im2]['camera']]
    b1 = camera1.pixel_bearings(p1_)
    b2 = camera2.pixel_bearings(p2_)

    T = pyopengv.relative_pose_ransac(b1, b2, "STEWENIUS", 1 - np.cos(threshold), 1000)
    return im1, im2, T

def calculate_transformations(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = config['processes']
    threshold = config['robust_matching_threshold']
    cached_p = {}
    args = []
    Rs = []
    transformations = {}
    num_pairs = 0
    
    if data.transformations_exists():
        logger.info('Pairwise transformations exist!')
        return data.load_transformations(), None

    logger.info('Calculating pairwise transformations...')
    for im1 in images:
        im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
        if im1_all_robust_matches is None:
            continue
        
        if im1 not in cached_p:
            p1, f1, c1 = ctx.data.load_features(im1)
            cached_p[im1] = p1
        else:
            p1 = cached_p[im1]

        for im2 in im1_all_robust_matches:
            rmatches = im1_all_robust_matches[im2]
            p2, f2, c2 = ctx.data.load_features(im2)
            if im2 not in cached_p:
                p2, f2, c2 = ctx.data.load_features(im2)
                cached_p[im2] = p2
            else:
                p2 = cached_p[im2]
            if len(rmatches) == 0:
                continue
            args.append([im1, im2, p1, p2, cameras, exifs, rmatches, threshold])

    p_results = []
    results = {}
    p = Pool(processes)
    if processes == 1:
        for arg in args:
            p_results.append(relative_pose(arg))
    else:
        p_results = p.map(relative_pose, args)
        p.close()

    for r in p_results:
        im1, im2, T = r
        R = T[0:3,0:3]
        Rs.append(R.reshape((1,-1)))
        if im1 not in transformations:
            transformations[im1] = {}
        transformations[im1][im2] = {'im1': im1, 'im2': im2, 'rotation': R.tolist(), 'transformation': T.tolist()}
        num_pairs = num_pairs + 1

    data.save_transformations(transformations)
    return transformations, num_pairs

def calculate_spatial_entropies(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    cached_p = {}
    entropies = {}
    
    if data.spatial_entropies_exists():
        logger.info('Spatial entropies exist!')
        return

    logger.info('Calculating spatial entropies...')
    for im1 in images:
        im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
        
        if im1 not in cached_p:
            p1, f1, c1 = ctx.data.load_features(im1)
            cached_p[im1] = p1
        else:
            p1 = cached_p[im1]

        for im2 in im1_all_robust_matches:
            print '{} - {}'.format(im1, im2)
            rmatches = im1_all_robust_matches[im2]
            matches = im1_all_matches[im2]
            
            p2, f2, c2 = ctx.data.load_features(im2)
            if im2 not in cached_p:
                p2, f2, c2 = ctx.data.load_features(im2)
                cached_p[im2] = p2
            else:
                p2 = cached_p[im2]
            if len(rmatches) == 0:
                continue

            p1_rmatches = p1[rmatches[:, 0].astype(int)]
            p2_rmatches = p2[rmatches[:, 1].astype(int)]

            p1_non_rmatches = p1[np.array(list(set(list(range(0,len(p1)))) - set(rmatches[:, 0].astype(int))))]
            p2_non_rmatches = p2[np.array(list(set(list(range(0,len(p2)))) - set(rmatches[:, 1].astype(int))))]

            entropy_im1_8, _ = calculate_spatial_entropy(p1_rmatches, 8)
            entropy_im2_8, _ = calculate_spatial_entropy(p2_rmatches, 8)
            entropy_im1_16, _ = calculate_spatial_entropy(p1_rmatches, 16)
            entropy_im2_16, _ = calculate_spatial_entropy(p2_rmatches, 16)
            
            p1_matches = p1[matches[:, 0].astype(int)]
            p2_matches = p2[matches[:, 1].astype(int)]
            entropy_rmatches_im1_224, rmatches_map_im1 = calculate_spatial_entropy(p1_rmatches, 224)
            entropy_rmatches_im2_224, rmatches_map_im2 = calculate_spatial_entropy(p2_rmatches, 224)
            entropy_matches_im1_224, matches_map_im1 = calculate_spatial_entropy(p1_matches, 224)
            entropy_matches_im2_224, matches_map_im2 = calculate_spatial_entropy(p2_matches, 224)
            entropy_non_rmatches_im1_224, non_rmatches_map_im1 = calculate_spatial_entropy(p1_non_rmatches, 224)
            entropy_non_rmatches_im2_224, non_rmatches_map_im2 = calculate_spatial_entropy(p2_non_rmatches, 224)

            data.save_match_map('rmatches---{}-{}'.format(im1,im2), rmatches_map_im1)
            data.save_match_map('rmatches---{}-{}'.format(im2,im1), rmatches_map_im2)
            data.save_match_map('matches---{}-{}'.format(im1,im2), matches_map_im1)
            data.save_match_map('matches---{}-{}'.format(im2,im1), matches_map_im2)
            data.save_match_map('non_rmatches---{}-{}'.format(im1,im2), non_rmatches_map_im1)
            data.save_match_map('non_rmatches---{}-{}'.format(im2,im1), non_rmatches_map_im2)

            if im1 not in entropies:
                entropies[im1] = {}
            if im2 not in entropies:
                entropies[im2] = {}

            result = {
                'entropy_im1_8': entropy_im1_8, 'entropy_im2_8': entropy_im2_8, \
                'entropy_im1_16': entropy_im1_16, 'entropy_im2_16': entropy_im2_16 \
                }
            entropies[im1][im2] = result.copy()
            entropies[im2][im1] = result.copy()
    data.save_spatial_entropies(entropies)

def calculate_histogram(full_image, mask, x, y, w, h, histSize, color_image, debug):
    if color_image:
        image = full_image[x:x+w, y:y+h, 0:3]
    else:
        image = full_image[x:x+w, y:y+h]

    chans = cv2.split(image)
    if color_image:
        colors = ("b", "g", "r")
    else:
        colors = ("k")
    features = []

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], channels=[0], mask=mask, histSize=[histSize], ranges=[0, 256]).reshape(-1,)
        features = np.append(features, hist)
    return features

def calculate_full_image_histograms(arg):
    image_name, histSize, histogram_images, color_image, debug = arg
    if color_image:
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        width, height, channels = image.shape
    else:
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        width, height = image.shape

    histograms = []
    if debug:
        counter = 1
    else:
        counter = None
    if histogram_images == 6:
        histogram_metadata = [(0, width/2, 0, height/2), (0, width/2, height/2, height/2), (width/2, width/2, 0, height/2), \
            (width/2, width/2, height/2, height/2), (width/4, width/2, height/4, height/2), (0, width, 0, height)]
    elif histogram_images == 4:
        histogram_metadata = [(0, width/2, 0, height/2), (0, width/2, height/2, height/2), (width/2, width/2, 0, height/2), \
            (width/2, width/2, height/2, height/2)]
    elif histogram_images == 1:
        histogram_metadata = [(0, width, 0, height)]
        
    for x, w, y, h in histogram_metadata:
        histograms = np.append(histograms, calculate_histogram(image, None, x, y, w, h, histSize, color_image=color_image, debug=counter))
        if debug:
            counter += 2
    if debug:
        plt.show()
    return os.path.basename(image_name), histograms.tolist()

def calculate_color_histograms(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = config['processes']
    args = []
    color_image = True
    num_histogram_images = 4
    histogram_size = 32 * num_histogram_images
    if color_image:
        histogram_size = histogram_size * 3
    if data.color_histograms_exists():
        logger.info('Color histograms exist!')
        return
    logger.info('Calculating color histograms...')
    if color_image:
        hs = histogram_size/(num_histogram_images * 3)
    else:
        hs = histogram_size/num_histogram_images
    for im in images:
        args.append([os.path.join(ctx.data.data_path + '/images', im), hs, num_histogram_images, color_image, False])
    
    p_results = []
    results = {}
    p = Pool(processes)
    if processes == 1:
        for arg in args:
            p_results.append(calculate_full_image_histograms(arg))
    else:
        p_results = p.map(calculate_full_image_histograms, args)
        p.close()

    for im, histogram in p_results:
        results[im] = {'histogram': histogram}
    data.save_color_histograms(results)

def sample_points_triangle(v, n):
  x = np.sort(np.random.rand(2, n), axis=0)
  return np.dot(np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]), v)

def get_triangle_points(triangle_pts, sampled_points):
  polygon = Polygon([(pt[0][0],pt[0][1]) for pt in triangle_pts])
  min_x, min_y, max_x, max_y = polygon.bounds
  points = []

  for i in xrange(len(sampled_points)-1, -1, -1):
    random_point = sampled_points[i]
    if (random_point.within(polygon)):
      points.append(np.array([random_point.coords.xy[0][0], random_point.coords.xy[1][0]]) )
      del sampled_points[i]

  return np.array(points)

def sample_points_polygon(denormalized_points, v, n):
  polygon_pts = []
  for i,_ in enumerate(v):
    polygon_pts.append(( int(denormalized_points[v[i], 0]), int(denormalized_points[v[i], 1]) ))

  polygon = Polygon(polygon_pts)
  min_x, min_y, max_x, max_y = polygon.bounds
  points = []

  if True:
    while len(points) < n:
      random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
      if (random_point.within(polygon)):
        points.append(random_point)
  else:
    for i in xrange(0,n):
      x = random.uniform(min_x, max_x)
      x_line = LineString([(x, min_y), (x, max_y)])
      x_line_intercept_min, x_line_intercept_max = x_line.intersection(polygon).xy[1].tolist()
      y = random.uniform(x_line_intercept_min, x_line_intercept_max)
      points.append(Point([x, y]))

  return polygon, points

def warp_image(data, Ms, triangle_pts_img1, triangle_pts_img2, img1, img2, im1, im2, flags, colors, grid_size, save_individual_triangles=False):
    # img2_o_final = -255 * np.ones(img1.shape, dtype=img1.dtype)
    imgs_per_triangle = []

    for s, _ in enumerate(triangle_pts_img1):
        pts_img1 = triangle_pts_img1[s].reshape((1,3,2))
        pts_img2 = triangle_pts_img2[s].reshape((1,3,2))
        r1 = cv2.boundingRect(pts_img1)
        r2 = cv2.boundingRect(pts_img2)

        img1_tri_cropped = []
        img2_tri_cropped = []
        for i in xrange(0, 3):
            img1_tri_cropped.append(((pts_img1[0][i][0] - r1[0]),(pts_img1[0][i][1] - r1[1])))
            img2_tri_cropped.append(((pts_img2[0][i][0] - r2[0]),(pts_img2[0][i][1] - r2[1])))

        img1_cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        M = cv2.getAffineTransform(np.float32(img1_tri_cropped),np.float32(img2_tri_cropped))
        img2_cropped = cv2.warpAffine( img1_cropped, M, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(img2_tri_cropped), (1.0, 1.0, 1.0), 16, 0);
        img2_cropped = img2_cropped * mask

        # Output image is set to white
        img2_o = -255 * np.ones(img1.shape, dtype=img1.dtype)
        img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_cropped

        imgs_per_triangle.append(img2_o)
        # masks_per_triangle.append(mask_o)

        # img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
        # img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_cropped

        if flags['draw_points']:
            for i, _ in enumerate(pts_img1[0]):
                color_index = random.randint(0,len(colors)-1)
                cv2.circle(img1, (int(pts_img1[0][i][0]), int(pts_img1[0][i][1])), 2, colors[color_index], 2)
                cv2.circle(img2, (int(pts_img2[0][i][0]), int(pts_img2[0][i][1])), 2, colors[color_index], 2)
    
    triangle_ids, normalized_error_per_triangle, aggregated_wi, aggregated_m, aggregated_em, wi_cache, m_cache, em_cache = \
        calculate_individual_errors_and_aggregate_triangles(data, img1, img2, np.array(imgs_per_triangle).copy(), grid_size, \
            outlier_indices=[], cached_wi=[], cached_m=[], cached_em=[])

    normalized_error_per_triangle = np.array(normalized_error_per_triangle)
    order = np.argsort(-normalized_error_per_triangle)
    outliers, outlier_indices = detect_outliers(normalized_error_per_triangle)
    # print 'here'
    # print wi_cache
    _, _, aggregated_wi_filtered, aggregated_m_filtered, aggregated_em_filtered, _, _, _ = \
        calculate_individual_errors_and_aggregate_triangles(data, img1, img2, np.array(imgs_per_triangle).copy(), grid_size, \
        outlier_indices=outlier_indices, cached_wi=wi_cache, cached_m=m_cache, cached_em=em_cache)


    data.save_photometric_errors_map('{}-{}-a-wi-unfiltered'.format(os.path.basename(im1), os.path.basename(im2)), aggregated_wi, size=grid_size)
    data.save_photometric_errors_map('{}-{}-a-m-unfiltered'.format(os.path.basename(im1), os.path.basename(im2)), aggregated_m, size=grid_size)
    data.save_photometric_errors_map('{}-{}-a-em-unfiltered'.format(os.path.basename(im1), os.path.basename(im2)), aggregated_em, size=grid_size)

    data.save_photometric_errors_map('{}-{}-a-wi-filtered'.format(os.path.basename(im1), os.path.basename(im2)), aggregated_wi_filtered, size=grid_size)
    data.save_photometric_errors_map('{}-{}-a-m-filtered'.format(os.path.basename(im1), os.path.basename(im2)), aggregated_m_filtered, size=grid_size)
    data.save_photometric_errors_map('{}-{}-a-em-filtered'.format(os.path.basename(im1), os.path.basename(im2)), aggregated_em_filtered, size=grid_size)

    # return img2_o_final, aggregated_m
    return aggregated_wi_filtered, aggregated_m_filtered

def calculate_individual_errors_and_aggregate_triangles(data, img1, img2, imgs_per_triangle, grid_size, outlier_indices=[], cached_wi=[], cached_m=[], cached_em=[]):
    triangle_ids = []
    normalized_error_per_triangle = []
    wi_cache = []
    m_cache = []
    em_cache = []
    aggregated_wi = np.zeros(img1.shape, dtype=img1.dtype)
    aggregated_m = np.zeros((grid_size, grid_size), dtype=img1.dtype)
    aggregated_em = np.zeros((grid_size, grid_size), dtype=img1.dtype)

    # print imgs_per_triangle
    for count, _ in enumerate(imgs_per_triangle):
        if count in outlier_indices:
            continue
        if len(cached_wi) > 0 and len(cached_m) > 0 and len(cached_em) > 0:
            rr,rc,rd = np.where(cached_wi[count] >= 0)
            aggregated_wi[cached_wi[count] > 0] = cached_wi[count][cached_wi[count] > 0].astype(np.uint8)
            aggregated_m[cached_m[count] > 0] = cached_m[count][cached_m[count] > 0].astype(np.uint8)
            aggregated_em[rr,rc] = cached_em[count][rr,rc]
        else:
            wi = imgs_per_triangle[count]
            m = np.zeros((grid_size, grid_size), dtype=img1.dtype)
            rr,rc,rd = np.where(wi >= 0)
            rr_,rc_,rd_ = np.where(wi < 0)
            
            m[rr,rc] = 255
            m[rr_,rc_] = 0
            
            em = calculate_error_map(img2, wi)

            aggregated_wi[wi > 0] = wi[wi > 0].astype(np.uint8)
            aggregated_m[m > 0] = m[m > 0].astype(np.uint8)
            aggregated_em[rr,rc] = em[rr,rc]
            wi_cache.append(wi)
            m_cache.append(m)
            em_cache.append(em)
            wi[wi < 0] = 0
            
            triangle_ids.append(count)
            normalized_error = 1.0*np.sum(em)/len(rr)
            normalized_error_per_triangle.append(normalized_error)
            # if count < 0:
            #     data.save_photometric_errors_map('{}-{}-t-{}-w'.format(os.path.basename(im1), os.path.basename(im2), count), wi, size=224)
            #     data.save_photometric_errors_map('{}-{}-t-{}-m'.format(os.path.basename(im1), os.path.basename(im2), count), m, size=224)
            #     data.save_photometric_errors_map('{}-{}-t-{}-em'.format(os.path.basename(im1), os.path.basename(im2), count), em, size=224)

    return triangle_ids, normalized_error_per_triangle, aggregated_wi, aggregated_m, aggregated_em, wi_cache, m_cache, em_cache

def detect_outliers(data_pts):
    threshold = 2
    outliers = []
    outlier_indices = []
    # mean = np.mean(data_pts)
    median = np.median(data_pts)
    std = np.std(data_pts)
    
    for i,y in enumerate(data_pts):
        z_score = (y - median)/std
        # if np.abs(z_score) > threshold:
        if z_score > threshold:
            outliers.append(y)
            outlier_indices.append(i)
    return outliers, outlier_indices

def calculate_photometric_error_histogram(errors):
    histogram, bins = np.histogram(np.array(errors), bins=51, range=(0,250))
    epsilon = 0.000000001
    # if debug:
    #     width = 0.9 * (bins[1] - bins[0])
    #     center = (bins[:-1] + bins[1:]) / 2

    #     plt.bar(center, histogram, align='center', width=width)
    #     plt.xlabel('L2 Error (LAB)', fontsize=16)
    #     plt.ylabel('Error count', fontsize=16)

    #     fig = plt.gcf()
    #     fig.set_size_inches(18.5, 10.5)
    #     plt.savefig(os.path.join(patchdataset,'photometric-data/photometric-histogram-{}-{}.png'.format(os.path.basename(im1), os.path.basename(im2))))
  # return histogram
    return histogram.tolist(), \
        np.cumsum(np.round((1.0 * histogram / (np.sum(histogram) + epsilon)), 2)).tolist(), \
        np.round((1.0 * histogram / (np.sum(histogram) + epsilon)), 2).tolist(), \
        bins.tolist()

def calculate_convex_hull(data, img1_o, denormalized_p1_points, img2_o, denormalized_p2_points, flags):
    try:
        hull_img1 = ConvexHull( [ (x,y) for x,y in denormalized_p1_points[:,0:2].tolist()] )
        hull_img2 = ConvexHull( [ (x,y) for x,y in denormalized_p2_points[:,0:2].tolist()] )

        if flags['draw_hull']:
            for v, vertex in enumerate(hull_img1.vertices):
                if v == 0:
                    continue
                cv2.line(img1_o, ( int(denormalized_p1_points[hull_img1.vertices[v-1], 0]), int(denormalized_p1_points[hull_img1.vertices[v-1], 1]) ), \
                    ( int(denormalized_p1_points[hull_img1.vertices[v], 0]), int(denormalized_p1_points[hull_img1.vertices[v], 1]) ), (255,0,0), 3)

            cv2.line(img1_o, ( int(denormalized_p1_points[hull_img1.vertices[v], 0]), int(denormalized_p1_points[hull_img1.vertices[v], 1]) ), \
                (int(denormalized_p1_points[hull_img1.vertices[0], 0]), int(denormalized_p1_points[hull_img1.vertices[0], 1]) ), (255,0,0), 3)

            for v, vertex in enumerate(hull_img2.vertices):
                if v == 0:
                    continue
                cv2.line(img2_o, ( int(denormalized_p2_points[hull_img2.vertices[v-1], 0]), int(denormalized_p2_points[hull_img2.vertices[v-1], 1]) ), \
                    (int(denormalized_p2_points[hull_img2.vertices[v], 0]), int(denormalized_p2_points[hull_img2.vertices[v], 1]) ), (255,0,0), 5)

            cv2.line(img2_o, ( int(denormalized_p2_points[hull_img2.vertices[v], 0]), int(denormalized_p2_points[hull_img2.vertices[v], 1]) ), \
                (int(denormalized_p2_points[hull_img2.vertices[0], 0]), int(denormalized_p2_points[hull_img2.vertices[0], 1]) ), (255,0,0), 5)
    except:
        hull_img1, hull_img2 = None, None    

    return hull_img1, hull_img2

def get_photometric_error(f_img1, f_img2, sampled_points_img1, img1_original, sampled_points_transformed, img2_original, error_threshold):
  errors = []

  for i,pt in enumerate(sampled_points_transformed):
    u,v = sampled_points_img1[i]
    u_,v_ = sampled_points_transformed[i]
    
    error = np.power( \
        np.power(f_img2[0](u_,v_) - f_img1[0](u,v),2) + \
        np.power(f_img2[1](u_,v_) - f_img1[1](u,v),2) + \
        np.power(f_img2[2](u_,v_) - f_img1[2](u,v),2) \
      , 0.5)
    errors.append(error)
  return errors

def tesselate_matches(ransac_count, grid_size, data, im1, im2, img1, img2, matches, p1, p2, patchdataset, flags, ii, jj):
    n, outlier_threshold, debug, error_threshold, num_clusters, sample_matches = [flags['num_samples'], flags['outlier_threshold_percentage'], \
        flags['debug'], flags['lab_error_threshold'], flags['kmeans_num_clusters'], flags['use_kmeans']]

    Ms = []
    triangle_pts_img1 = []
    triangle_pts_img2 = []
    colors = [(int(random.random()*255), int(random.random()*255), int(random.random()*255)) for i in xrange(0,100)]
    t_start_img_loading = timer()

    img1_o = img1
    img2_o = img2
    img1_w = img1
    img2_w = img2
    img1_original = img1
    img2_original = img2
    im1_fn = os.path.basename(im1)
    im2_fn = os.path.basename(im2)
    if False and data.photometric_error_triangle_transformations_exists(im1_fn,im2_fn):
        logger.info('Loading cached triangles')
        triangles_data = data.load_photometric_error_triangle_transformations(im1_fn, im2_fn)
        Ms = triangles_data['Ms']
        triangle_pts_img1 = triangles_data['triangle_pts_img1']
        triangle_pts_img2 = triangles_data['triangle_pts_img2']
    else:
        p1_points = p1[ matches[:,0].astype(np.int) ]
        p2_points = p2[ matches[:,1].astype(np.int) ]
        denormalized_p1_points = features.denormalized_image_coordinates(p1_points, grid_size, grid_size)
        denormalized_p2_points = features.denormalized_image_coordinates(p2_points, grid_size, grid_size)

        # hull_img1, hull_img2 = calculate_convex_hull(data, img1_o, denormalized_p1_points, img2_o, denormalized_p2_points, flags)
        # if hull_img1 is None or hull_img2 is None:
        #     return None, None, 0, np.array([]), np.array([]), np.array([]), np.array([]), None, None, None

        tesselation_vertices_im1 = denormalized_p1_points[:,0:2]
        tesselation_vertices_im2 = denormalized_p2_points[:,0:2]

        if debug and flags['draw_matches']:
            for i, cc in enumerate(tesselation_vertices_im1):
                color_index = random.randint(0,len(colors)-1)
                cv2.circle(img1_o, (int(cc[0]), int(cc[1])), 5, colors[color_index], -1)
                cv2.circle(img2_o, (int(tesselation_vertices_im2[i, 0]), int(tesselation_vertices_im2[i, 1])), 5, colors[color_index], -1)

        try:
            triangles_img1 = Delaunay(tesselation_vertices_im1, qhull_options='Pp Qt', incremental=True)
        except:
            return None, None, 0, np.array([]), np.array([]), np.array([]), np.array([]), None, None, None

        # x = np.linspace(0, grid_size - 1, grid_size).astype(int)
        # y = np.linspace(0, grid_size - 1, grid_size).astype(int)

        # z_img1 = [None] * 3
        # z_img2 = [None] * 3
        # f_img1 = [None] * 3
        # f_img2 = [None] * 3
        # for i in xrange(0,3):
        #     z_img1[i] = img1_original[0:grid_size,0:grid_size,i]
        #     z_img2[i] = img2_original[0:grid_size,0:grid_size,i]
        #     f_img1[i] = interpolate.interp2d(x, y, z_img1[i], kind='cubic')
        #     f_img2[i] = interpolate.interp2d(x, y, z_img2[i], kind='cubic')

        t_start_triangle_loop = timer()
        for s, simplex in enumerate(triangles_img1.simplices):
            color_index = random.randint(0,len(colors)-1)
            pts_img1_ = np.array([ 
                tesselation_vertices_im1[simplex[0], 0:2], \
                tesselation_vertices_im1[simplex[1], 0:2], \
                tesselation_vertices_im1[simplex[2], 0:2], \
                ])
            pts_img2_ = np.array([ 
                tesselation_vertices_im2[simplex[0], 0:2], \
                tesselation_vertices_im2[simplex[1], 0:2], \
                tesselation_vertices_im2[simplex[2], 0:2], \
            ])
            pts_img1 = pts_img1_.astype(np.int32).reshape((-1,1,2))
            pts_img2 = pts_img2_.astype(np.int32).reshape((-1,1,2))

            M = cv2.getAffineTransform(np.float32(pts_img1_),np.float32(pts_img2_))
            triangle_pts_img1.append(pts_img1)
            triangle_pts_img2.append(pts_img2)

            Ms.append(M)

            if debug and flags['draw_triangles']:
                cv2.polylines(img1_o,[pts_img1],True,colors[color_index])
                cv2.polylines(img2_o,[pts_img2],True,colors[color_index])
        Ms = np.array(Ms)
        if False:
            triangle_pts_img1 = np.array(triangle_pts_img1)
            triangle_pts_img2 = np.array(triangle_pts_img2)
            triangles_data = {
                'im1': im1,
                'im2': im2,
                'Ms': Ms,
                'triangle_pts_img1': triangle_pts_img1,
                'triangle_pts_img2': triangle_pts_img2
            }

            data.save_photometric_error_triangle_transformations(im1_fn, im2_fn, triangles_data)
        if debug:
            imgs = np.concatenate((img1_o,img2_o),axis=1)
            data.save_photometric_errors_map('{}-{}-d-{}-{}'.format(os.path.basename(im1), os.path.basename(im2), os.path.basename(im1), ransac_count), img1_o, size=grid_size)
            data.save_photometric_errors_map('{}-{}-d-{}-{}'.format(os.path.basename(im1), os.path.basename(im2), os.path.basename(im2), ransac_count), img2_o, size=grid_size)

    warped_image, masked_image = warp_image(data, Ms, triangle_pts_img1, triangle_pts_img2, img1_w, img2_w, im1, im2, flags, colors, grid_size, save_individual_triangles=True)
    # masked_image = warp_image(data, Ms, triangle_pts_img1, triangle_pts_img2, 255*np.ones(img1_w.shape), np.zeros(img2_w.shape), im1, im2, flags, colors, grid_size, save_individual_triangles=False)

    # try:
    #     warped_image = warp_image(Ms, triangle_pts_img1, triangle_pts_img2, img1_w, img2_w, im1, im2, flags, colors)
    #     masked_image = warp_image(Ms, triangle_pts_img1, triangle_pts_img2, 255*np.ones(img1_w.shape), np.zeros(img2_w.shape), im1, im2, flags, colors)
    # except:
    #     # TODO(raj): debug error with ece_floor5_wall images: 2017-11-22_19-46-21_218.jpeg ---2017-11-22_19-46-33_796.jpeg
    #     return None, None, 0, np.array([]), np.array([]), np.array([]), np.array([]), None, None, None
    
    # masked_image = masked_image[:,:,0]
    # masked_image[masked_image < 0] = 0

    # if False:
    error_map = calculate_error_map(img2_w, warped_image)
    cum_errors = get_l2_errors(error_map)

    histogram_counts, histogram_cumsum, histogram, bins = calculate_photometric_error_histogram(cum_errors)
    polygon_area, polygon_area_percentage = get_polygon_area(masked_image)
    return polygon_area, polygon_area_percentage, len(triangles_img1.simplices), histogram_counts, histogram_cumsum, histogram, bins, error_map, masked_image, warped_image
    # return 

def get_polygon_area(masked_image):
    polygon_area =  1.0*len(np.where(masked_image > 0)[0])
    polygon_area_percentage = polygon_area / (masked_image.shape[0] * masked_image.shape[1])
    return polygon_area, polygon_area_percentage

def get_l2_errors(error_map):
    ii,jj = np.where(error_map > 0)
    cum_errors = []
    for i,_ in enumerate(ii):
        cum_errors.append(error_map[ii[i],jj[i]])
    return cum_errors

def calculate_error_map(img_o, warped_image):
    error_map = np.zeros((img_o.shape[0], img_o.shape[1])).astype(np.int)
    ii,jj,kk = np.where(warped_image > -255)
    for i,_ in enumerate(ii):
        error = np.power( \
            np.power(img_o[ii[i],jj[i],0] - warped_image[ii[i],jj[i],0], 2) + \
            np.power(img_o[ii[i],jj[i],1] - warped_image[ii[i],jj[i],1], 2) + \
            np.power(img_o[ii[i],jj[i],2] - warped_image[ii[i],jj[i],2], 2), \
            0.5 )
        error_map[ii[i],jj[i]] = min(error, 255)
        # error_map[ii[i],jj[i]] = \
        #     np.power(np.power(img_o[ii[i],jj[i],0] - warped_image[ii[i],jj[i],0], 2), 0.5) + \
        #     np.power(np.power(img_o[ii[i],jj[i],1] - warped_image[ii[i],jj[i],1], 2), 0.5) + \
        #     np.power(np.power(img_o[ii[i],jj[i],2] - warped_image[ii[i],jj[i],2], 2), 0.5)
    
    # error_map[error_map > 50] = 255
    # error_map[error_map <= 50] = 0
    return error_map

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def get_resized_image(data, im, grid_size):
    if not data.resized_image_exists(im):
        image_original = cv2.imread(os.path.join(data.data_path,'images',im),cv2.IMREAD_COLOR)
        data.save_resized_image(im_fn=im, image=image_original, grid_size=grid_size)

    image, im_fn, metadata = data.load_resized_image(im)
    
    return image, im_fn, metadata

def get_processed_images(data, im1, im2, grid_size):
    if not data.processed_image_exists(im1, im2) or not data.processed_image_exists(im2, im1):
        im1_a_fn, im2_a_fn, img1_adjusted_denormalized, img2_adjusted_denormalized, m1, m2 = perform_gamma_adjustment(data, im1, im2, grid_size, debug=False)
    else:
        im1_a_fn, img1_adjusted_denormalized, m1 = data.load_processed_image(im1, im2)
        im2_a_fn, img2_adjusted_denormalized, m2 = data.load_processed_image(im1, im2)

    return im1_a_fn, im2_a_fn, img1_adjusted_denormalized, img2_adjusted_denormalized, m1, m2

def calculate_photometric_error_convex_hull(arg):
    ii, jj, patchdataset, data, im1, im2, matches, flags = arg
    best_error = sys.float_info.max
    start_t = timer()
    grid_size = 224
    logger.info('Starting to process {} / {}'.format(im1, im2))
    if flags['masked_tags']:
        p1, f1, c1 = data.load_features_masked(im1)
        p2, f2, c2 = data.load_features_masked(im2)
    else:
        p1, f1, c1 = data.load_features(im1)
        p2, f2, c2 = data.load_features(im2)
  
    mkdir_p(os.path.join(data.data_path,'images-resized'))
    # img1, im1_fn, m1 = get_resized_image(data, im1, grid_size)
    # img2, im2_fn, m2 = get_resized_image(data, im2, grid_size)
    im1_fn, im2_fn, img1, img2, m1, m2 = get_processed_images(data, im1, im2, grid_size)


    denormalized_p1_points = features.denormalized_image_coordinates(p1[:,0:2], m1['width'], m1['height'])
    denormalized_p2_points = features.denormalized_image_coordinates(p2[:,0:2], m2['width'], m2['height'])
    w1_scale = 1.0 * grid_size / m1['width']
    h1_scale = 1.0 * grid_size / m1['height']
    w2_scale = 1.0 * grid_size / m2['width']
    h2_scale = 1.0 * grid_size / m2['height']

    scaled_denormalized_p1_points = np.zeros(denormalized_p1_points.shape)
    scaled_denormalized_p2_points = np.zeros(denormalized_p2_points.shape)
    scaled_denormalized_p1_points[:,0] = 1.0 * denormalized_p1_points[:,0] * w1_scale
    scaled_denormalized_p1_points[:,1] = 1.0 * denormalized_p1_points[:,1] * h1_scale
    scaled_denormalized_p2_points[:,0] = 1.0 * denormalized_p2_points[:,0] * w2_scale
    scaled_denormalized_p2_points[:,1] = 1.0 * denormalized_p2_points[:,1] * h2_scale

    renormalized_p1_points = features.normalized_image_coordinates(scaled_denormalized_p1_points, grid_size, grid_size)
    renormalized_p2_points = features.normalized_image_coordinates(scaled_denormalized_p2_points, grid_size, grid_size)

    best_warped_image = None
    best_error_map = None
    best_masked_image = None
    best_polygon_area, best_polygon_area_percentage, best_total_triangles, best_histogram_counts, best_histogram_cumsum, best_histogram, best_bins = \
        None, None, 0, np.array([]), np.array([]), np.array([]), np.array([])

    wi_fn = '{}-{}-wi'.format(os.path.basename(im1), os.path.basename(im2))
    em_fn = '{}-{}-em'.format(os.path.basename(im1), os.path.basename(im2))
    m_fn = '{}-{}-m'.format(os.path.basename(im1), os.path.basename(im2))
    # print 'here-1'
    if data.photometric_errors_map_exists(wi_fn):
        best_warped_image = data.load_photometric_errors_map(wi_fn)
        best_error_map = data.load_photometric_errors_map(em_fn, grayscale=True)
        best_masked_image = data.load_photometric_errors_map(m_fn, grayscale=True)
        cum_errors = get_l2_errors(best_error_map)
        best_histogram_counts, best_histogram_cumsum, best_histogram, best_bins = calculate_photometric_error_histogram(cum_errors)
        best_polygon_area, best_polygon_area_percentage = get_polygon_area(best_masked_image)
    else:
        ransac_count = 0
        polygon_area, polygon_area_percentage, total_triangles, histogram_counts, histogram_cumsum, histogram, bins, error_map, masked_image, warped_image = \
            tesselate_matches(ransac_count, grid_size, data, im1_fn, im2_fn, img1, img2, matches, renormalized_p1_points, renormalized_p2_points, patchdataset, flags, ii, jj)
        if False:
            for ransac_count in range(0,1):
                # First iteration always has all matches
                if ransac_count == 0:
                    rid = []
                else:
                    random_match_count = np.random.randint(1,int(0.5 * len(matches)))
                    rid = np.sort(np.random.choice(len(matches), random_match_count, replace=False))

                random_matches = np.ones(len(matches)).astype(np.bool)
                random_matches[rid] = False
                # print 'here'
                polygon_area, polygon_area_percentage, total_triangles, histogram_counts, histogram_cumsum, histogram, bins, error_map, masked_image, warped_image = \
                    tesselate_matches(ransac_count, grid_size, data, \
                        im1_fn, im2_fn, \
                        img1, img2, \
                        matches[random_matches], renormalized_p1_points, renormalized_p2_points, patchdataset, flags, ii, jj)
                # print 'here2'
                if warped_image is None or error_map is None or masked_image is None:
                    continue

                error = np.sum(np.multiply(error_map, masked_image)) / np.sum(masked_image > 0)
                if error < best_error:
                    best_rid = rid
                    best_warped_image = warped_image.copy()
                    best_error_map = error_map.copy()
                    best_masked_image = masked_image.copy()
                    best_polygon_area = polygon_area
                    best_polygon_area_percentage = polygon_area_percentage
                    best_total_triangles = total_triangles
                    best_histogram_counts = histogram_counts
                    best_histogram_cumsum = histogram_cumsum
                    best_histogram = histogram
                    best_bins = bins
                    best_error = error


            logger.info('Best error: {} rid: {}'.format(best_error, best_rid))
            end_t = timer()

    return im1, im2, polygon_area, polygon_area_percentage, total_triangles, histogram_counts, histogram_cumsum, histogram, bins
   # return im1, im2, best_polygon_area, best_polygon_area_percentage, best_total_triangles, best_histogram_counts, best_histogram_cumsum, best_histogram, best_bins
    # return im1, im2

def calculate_photometric_errors(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = config['processes']
    args = []
    flags = {'masked_tags': False, 'num_samples': 500, 'outlier_threshold_percentage': 0.7, 'debug': False, 'lab_error_threshold': 50, \
        'use_kmeans': False, 'sampling_method': 'sample_polygon_uniformly', 'draw_hull': False, 'draw_matches': False, 'draw_triangles': False, 'draw_points': False, \
        'processes': processes, 'draw_outliers': False, 'kmeans_num_clusters': None }

    if data.photometric_errors_exists():
        logger.info('Photometric errors exist!')
        return
    logger.info('Calculating photometric errors...')
    for im1 in images:
        im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
        for im2 in im1_all_robust_matches:
            rmatches = im1_all_robust_matches[im2]
            if len(rmatches) == 0:
                continue
            # if im1 == 'DSC_0286.JPG' and im2 == 'DSC_0289.JPG' or im1 == 'DSC_0286.JPG' and im2 == 'DSC_0288.JPG' or im1 == 'DSC_0286.JPG' and im2 == 'DSC_0287.JPG':
            # if im1 == 'DSC_1744.JPG' and im2 == 'DSC_1746.JPG' or im1 == 'DSC_1744.JPG' and im2 == 'DSC_1800.JPG':
            # if im1 == 'DSC_1773.JPG' and im2 == 'DSC_1778.JPG':

            # if im1 == 'DSC_1744.JPG' and im2 == 'DSC_1800.JPG':
                
            # if im1 == 'DSC_1744.JPG' and im2 == 'DSC_1746.JPG':# or \
            #     # im1 == 'DSC_1744.JPG' and im2 == 'DSC_1780.JPG' or \
            #     # im1 == 'DSC_1744.JPG' and im2 == 'DSC_1800.JPG':
            # if im1 == 'DSC_1761.JPG' and im2 == 'DSC_1762.JPG':

            # if im1 == 'DSC_1744.JPG' and im2 == 'DSC_1746.JPG' or \
            #     im1 == 'DSC_1744.JPG' and im2 == 'DSC_1800.JPG' or \
            #     im1 == 'DSC_1773.JPG' and im2 == 'DSC_1778.JPG' or \
            #     im1 == 'DSC_1744.JPG' and im2 == 'DSC_1780.JPG' or \
            #     im1 == 'DSC_1761.JPG' and im2 == 'DSC_1762.JPG':
                
            #     args.append([0, 1, None, ctx.data, im1, im2, rmatches[:, 0:2].astype(int), flags])
            #     args.append([0, 1, None, ctx.data, im2, im1, np.concatenate((rmatches[:, 1].reshape((-1,1)), rmatches[:, 0].reshape((-1,1))), axis=1).astype(int), flags])

            # if im1 == 'DSC_1744.JPG' and im2 == 'DSC_1800.JPG':
            #     args.append([0, 1, None, ctx.data, im1, im2, rmatches[:, 0:2].astype(int), flags])
                # args.append([0, 1, None, ctx.data, im2, im1, np.concatenate((rmatches[:, 1].reshape((-1,1)), rmatches[:, 0].reshape((-1,1))), axis=1).astype(int), flags])
            # if im1 == 'DSC_1770.JPG' and im2 == 'DSC_1779.JPG':
            #     args.append([0, 1, None, ctx.data, im1, im2, rmatches[:, 0:2].astype(int), flags])
            #     args.append([0, 1, None, ctx.data, im2, im1, np.concatenate((rmatches[:, 1].reshape((-1,1)), rmatches[:, 0].reshape((-1,1))), axis=1).astype(int), flags])

            # if im1 == 'DSC_1744.JPG' and im2 == 'DSC_1746.JPG' or \
            #     im1 == 'DSC_1744.JPG' and im2 == 'DSC_1800.JPG' or \
            #     im1 == 'DSC_1773.JPG' and im2 == 'DSC_1778.JPG' or \
            #     im1 == 'DSC_1744.JPG' and im2 == 'DSC_1780.JPG' or \
            #     im1 == 'DSC_1761.JPG' and im2 == 'DSC_1762.JPG':

            args.append([0, 1, None, ctx.data, im1, im2, rmatches[:, 0:2].astype(int), flags])
            args.append([0, 1, None, ctx.data, im2, im1, np.concatenate((rmatches[:, 1].reshape((-1,1)), rmatches[:, 0].reshape((-1,1))), axis=1).astype(int), flags])

    t_start = timer()
    p_results = []
    results = {}
    p = Pool(processes)
    logger.info('Using {} thread(s)'.format(processes))
    # print args
    if processes == 1:
        for a, arg in enumerate(args):
            # logger.info('Finished processing photometric errors: {} / {} : {} / {}'.format(a, len(args), arg[4], arg[5]))
            p_results.append(calculate_photometric_error_convex_hull(arg))
    else:
        p_results = p.map(calculate_photometric_error_convex_hull, args)
        p.close()

    for r in p_results:
        # if False:
        im1, im2, polygon_area, polygon_area_percentage, total_triangles, histogram_counts, histogram_cumsum, histogram, bins = r
        if polygon_area is None or polygon_area_percentage is None:
            continue

        element = {'polygon_area': polygon_area, 'polygon_area_percentage': polygon_area_percentage, \
          'total_triangles': total_triangles, 'histogram': histogram, 'histogram-cumsum': histogram_cumsum, \
          'histogram-counts': histogram_counts, 'bins': bins}
        if im1 not in results:
            results[im1] = {}
        results[im1][im2] = element

        # im1, im2 = r
        # if im1 not in results:
        #     results[im1] = {}
        # results[im1][im2] = True

    data.save_photometric_errors(results)

# def save_processed_image(data, im_fn, img, grid_size):
#     mkdir_p(os.path.join(data.data_path,'images-resized-processed'))
#     if grid_size is None:
#         cv2.imwrite(os.path.join(data.data_path,'images-resized-processed',im_fn), img)
#     else:
#         cv2.imwrite(os.path.join(data.data_path,'images-resized-processed',im_fn), cv2.resize(img, (grid_size, grid_size)))

def perform_gamma_adjustment(data, im1, im2, grid_size, debug=False):
    im1_fn = os.path.basename(im1)
    im2_fn = os.path.basename(im2)

    img1, _, m1 = get_resized_image(data, im1, grid_size)
    img2, _, m2 = get_resized_image(data, im2, grid_size)

    img1_normalized = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img2_normalized = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    gamma_im1 = 1.0 * max(np.sum(img1_normalized), np.sum(img2_normalized)) / np.sum(img1_normalized)
    gamma_im2 = 1.0 * max(np.sum(img1_normalized), np.sum(img2_normalized)) / np.sum(img2_normalized)

    # print img1_normalized

    img1_adjusted = np.power(img1_normalized, 1.0/gamma_im1)
    img2_adjusted = np.power(img2_normalized, 1.0/gamma_im2)

    img1_adjusted_denormalized = cv2.normalize(img1_adjusted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16S)
    img2_adjusted_denormalized = cv2.normalize(img2_adjusted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16S)
    # print img1_adjusted_denormalized
    # print '{} / {} : {}'.format(np.sum(img1), np.sum(img2), 1.0 * )


    # if debug: 
    #     imgs_u = np.concatenate((img1,cv2.resize(img2, (img1.shape[1], img1.shape[0]))),axis=1)
    #     imgs_p = np.concatenate((img1_adjusted_denormalized,cv2.resize(img2_adjusted_denormalized, (img1_adjusted_denormalized.shape[1], img1_adjusted_denormalized.shape[0]))),axis=1)
    #     imgs_ = np.concatenate((imgs_u, cv2.resize(imgs_p, (imgs_u.shape[1], imgs_u.shape[0]))),axis=1)
    #     save_processed_image(data, im_fn='{}-{}.png'.format(im1_fn, im1_fn), img=imgs_, grid_size=None)
    
    # im1_a_fn = '{}---{}-{}-processed.png'.format(im1_fn, im1_fn, im2_fn)
    # im2_a_fn = '{}---{}-{}-processed.png'.format(im2_fn, im1_fn, im2_fn)
    im1_a_fn = data.save_processed_image(im1_fn=im1_fn, im2_fn=im2_fn, image=img1_adjusted_denormalized, grid_size=grid_size)
    im2_a_fn = data.save_processed_image(im1_fn=im2_fn, im2_fn=im1_fn, image=img2_adjusted_denormalized, grid_size=grid_size)
    # save_processed_image(data, im_fn='{}---{}-{}-unprocessed.png'.format(im1_fn, im1_fn, im2_fn), img=img1, grid_size=grid_size)
    # save_processed_image(data, im_fn='{}---{}-{}-unprocessed.png'.format(im2_fn, im1_fn, im2_fn), img=img2, grid_size=grid_size)

    return im1_a_fn, im2_a_fn, img1_adjusted_denormalized, img2_adjusted_denormalized, m1, m2

def output_image_keypoints(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = config['processes']
    args = []

    logger.info('Outputting image keypoint maps...')
    for im in images:
        p, f, c = ctx.data.load_features(im)
        _, feature_map_im = calculate_spatial_entropy(p, 224)
        data.save_feature_map('feature---{}'.format(im), feature_map_im)

def calculate_nbvs(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    cached_p = {}
    nbvs = {}
    
    if data.nbvs_exists():
        logger.info('NBVS exist!')
        return

    logger.info('Calculating NBVS...')
    for im1 in images:
        im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
        
        if im1 not in cached_p:
            p1, f1, c1 = ctx.data.load_features(im1)
            cached_p[im1] = p1
        else:
            p1 = cached_p[im1]

        for im2 in im1_all_robust_matches:
            rmatches = im1_all_robust_matches[im2]
            
            p2, f2, c2 = ctx.data.load_features(im2)
            if im2 not in cached_p:
                p2, f2, c2 = ctx.data.load_features(im2)
                cached_p[im2] = p2
            else:
                p2 = cached_p[im2]
            if len(rmatches) == 0:
                continue

            p1_ = p1[rmatches[:, 0].astype(int)]
            p2_ = p2[rmatches[:, 1].astype(int)]

            nbvs_im1 = next_best_view_score(p1_)
            nbvs_im2 = next_best_view_score(p2_)

            if im1 not in nbvs:
                nbvs[im1] = {}
            nbvs[im1][im2] = {
                'nbvs_im1': nbvs_im1, 'nbvs_im2': nbvs_im2
                }
    data.save_nbvs(nbvs)

# def triplet_arguments(fns, Rs):
#     unique_filenames = sorted(list(set(fns[:,0]).union(set(fns[:,1]))))
#     for i, fn1 in enumerate(unique_filenames):
#         yield unique_filenames, fns, Rs, i, fn1

# def triplet_pairwise_arguments(fns, t_fns, t_errors, processes):
#     fns_ = []
#     args = []
#     for i, (im1,im2) in enumerate(fns):
#         fns_.append([im1, im2])
#         if len(fns_) >= math.ceil(len(fns) / processes):
#             args.append([None, fns_, t_fns, t_errors, False])
#             fns_ = []
    
#     if len(fns_) > 0:
#         args.append([None, fns_, t_fns, t_errors, False])

#     return args

# def calculate_rotation_triplet_errors(arg):
#   unique_filenames, fns, Rs, i, fn1 = arg
#   results = {}

#   t_start = timer()
#   for j, fn2 in enumerate(unique_filenames):
#     if j <= i:
#       continue

#     if fn2 not in results:
#       results[fn2] = {}
#     fn12_ris = np.where((fns[:,0] == fn1) & (fns[:,1] == fn2) | (fns[:,0] == fn2) & (fns[:,1] == fn1))[0]
#     if len(fn12_ris) == 0:
#       continue

#     if fns[fn12_ris, 0][0] == fn1:
#       R12 = np.matrix(Rs[fn12_ris].reshape(3,3))
#     else:
#       R12 = np.matrix(Rs[fn12_ris].reshape(3,3)).T

#     for k, fn3 in enumerate(unique_filenames):
#       if k <= j:
#         continue

#       fn23_ris = np.where((fns[:,0] == fn2) & (fns[:,1] == fn3) | (fns[:,0] == fn3) & (fns[:,1] == fn2))[0]
#       if len(fn23_ris) == 0:
#         continue

#       if fns[fn23_ris, 0][0] == fn2:
#         R23 = np.matrix(Rs[fn23_ris].reshape(3,3))
#       else:
#         R23 = np.matrix(Rs[fn23_ris].reshape(3,3)).T

#       fn13_ris = np.where((fns[:,0] == fn1) & (fns[:,1] == fn3) | (fns[:,0] == fn3) & (fns[:,1] == fn1))[0]

#       if len(fn13_ris) == 0:
#         continue
#       if fns[fn13_ris, 0][0] == fn1:
#         R13 = np.matrix(Rs[fn13_ris].reshape(3,3))
#       else:
#         R13 = np.matrix(Rs[fn13_ris].reshape(3,3)).T

#       R11 = R12 * R23 * R13.T
#       error = np.arccos((np.trace(R11) - 1.0)/2.0)
#       if np.isnan(error):
#         error = -1.0

#       results[fn2][fn3] = {'error': math.fabs(error), 'R11': R11.tolist(), 'triplet': '{}--{}--{}'.format(fn1,fn2,fn3)}
#   return fn1, results

def calculate_triplet_error_histogram(errors, im1, im2, output_dir, debug):
    histogram, bins = np.histogram(np.array(errors), bins=80, range=(0.0,np.pi/4.0))
    epsilon = 0.000000001
    return histogram.tolist(), \
        np.cumsum(np.round((1.0 * histogram / (np.sum(histogram) + epsilon)), 2)).tolist(), \
        np.round((1.0 * histogram / (np.sum(histogram) + epsilon)), 2).tolist(), \
        bins.tolist()

# def flatten_triplets(triplets):
#   flattened_triplets = {}
#   fns1 = []
#   fns2 = []
#   fns3 = []
#   errors = []
#   for t1 in triplets:
#       for t2 in triplets[t1]:
#         for t3 in triplets[t1][t2]:
#           triplet = [t1, t2, t3]
#           flattened_triplets['--'.join(triplet)] = triplets[t1][t2][t3]['error']
#           fns1.append(t1)
#           fns2.append(t2)
#           fns3.append(t3)
#           errors.append(triplets[t1][t2][t3]['error'])

#   return np.array(fns1), np.array(fns2), np.array(fns3), np.array(errors)

# def calculate_triplet_pairwise_errors(arg):
#   output_dir, fns, t_fns, t_errors, debug = arg
#   errors = {}
#   histograms = {}
#   histograms_list = []
#   stime = timer()
#   for i, (im1,im2) in enumerate(fns):
#     relevant_indices = np.where((t_fns[:,0] == im1) & (t_fns[:,1] == im2) | (t_fns[:,0] == im1) & (t_fns[:,2] == im2) | \
#       (t_fns[:,1] == im1) & (t_fns[:,2] == im2))[0]

#     histogram_counts, histogram_cumsum, histogram, bins = calculate_triplet_error_histogram(t_errors[relevant_indices], im1, im2, output_dir, debug=debug)
#     if im1 not in histograms:
#         histograms[im1] = {}
#     histograms[im1][im2] = { 'im1': im1, 'im2': im2, 'histogram': histogram, 'histogram-cumsum': histogram_cumsum, 'histogram-counts': histogram_counts, 'bins': bins }
#     histograms_list.append({ 'im1': im1, 'im2': im2, 'histogram': histogram, 'histogram-cumsum': histogram_cumsum, 'histogram-counts': histogram_counts, 'bins': bins })
#   return histograms, histograms_list

# def calculate_triplet_errors(ctx):
#     data = ctx.data
#     cameras = data.load_camera_models()
#     images = data.images()
#     exifs = ctx.exifs
#     config = data.config
#     processes = ctx.data.config['processes']
#     threshold = config['robust_matching_threshold']
#     cached_p = {}
#     fns = []
#     Rs = []
#     transformations = {}
    
#     if data.triplet_errors_exists():
#         logger.info('Triplet errors exist!')
#         return

#     if data.transformations_exists():
#         transformations = data.load_transformations()
#     else:
#         transformations, t_num_pairs = calculate_transformations(ctx)

#     for im1 in transformations:
#         for im2 in transformations[im1]:
#             R = transformations[im1][im2]['rotation']
#             fns.append(np.array([im1, im2]))
#             Rs.append(np.array(R).reshape((1,-1)))
    
#     logger.info('Calculating triplet errors...')
#     args = triplet_arguments(np.array(fns), np.array(Rs))
#     triplet_results = {}
#     p = Pool(processes)
#     if processes > 1:
#         t_results = p.map(calculate_rotation_triplet_errors, args)
#     else:
#         t_results = []
#         for arg in args:
#             t_results.append(calculate_rotation_triplet_errors(arg))
#     for r in t_results:
#         fn1, triplets = r
#         triplet_results[fn1] = triplets
#     data.save_triplet_errors(triplet_results)

#     logger.info('Calculating triplet pairwise errors...')
#     t_fns1, t_fns2, t_fns3, t_errors = flatten_triplets(triplet_results)
#     t_fns = np.concatenate((t_fns1.reshape(-1,1), t_fns2.reshape(-1,1), t_fns3.reshape(-1,1)), axis=1)
#     args = triplet_pairwise_arguments(np.array(fns), t_fns, t_errors, processes)
#     p = Pool(processes)
#     p_results = []
#     t_start = timer()
#     if processes == 1:
#       for arg in args:
#         p_results.append(calculate_triplet_pairwise_errors(arg))
#     else:
#       p_results = p.map(calculate_triplet_pairwise_errors, args)
#     p.close()
#     triplet_pairwise_results = {}
#     for i, r in enumerate(p_results):
#       histograms, histograms_list = r
#       for k in histograms:
#         if k not in triplet_pairwise_results:
#             triplet_pairwise_results[k] = histograms[k]
#         else:
#             triplet_pairwise_results[k].update(histograms[k])
#     data.save_triplet_pairwise_errors(triplet_pairwise_results)

def get_rotation_matrix(transformations, im1, im2):
    R = None
    if im1 in transformations and im2 in transformations[im1]:
        R = np.matrix(np.array(transformations[im1][im2]['rotation']).reshape((1,-1)).reshape(3,3))
    if im2 in transformations and im1 in transformations[im2]:
        R = np.matrix(np.array(transformations[im2][im1]['rotation']).reshape((1,-1)).reshape(3,3)).T
    return R

def calculate_consistency_errors_per_node(arg):
    ctx, i, transformations, n1, G, cutoff, debug = arg
    n1_histograms = {}
    # print sorted(G.nodes())
    G = ctx.data.load_graph('rm', 15)
    for j, n2 in enumerate(sorted(G.nodes())):
        errors = []
        if j <= i:
            continue
        paths = nx.all_simple_paths(G, n1, n2, cutoff=cutoff)
        # if n2 != 'DSC_1800.JPG':
        #     continue
        # print ('\t\t\t**************{} - {} : rm={}**************'.format(n1, n2, nx.shortest_path(G_rm_cost, n1, n2, weight='weight') ))
        for path_counter, p in enumerate(paths):
            if len(p) == 2:
                continue
            # print ('cutoff: {}    p: {}'.format(cutoff, p))
            # import sys; sys.exit(1)
            if debug:
                print ('\t\t{}'.format(p))
            R_ = np.identity(3)
            complete_chain = True
            for step, n in enumerate(p):
                if step == 0:
                    continue
                im1 = p[step-1]
                im2 = p[step]
                edge_data = G.get_edge_data(im1, im2)
                R = get_rotation_matrix(transformations, im1, im2)
                if edge_data is not None and R is not None:
                    if debug:
                        print ('\t\t\t{} - {} : rm={}'.format(im1, im2, edge_data['weight']))
                    R_ = R_ * R
                else:
                    if debug:
                        print ('\t\t\t{} - {} : rm=-'.format(im1, im2))
                    complete_chain = False
                    break

            im1 = p[-1]
            im2 = p[0]
            edge_data = G.get_edge_data(im1, im2)
            R = get_rotation_matrix(transformations, im1, im2)
            if complete_chain and edge_data is not None and R is not None:
                if debug:
                    print ('\t\t\t{} - {} : rm={}'.format(im1, im2, edge_data['weight']))
                R_ = R_ * R
                error = np.arccos((np.trace(R_) - 1.0)/2.0) / len(p)
                errors.append(error)
                if debug:
                    print ('\t\tConsistency error: {}'.format(error * 180.0/np.pi))
            else:
                if debug:
                    print ('\t\tConsistency error: path invalid')

        histogram_counts, histogram_cumsum, histogram, bins = calculate_triplet_error_histogram(errors, n1, n2, output_dir=None, debug=debug)
        n1_histograms[n2] = { 'im1': n1, 'im2': n2, 'histogram': histogram, 'histogram-cumsum': histogram_cumsum, 'histogram-counts': histogram_counts, 'bins': bins }
        # histograms_list.append({ 'im1': n1, 'im2': n2, 'histogram': histogram, 'histogram-cumsum': histogram_cumsum, 'histogram-counts': histogram_counts, 'bins': bins })

    # if debug:
    logger.info('\tProcessed file #{}/{}'.format(i, len(G.nodes())))
    return n1, n1_histograms

def formulate_paths(ctx, transformations, cutoff, edge_threshold):
    data = ctx.data
    processes = ctx.data.config['processes']
    graph_label = 'rm'
    debug = False
    histograms = {}

    if data.graph_exists('rm', edge_threshold):
        G = data.load_graph('rm', edge_threshold)
    else:
        num_rmatches, _, _ = rmatches_adapter(data)
        G = opensfm.commands.formulate_graphs.formulate_graph([data, data.images(), num_rmatches, graph_label, edge_threshold])
        data.save_graph(G, 'rm', edge_threshold)

    args = []
    for i, n1 in enumerate(sorted(G.nodes())):
        # if n1 == 'DSC_1744.JPG':
        args.append([ctx, i, transformations, n1, G, cutoff, debug])

    p = Pool(processes)
    p_results = []
    if processes == 1:    
        for arg in args:
            p_results.append(calculate_consistency_errors_per_node(arg))
    else:
        p_results = p.map(calculate_consistency_errors_per_node, args)

    for n1, histogram in p_results:
        histograms[n1] = histogram

    return histograms

def calculate_consistency_errors(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = ctx.data.config['processes']
    threshold = config['robust_matching_threshold']
    fns = []
    Rs = []
    transformations = {}
    cutoffs = [2, 3]
    # graph_rm_thresholds = [15, 24, 30, 50, 100]
    graph_rm_thresholds = [15]
    all_cutoffs_exist = True
    for edge_threshold in graph_rm_thresholds:
        for cutoff in cutoffs:
            if not data.consistency_errors_exists(cutoff, edge_threshold):
                all_cutoffs_exist = False

    if all_cutoffs_exist:
        logger.info('Consistency errors for all cutoffs exist!')
        return

    if data.transformations_exists():
        transformations = data.load_transformations()
    else:
        transformations, t_num_pairs = calculate_transformations(ctx)
    
    logger.info('Calculating consistency errors...')
    for edge_threshold in graph_rm_thresholds:
        for cutoff in cutoffs:#
            if data.consistency_errors_exists(cutoff, edge_threshold):
                continue
            s_time = timer()
            logger.info('Starting to formulate consistency errors - Cutoff: {} Edge threshold: {}'.format(cutoff, edge_threshold))
            histograms = formulate_paths(ctx, transformations, cutoff=cutoff, edge_threshold=edge_threshold)
            logger.info('Finished formulating consistency errors - Cutoff: {} Edge threshold: {}   Time: {}'.format(cutoff, edge_threshold, timer() - s_time))
            data.save_consistency_errors(histograms, cutoff=cutoff, edge_threshold=edge_threshold)

def get_rmatches_and_edge_costs(G, im1, im2, num_rmatches):
    edge_data = G.get_edge_data(im1, im2)
    if edge_data and 'weight' in edge_data:
        edge_cost = edge_data['weight']
    else:
        edge_cost = 1000000000.0

    if im1 in num_rmatches and im2 in num_rmatches[im1]:
        rmatches = num_rmatches[im1][im2]
    elif im2 in num_rmatches and im1 in num_rmatches[im2]:
        rmatches = num_rmatches[im2][im1]
    else:
        rmatches = 0

    return rmatches, edge_cost

def shortest_path_per_image(arg):
    shortest_paths = {}
    data, i, im1, images, G, num_rmatches = arg
    
    for j,im2 in enumerate(images):
        if j <= i:
            continue
        rmatches, edge_cost = get_rmatches_and_edge_costs(G, im1, im2, num_rmatches)
        shortest_path = nx.shortest_path(G, im1, im2, weight='weight')
        path = {}
        for k, _ in enumerate(shortest_path):
            if k == 0:
                continue
            node_rmatches, node_edge_cost = get_rmatches_and_edge_costs(G, shortest_path[k-1], shortest_path[k], num_rmatches)
            path['{}---{}'.format(shortest_path[k-1], shortest_path[k])] = {'rmatches': node_rmatches, 'cost': node_edge_cost}

        shortest_paths[im2] = {'rmatches': rmatches, 'path': path, 'shortest_path': shortest_path, 'cost': edge_cost}
    return im1, shortest_paths

def calculate_shortest_paths(ctx):
    data = ctx.data
    processes = ctx.data.config['processes']
    images = sorted(data.images())
    shortest_paths = {}
    edge_threshold = 0
    graph_label = 'rm-cost'
    seq_cost_factor = ctx.sequence_cost_factor
    graph_label_with_sequences = 'rm-seq-cost-{}'.format(seq_cost_factor)
    
    num_rmatches, num_rmatches_cost, im_num_rmatches_cost_with_sequences = rmatches_adapter(data, {'apply_sequence_ranks': True, 'sequence_cost_factor': seq_cost_factor})

    # if data.graph_exists(graph_label, edge_threshold) and data.graph_exists(graph_label_with_sequences, edge_threshold):
    #     G = data.load_graph(graph_label, edge_threshold)
    #     G_ = data.load_graph(graph_label_with_sequences, edge_threshold)
    # else:
    G = opensfm.commands.formulate_graphs.formulate_graph([data, images, num_rmatches_cost, graph_label, edge_threshold])
    G_ = opensfm.commands.formulate_graphs.formulate_graph([data, images, im_num_rmatches_cost_with_sequences, graph_label_with_sequences, edge_threshold])
    data.save_graph(G, graph_label, edge_threshold)
    data.save_graph(G_, graph_label_with_sequences, edge_threshold)


    for datum in [[G, graph_label], [G_, graph_label_with_sequences]]:
        args = []
        graph, label = datum
        for i,im1 in enumerate(images):
            args.append([data, i, im1, images, graph, num_rmatches])
        
        p = Pool(processes)
        p_results = []
        if processes == 1:    
            for arg in args:
                p_results.append(shortest_path_per_image(arg))
        else:
            p_results = p.map(shortest_path_per_image, args)

        for im, im_shortest_paths in p_results:
            shortest_paths[im] = im_shortest_paths    

        data.save_shortest_paths(shortest_paths, label=label)

def calculate_sequence_ranks(ctx):
    data = ctx.data
    images = sorted(data.images())
    sequence_distances = {}
    sequence_ranks = {}
    seq_fn_list = {}
    seq_dist_list = {}
    for i,im1 in enumerate(images):
        if im1 not in sequence_distances:
            sequence_distances[im1] = {}
            sequence_ranks[im1] = {}
            seq_fn_list[im1] = []
            seq_dist_list[im1] = []
        im1_t = int(''.join(re.findall(r'\d+', im1)))
        for j,im2 in enumerate(images):
            if j == i:
                continue
            im2_t = int(''.join(re.findall(r'\d+', im2)))

            distance = math.fabs(im1_t - im2_t)
            sequence_distances[im1][im2] = distance
            seq_fn_list[im1].append(im2)
            seq_dist_list[im1].append(distance)


        sorted_dist_indices = np.array(seq_dist_list[im1]).argsort()#.argsort()
        seq_fn_list[im1] = np.array(seq_fn_list[im1])[sorted_dist_indices]
        seq_dist_list[im1] = np.array(seq_dist_list[im1])[sorted_dist_indices]

        # add ranks
        for j,im2 in enumerate(seq_fn_list[im1]):
            sequence_ranks[im1][im2] = {'rank': j, 'distance': seq_dist_list[im1][j]}
    
    data.save_sequence_ranks(sequence_ranks)

    return sequence_ranks
    
##############################################################################################
##############################################################################################
# These functions are pretty much the same as what's in opensfm but have "additional"        #
# parameters and/or return values to return additional information (like size, angles, etc.) #
##############################################################################################
##############################################################################################
def _convert_matches_to_vector(matches, distances=False):
    if distances:
        matches_vector = np.zeros((len(matches),3),dtype=np.float)
        k = 0
        for mm,d in matches:
            matches_vector[k,0] = mm.queryIdx
            matches_vector[k,1] = mm.trainIdx
            matches_vector[k,2] = d
            k = k+1
    else:
        matches_vector = np.zeros((len(matches),2),dtype=np.int)
        k = 0
        for mm in matches:
            matches_vector[k,0] = mm.queryIdx
            matches_vector[k,1] = mm.trainIdx
            k = k+1
    return matches_vector

def match_lowe(index, f2, config, ratio, distances=False):
    search_params = dict(checks=config['flann_checks'])
    results, dists = index.knnSearch(f2, 2, params=search_params)
    squared_ratio = ratio**2  # Flann returns squared L2 distances
    good = dists[:, 0] < squared_ratio * dists[:, 1]
    matches = zip(results[good, 0], good.nonzero()[0])
    matches_with_distances = zip(results[good, 0], good.nonzero()[0], dists[good, 0]/dists[good, 1])
    if distances:
        return np.array(matches_with_distances, dtype=np.float)
    return np.array(matches, dtype=np.float)

def match_lowe_bf(f1, f2, ratio, distances=False):
    assert(f1.dtype.type==f2.dtype.type)
    if (f1.dtype.type == np.uint8):
        matcher_type = 'BruteForce-Hamming'
    else:
        matcher_type = 'BruteForce'
    matcher = cv2.DescriptorMatcher_create(matcher_type)
    matches = matcher.knnMatch(f1, f2, k=2)

    good_matches = []
    good_matches_with_distances = []
    for match in matches:
        if match and len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
                good_matches_with_distances.append((m,m.distance/n.distance))
    if distances:
        good_matches_ = _convert_matches_to_vector(good_matches_with_distances, distances=True).astype(np.float)
    else:
        good_matches_ = _convert_matches_to_vector(good_matches, distances=False).astype(np.int)

    return good_matches_

def unthresholded_match_symmetric(fi, indexi, fj, indexj, config):
    if config['matcher_type'] == 'FLANN':
        matches_ijd = [(a,b,d) for a,b,d in match_lowe(indexi, fj, config, ratio=1.0, distances=True)]
        matches_jid = [(b,a,d) for a,b,d in match_lowe(indexj, fi, config, ratio=1.0, distances=True)]
    else:
        matches_ijd = [(a,b,d) for a,b,d in match_lowe_bf(fi, fj, ratio=1.0, distances=True)]
        matches_jid = [(b,a,d) for a,b,d in match_lowe_bf(fj, fi, ratio=1.0, distances=True)]

    matches_ij = [(a,b) for a,b,d in matches_ijd]
    matches_ji = [(a,b) for a,b,d in matches_jid]

    matches = set(matches_ij).intersection(set(matches_ji))
    # Reformat matches to appear as index1, index2, distance1, distance2
    # Not done in a super efficient way
    matches_dists = []
    for a, b in list(matches):
        for a1, b1, d1 in matches_ijd:
          if a == a1 and b == b1:
            for a2, b2, d2 in matches_jid:
              if a == a2 and b == b2:
                matches_dists.append((a, b, d1, d2))
    return np.array(list(matches_dists), dtype=np.float)

def _compute_inliers_bearings(b1, b2, T, min_thresh=0.01, max_thresh=0.01):
    R = T[:, :3]
    t = T[:, 3]
    p = pyopengv.triangulation_triangulate(b1, b2, t, R)

    br1 = p.copy()
    br1 /= np.linalg.norm(br1, axis=1)[:, np.newaxis]

    br2 = R.T.dot((p - t).T).T
    br2 /= np.linalg.norm(br2, axis=1)[:, np.newaxis]

    err1 = multiview.vector_angle_many(br1, b1)
    err2 = multiview.vector_angle_many(br2, b2)

    ok1 = err1 < min_thresh
    ok2 = err2 < min_thresh

    nok1 = err1 >= max_thresh
    nok2 = err2 >= max_thresh

    return ok1 * ok2, nok1 + nok2, err1, err2
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

def compute_gt_inliers(data, gt_poses, im1, im2, p1, p2, camera1, camera2, matches, ratio):
    if len(matches) < 8:
        return np.array([])

    p1_ = p1[matches[:, 0].astype(int)][:, :2].copy()
    p2_ = p2[matches[:, 1].astype(int)][:, :2].copy()
    dist1 = matches[:, 2].copy()
    dist2 = matches[:, 3].copy()

    b1 = camera1.pixel_bearings(p1_)
    b2 = camera2.pixel_bearings(p2_)
   
    R_ = gt_poses['R2'].dot(gt_poses['R1'].T)
    t_ = gt_poses['R1'].dot(-gt_poses['R2'].T.dot(gt_poses['t2']) + gt_poses['R1'].T.dot(gt_poses['t1']))
    
    # R_ = gt_poses['R2'].dot(gt_poses['R1'].T)
    # t_ = gt_poses['R2'].dot(-gt_poses['R2'].T.dot(gt_poses['t2']) + gt_poses['R1'].T.dot(gt_poses['t1']))

    T_ = np.empty((3, 4))
    T_[0:3,0:3] = R_.T
    T_[0:3,3] = t_[0:3]

    inliers, outliers, n1, n2 = _compute_inliers_bearings(b1, b2, T_, \
        data.config.get('error_inlier_threshold'), data.config.get('error_outlier_threshold'))

    if False:
        T_o = pyopengv.relative_pose_ransac(b1, b2, "STEWENIUS", \
            1 - np.cos(data.config['robust_matching_threshold']), 1000)
        inliers_o, outliers_o, n1_o, n2_o = _compute_inliers_bearings(b1, b2, T_o, \
            data.config.get('error_inlier_threshold'), data.config.get('error_outlier_threshold'))

        logger.debug('\t\tInliers: ' + str(len(matches[inliers])) + '\tOutliers: ' + str(len(matches[outliers])) + '\t\t\tInliers-Original: ' + str(len(matches[inliers_o])) + '\tOutliers-Original: ' + str(len(matches[outliers_o])) +'\t\t\tTotal: ' + str(len(matches)))

    logger.debug('{} - {} has {} candidate unthresholded matches.  Inliers: {}  Outliers: {}  Total: {}'.format( \
        im1, im2, len(matches), len(matches[inliers]), len(matches[outliers]), len(matches)))
    return matches[inliers], matches[outliers], np.array(n1), np.array(n2), dist1, dist2

def get_camera(d, cameras, reconstruction_gt):
    if d['camera'] in cameras:
        camera = cameras[d['camera']]
    else:
        camera = cameras[cameras.keys()[0]] # Pick the first camera

    camera.k1 = 0.0 if camera.k1 is None else camera.k1
    camera.k2 = 0.0 if camera.k2 is None else camera.k2
    camera.k1_prior = 0.0 if camera.k1_prior is None else camera.k1_prior
    camera.k2_prior = 0.0 if camera.k2_prior is None else camera.k2_prior
    camera.focal = 0.85 if camera.focal is None else camera.focal
    camera.focal_prior = 0.85 if camera.focal_prior is None else camera.focal_prior
    return camera


def compute_matches_using_gt_reconstruction(args):
    data, im1, reconstruction_gt, lowes_ratio, error_inlier_threshold, error_outlier_threshold = \
        args

    im1_unthresholded_matches = {}
    im1_unthresholded_inliers = {}
    im1_unthresholded_outliers = {}
    im1_unthresholded_features = {}

    if data.unthresholded_matches_exists(im1):
        im1_unthresholded_matches = data.load_unthresholded_matches(im1)

    for d,recon in enumerate(reconstruction_gt):
        if im1 not in recon.shots.keys():
            continue

        cameras = recon.cameras
        d1 = data.load_exif(im1)
        camera1 = get_camera(d1, cameras, reconstruction_gt)
        R1 = recon.shots[im1].pose.get_rotation_matrix()
        t1 = recon.shots[im1].pose.translation
        
        im_all_matches, _, _ = data.load_all_matches(im1)

        for im2 in im_all_matches.keys():
            if im2 not in recon.shots.keys():
                continue
            # if im2 != 'DSC_1804.JPG' and im2 != 'DSC_1762.JPG':
            #     continue
            d2 = data.load_exif(im2)
            camera2 = get_camera(d2, cameras, reconstruction_gt)
            R2 = recon.shots[im2].pose.get_rotation_matrix()
            t2 = recon.shots[im2].pose.translation
            gt_poses = {'R1': R1, 't1': t1, 'R2': R2, 't2': t2}

            # symmetric matching
            p1, f1, c1 = data.load_features(im1)
            p2, f2, c2 = data.load_features(im2)
            i1 = data.load_feature_index(im1, f1)
            i2 = data.load_feature_index(im2, f2)

            if im2 in im1_unthresholded_matches:
                unthresholded_matches = im1_unthresholded_matches[im2]
            else:
                unthresholded_matches = unthresholded_match_symmetric(f1, i1, f2, i2, data.config)

            if data.config['matcher_type'] == 'FLANN':
                # Flann returns squared L2 distances
                relevant_indices = np.where((unthresholded_matches[:,2] <= lowes_ratio**2) & (unthresholded_matches[:,3] <= lowes_ratio**2))[0]
            else:
                relevant_indices = np.where((unthresholded_matches[:,2] <= lowes_ratio) & (unthresholded_matches[:,3] <= lowes_ratio))[0]
            unthresholded_matches = unthresholded_matches[relevant_indices,:]

            logger.debug('{} - {} has {} candidate unthresholded matches'.format(im1, im2, len(unthresholded_matches)))

            if len(unthresholded_matches) < 8:
                im1_inliers[im2] = []
                continue

            sizes1 = p1[unthresholded_matches[:, 0].astype(int)][:, 2].copy()
            angles1 = p1[unthresholded_matches[:, 0].astype(int)][:, 3].copy()
            sizes2 = p2[unthresholded_matches[:, 1].astype(int)][:, 2].copy()
            angles2 = p2[unthresholded_matches[:, 1].astype(int)][:, 3].copy()

            unthresholded_rmatches, unthresholded_outliers, err1, err2, dist1, dist2 = \
                compute_gt_inliers(data, gt_poses, im1, im2, p1, p2, \
                    camera1, camera2, unthresholded_matches, lowes_ratio)

            im1_unthresholded_matches[im2] = unthresholded_matches
            im1_unthresholded_inliers[im2] = unthresholded_rmatches
            im1_unthresholded_outliers[im2] = unthresholded_outliers
            im1_unthresholded_features[im2] = np.stack([err1,err2,dist1,dist2,sizes1,angles1,sizes2,angles2], axis=1)

        data.save_unthresholded_matches(im1, im1_unthresholded_matches)
        data.save_unthresholded_inliers(im1, im1_unthresholded_inliers)
        data.save_unthresholded_outliers(im1, im1_unthresholded_outliers)
        data.save_unthresholded_features(im1, im1_unthresholded_features)

    del im1_unthresholded_matches
    del im1_unthresholded_inliers
    del im1_unthresholded_outliers
    del im1_unthresholded_features

def create_feature_matching_dataset(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = ctx.data.config['processes']

    if not data.reconstruction_exists('reconstruction_gt.json'):
        logger.info('Creating feature matching dataset - No ground-truth reconstruction exists!')
        return

    args = []
    for im in images:
        # if im != 'DSC_1761.JPG':
        #     continue
        element = [data, im, data.load_reconstruction('reconstruction_gt.json'), 1.0,\
            config.get('error_inlier_threshold'), config.get('error_outlier_threshold')]
        args.append(element)

    p = Pool(processes, maxtasksperchild=2)
    if processes == 1:    
        for arg in args:
            compute_matches_using_gt_reconstruction(arg)
    else:
        p.map(compute_matches_using_gt_reconstruction, args)
    p.close()

    data.save_feature_matching_dataset(lowes_threshold=0.8)
    # data.save_feature_matching_dataset(lowes_threshold=0.85)
    # data.save_feature_matching_dataset(lowes_threshold=0.9)
    # data.save_feature_matching_dataset(lowes_threshold=0.95)

def create_image_matching_dataset(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = ctx.data.config['processes']

    if not data.reconstruction_exists('reconstruction_gt.json'):
        logger.info('Creating image matching dataset - No ground-truth reconstruction exists!')
        return

    data.save_image_matching_dataset(robust_matches_threshold=15)
    data.save_image_matching_dataset(robust_matches_threshold=20)

def rmatches_adapter(data, options={}):
    im_all_rmatches = {}
    im_num_rmatches = {}
    im_num_rmatches_cost = {}
    im_num_rmatches_cost_with_sequences = {}

    for img in data.images():
        _, _, rmatches = data.load_all_matches(img)
        im_all_rmatches[img] = rmatches

    if 'apply_sequence_ranks' in options and options['apply_sequence_ranks'] is True:
        sequence_ranks = data.load_sequence_ranks()
        sequence_count = 0
        total_rmatches = 0
        sequences = {}
        adjacent_pairs = {}
        for i, im1 in enumerate(sorted(data.images())):
            for j,im2 in enumerate(sorted(data.images())):
                if j != i + 1:
                    continue
                adjacent_pairs[im1] = {im2: True}
                adjacent_pairs[im2] = {im1: True}
                if im1 in im_all_rmatches and im2 in im_all_rmatches[im1]:
                    rmatches = len(im_all_rmatches[im1][im2])
                elif im2 in im_all_rmatches and im1 in im_all_rmatches[im2]:
                    rmatches = len(im_all_rmatches[im2][im1])
                else:
                    rmatches = 0
                total_rmatches += rmatches
                sequence_count += 1
                # if sequence_ranks[im1][im2]['rank'] == 0:
                #     if im1 in im_all_rmatches and im2 in im_all_rmatches[im1]:
                #         rmatches = len(im_all_rmatches[im1][im2])
                #     elif im2 in im_all_rmatches and im1 in im_all_rmatches[im2]:
                #         rmatches = len(im_all_rmatches[im2][im1])
                #     else:
                #         rmatches = 0
                #     sequences['{}---{}'.format(im1,im2)] = True
                #     total_rmatches += rmatches
                #     sequence_count += 1
        c = 1.0 * total_rmatches / sequence_count
        logger.info('\tTotal Sequence pairs: {}  Average rmatches between consecutive images: {}'.format(sequence_count, c))
        logger.info(json.dumps(sequences, sort_keys=True, indent=4, separators=(',', ': ')))

    for im1 in im_all_rmatches:
        if im1 not in im_num_rmatches:
            im_num_rmatches[im1] = {}
            im_num_rmatches_cost[im1] = {}
        
        if 'apply_sequence_ranks' in options and options['apply_sequence_ranks'] is True:
            if im1 not in im_num_rmatches_cost_with_sequences:
                im_num_rmatches_cost_with_sequences[im1] = {}
            
        for im2 in im_all_rmatches[im1]:
            if im2 not in im_num_rmatches:
                im_num_rmatches[im2] = {}
                im_num_rmatches_cost[im2] = {}
            if 'apply_sequence_ranks' in options and options['apply_sequence_ranks'] is True:
                if im2 not in im_num_rmatches_cost_with_sequences:
                    im_num_rmatches_cost_with_sequences[im2] = {}

            im_num_rmatches[im1][im2] = len(im_all_rmatches[im1][im2])
            im_num_rmatches[im2][im1] = im_num_rmatches[im1][im2]
            if len(im_all_rmatches[im1][im2]) == 0:
                im_num_rmatches_cost[im1][im2] = 100000000.0
            else:
                im_num_rmatches_cost[im1][im2] = 1.0 / im_num_rmatches[im1][im2]

            im_num_rmatches_cost[im2][im1] = im_num_rmatches_cost[im1][im2]
            if 'apply_sequence_ranks' in options and options['apply_sequence_ranks'] is True:
                if im1 in adjacent_pairs and im2 in adjacent_pairs[im1] and adjacent_pairs[im1][im2] is True or \
                    im2 in adjacent_pairs and im1 in adjacent_pairs[im2] and adjacent_pairs[im2][im1] is True:
                    im_num_rmatches_cost_with_sequences[im1][im2] = min(im_num_rmatches_cost[im1][im2], 1.0/( options['sequence_cost_factor'] * c ) )
                    im_num_rmatches_cost_with_sequences[im2][im1] = im_num_rmatches_cost_with_sequences[im1][im2]
                else:
                    im_num_rmatches_cost_with_sequences[im1][im2] = im_num_rmatches_cost[im1][im2]
                    im_num_rmatches_cost_with_sequences[im2][im1] = im_num_rmatches_cost[im2][im1]

    return im_num_rmatches, im_num_rmatches_cost, im_num_rmatches_cost_with_sequences

def vocab_tree_adapter(data, options={}):
    vtranks, vtscores = data.load_vocab_ranks_and_scores()
    vt_rank_scores_mean = {}
    vt_rank_scores_min = {}
    vt_rank_scores_max = {}
    vt_scores_mean = {}
    vt_scores_min = {}
    vt_scores_max = {}
    
    # total_images = len(vt_ranks.keys())

    for im1 in vtranks:
        if im1 not in vt_rank_scores_mean:
            vt_rank_scores_mean[im1] = {}
            vt_rank_scores_min[im1] = {}
            vt_rank_scores_max[im1] = {}
            vt_scores_mean[im1] = {}
            vt_scores_min[im1] = {}
            vt_scores_max[im1] = {}

        for im2 in vtranks[im1]:
            vt_rank_scores_mean[im1][im2] = 0.5 * vtranks[im1][im2] + 0.5*vtranks[im2][im1]
            vt_rank_scores_min[im1][im2] = min(vtranks[im1][im2], vtranks[im2][im1])
            vt_rank_scores_max[im1][im2] = max(vtranks[im1][im2], vtranks[im2][im1])

            vt_scores_mean[im1][im2] = 0.5 * vtscores[im1][im2] + 0.5*vtscores[im2][im1]
            vt_scores_min[im1][im2] = min(vtscores[im1][im2], vtscores[im2][im1])
            vt_scores_max[im1][im2] = max(vtscores[im1][im2], vtscores[im2][im1])

    return vt_rank_scores_mean, vt_rank_scores_min, vt_rank_scores_max, vt_scores_mean, vt_scores_min, vt_scores_max

def triplet_errors_adapter(data, options={}):
    triplet_errors = data.load_triplet_pairwise_errors()
    triplet_scores_counts = {}
    triplet_scores_cumsum = {}

    for im1 in triplet_errors:
        if im1 not in triplet_scores_counts:
            triplet_scores_counts[im1] = {}
            triplet_scores_cumsum[im1] = {}
        for im2 in triplet_errors[im1]:
            # cum_error = \
            #     np.sum( \
            #         np.array(triplet_errors[im1][im2]['histogram-counts'][2:10]) * \
            #         np.power(0.5 + np.array(triplet_errors[im1][im2]['bins'][2:10]), 1) \
            #     ) + \
            #     np.sum( \
            #         np.array(triplet_errors[im1][im2]['histogram-counts'][10:]) * \
            #         np.power(0.5 + np.array(triplet_errors[im1][im2]['bins'][10:-1]), 2) \
            #     )

            # cum_error = \
            #     np.sum( \
            #         np.array(triplet_errors[im1][im2]['histogram-cumsum'][2:]) * \
            #         np.power(0.5 + np.array(triplet_errors[im1][im2]['bins'][2:-1]), 1) \
            #     )

            # cum_error = \
            #     np.sum( \
            #         np.array(triplet_errors[im1][im2]['histogram-counts'][2:]) * \
            #         np.power(2.0, 0.5 * np.array(triplet_errors[im1][im2]['bins'][2:-1])) \
            #     )

            if False:
                hist_counts = np.array(triplet_errors[im1][im2]['histogram-counts'][0:])
                hist_bins = np.array(triplet_errors[im1][im2]['bins'][0:-1])
                hist_cumsum = np.array(triplet_errors[im1][im2]['histogram-cumsum'][0:])
                cum_error = np.sum( hist_cumsum[0:] )


            te_histogram = np.array(triplet_errors[im1][im2]['histogram-cumsum'])
            mu, sigma = scipy.stats.norm.fit(te_histogram)
            # cum_error = np.sum( hist_counts * hist_bins )
            triplet_scores_cumsum[im1][im2] = mu

            te_histogram = np.array(triplet_errors[im1][im2]['histogram-counts'])
            mu, sigma = scipy.stats.norm.fit(te_histogram)
            # cum_error = np.sum( hist_counts * hist_bins )
            triplet_scores_counts[im1][im2] = mu

            # triplet_scores[im1][im2] = 1.0 / (cum_error + 1.0)

            # if im1 == 'DSC_1140.JPG':# and im2 == 'DSC_1159.JPG':
            # if im1 == 'DSC_1153.JPG' and im2 == 'DSC_1157.JPG':
            #     if im1 in options['scores_gt'] and im2 in options['scores_gt'][im1]:
            #         gts = options['scores_gt'][im1][im2]
            #     else:
            #         gts = 0.0
            #     print '{}-{} : {}  {}'.format(im1, im2, cum_error, gts)

            #     print np.array(triplet_errors[im1][im2]['histogram-cumsum'][0:])
            #     print np.array(triplet_errors[im1][im2]['histogram-counts'][0:])
            #     print np.array(triplet_errors[im1][im2]['bins'][0:])
            #     import sys;sys.exit(1)

    return triplet_scores_counts, triplet_scores_cumsum

def photometric_errors_adapter(data, options={}):
    photometric_errors = data.load_photometric_errors()
    photometric_scores_counts = {}
    photometric_scores_cumsum = {}

    for im1 in photometric_errors:
        if im1 not in photometric_scores_counts:
            photometric_scores_counts[im1] = {}
            photometric_scores_cumsum[im1] = {}
        for im2 in photometric_errors[im1]:
            pe_histogram = np.array(photometric_errors[im1][im2]['histogram-cumsum'])
            mu, sigma = scipy.stats.norm.fit(pe_histogram)
            photometric_scores_cumsum[im1][im2] = mu

            pe_histogram = np.array(photometric_errors[im1][im2]['histogram-counts'])
            mu, sigma = scipy.stats.norm.fit(pe_histogram)
            photometric_scores_counts[im1][im2] = mu

    return photometric_scores_counts, photometric_scores_cumsum

def groundtruth_image_matching_results_adapter(data):
    gt_results = data.load_groundtruth_image_matching_results()
    scores_gt = {}
    for im1 in gt_results:
        if im1 not in scores_gt:
            scores_gt[im1] = {}
        for im2 in gt_results[im1]:
            scores_gt[im1][im2] = gt_results[im1][im2]['score']
    return scores_gt

def calculate_lccs(ctx):
    data = ctx.data
    images = data.images()
    lccs = {}
    edge_threshold = 15

    num_rmatches, _, _ = rmatches_adapter(data)
    G = opensfm.commands.formulate_graphs.formulate_graph([data, images, num_rmatches, 'rm', edge_threshold])
    for threshold in [15, 20, 25, 30, 35, 40]:
        G_thresholded = opensfm.commands.formulate_graphs.threshold_graph_edges(G, threshold, key='weight')
        for i,im1 in enumerate(sorted(G_thresholded.nodes())):
            if im1 not in lccs:
                lccs[im1] = {}
            lccs[im1][threshold] = round(G_thresholded.node[im1]['lcc'], 3)

    # print lccs
    data.save_lccs(lccs)
    # return lccs
    # # cameras = data.load_camera_models()
    # exifs = ctx.exifs
    # config = data.config
    # processes = config['processes']
    # args = []
    # color_image = True
    # num_histogram_images = 4
    # histogram_size = 32 * num_histogram_images

    # im_all_rmatches = {}
    # im_num_rmatches = {}

    # for img in data.images():
    #     _, _, rmatches = data.load_all_matches(img)
    #     im_all_rmatches[img] = rmatches

    # for im1 in im_all_rmatches:
    #     if im1 not in im_num_rmatches:
    #         im_num_rmatches[im1] = {}
    #     for im2 in im_all_rmatches[im1]:
    #         im_num_rmatches[im1][im2] = len(im_all_rmatches[im1][im2])

    # return im_num_rmatches




def get_cmap(n, name='hsv'):
    colormaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r']
    return plt.get_cmap(colormaps[n])

def plot_mds(data, pos_gt, pos, iteration, opts):
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])

    s = 50
    cmap = get_cmap(opts['max_iterations'] + 1)
    if pos_gt is not None and iteration <= 0:
        for i in range(0,len(pos_gt)):
            if 'triangulated-indices' in opts and i in opts['triangulated-indices']:
                color = 'b'
            else:
                color = 'g'
            plt.scatter([pos_gt[i, 0]], [pos_gt[i, 1]], c=color, s=s, lw=0,
                        label='True Position')
        results_folder = os.path.join(data.data_path, 'results')
        mkdir_p(results_folder)
        plt.savefig(os.path.join(data.data_path, 'results', 'path-gt-lmds-{}-i-{}.png'.format(opts['lmds'], iteration)),dpi=90)
    
    if pos_gt is None and (iteration + 1) % opts['solve_mds_plot_freq'] == 0:
        for i in range(0,len(pos)):
            if 'triangulated-indices' in opts and i in opts['triangulated-indices']:
                color = 'b'
            else:
                color = 'r'
            plt.scatter(pos[i, 0], pos[i, 1], c=color, s=s, lw=0, label='MDS - Iteration {}'.format(iteration))
        #     index = (len(pos)-opts['triangulated-indices'])
        #     plt.scatter(pos[0:index, 0], pos[0:index, 1], c='r', s=s, lw=0, label='MDS - Iteration {}'.format(iteration))
        #     plt.scatter(pos[index:, 0], pos[index:, 1], c='b', s=s, lw=0, label='MDS - Iteration {}'.format(iteration))
        # else:
        #     plt.scatter(pos[:, 0], pos[:, 1], c='r', s=s, lw=0, label='MDS - Iteration {}'.format(iteration))
        results_folder = os.path.join(data.data_path, 'results')
        mkdir_p(results_folder)
        plt.savefig(os.path.join(results_folder, 'path-inferred-{}-lmds-{}.png'.format(opts['shortest_path_label'], opts['lmds'])),dpi=90)
        

    if False:
        similarities = similarities.max() / similarities * 100
        similarities[np.isinf(similarities)] = 0
        # Plot the edges
        start_idx, end_idx = np.where(pos)
        # a sequence of (*line0*, *line1*, *line2*), where::
        #            linen = (x0, y0), (x1, y1), ... (xm, ym)
        segments = [[pos_gt[i, :], pos_gt[j, :]]
                    for i in range(len(pos)) for j in range(len(pos))]
        values = np.abs(similarities)
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.Blues,
                            norm=plt.Normalize(0, values.max()))
        lc.set_array(similarities.flatten())
        lc.set_linewidths(0.5 * np.ones(len(segments)))
        ax.add_collection(lc)

    plt.clf()


def solve_mds(data, distances_gt, distances, num_fns, opts, debug):
    logger.info('#'*100)
    logger.info('#'*40 + ' Solving MDS problem... ' + '#'*40)
    logger.info('#'*100)
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)
    if distances_gt is not None:
        seed_gt = np.random.RandomState(seed=3)
        mds_gt = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=seed_gt,
                   dissimilarity="precomputed", n_jobs=1)
        mds_embedding_pos_gt = mds_gt.fit(distances_gt).embedding_
    
    mds_embedding_pos = mds.fit(distances).embedding_

    # Rotate the data
    clf = PCA(n_components=2)
    nmds = manifold.MDS(n_components=3, metric=False, max_iter=3000, eps=1e-9,
                    dissimilarity="precomputed", random_state=seed, n_jobs=1,
                    n_init=1)
    if distances_gt is not None:
        clf_gt = PCA(n_components=2)
        nmds_gt = manifold.MDS(n_components=3, metric=False, max_iter=3000, eps=1e-9,
                        dissimilarity="precomputed", random_state=seed_gt, n_jobs=1,
                        n_init=1)
        pos_gt = nmds_gt.fit_transform(distances_gt, init=mds_embedding_pos_gt)
        pos_gt = mds_embedding_pos_gt
        pos_gt = clf_gt.fit_transform(pos_gt)
        scale_gt = np.max(np.abs(pos_gt))
        pos_gt = pos_gt / scale_gt
    
    # pos = nmds.fit_transform(distances, init=mds_embedding_pos)
    pos = mds_embedding_pos
    pos = clf.fit_transform(pos)
    scale = np.max(np.abs(pos))
    inferred_positions = pos / scale

    if distances_gt is not None:
        plot_mds(data=data, pos_gt=pos_gt, pos=None, iteration=0, opts=opts)
    plot_mds(data=data, pos_gt=None, pos=inferred_positions, iteration=1, opts=opts)
    updated_distances = euclidean_distances(inferred_positions)

    if distances_gt is not None:
        return mds_embedding_pos, mds_embedding_pos_gt, inferred_positions, pos_gt, updated_distances
    return mds_embedding_pos, inferred_positions, updated_distances

def build_dummy_camera():
    camera = types.PerspectiveCamera()
    camera.id = "v2 sony a7sm2 1920 1080 perspective 0.5833"
    camera.width = 1920
    camera.height = 1080
    camera.focal_prior = 0.5833333333333334
    camera.focal = 0.6069623805001559
    camera.k1 = 0.0
    camera.k2 = 0.0
    camera.k1_prior = 0.0
    camera.k2_prior = 0.0
    return camera

def create_inferred_reconstruction(positions, reverse_image_mapping):
    recon_inferred = types.Reconstruction()
    camera = build_dummy_camera()
    recon_inferred.add_camera(camera)
    for i in range(0,len(positions)):
        pose = types.Pose()
        # pose.set_origin([positions[i][1], 0, positions[i][0]])
        pose.set_origin([positions[i][0], 0, positions[i][1]])
        
        shot = types.Shot()
        shot.camera = camera
        shot.pose = pose
        shot.id = reverse_image_mapping[i]

        sm = types.ShotMetadata()
        sm.orientation = 1
        sm.gps_position = [0.0, 0.0, 0.0]
        sm.gps_dop = 999999.0
        shot.metadata = sm

        # add shot to reconstruction
        recon_inferred.add_shot(shot)
    return recon_inferred

def distance_based_triangulation(data, landmark_positions, delta_a):
    delta_a_sq = np.multiply(delta_a, delta_a)
    L_inv = np.linalg.pinv(np.matrix(landmark_positions))

    delta_i = euclidean_distances(landmark_positions)
    delta_i_sq = np.multiply(delta_i, delta_i)


    # distances_squared = distances
    # landmark_distances_squared = euclidean_distances(landmark_positions)

    # landmark_distances_squared = np.dot(landmark_distances, landmark_distances)
    # print landmark_distances
    # print landmark_distances_squared
    # import sys; sys.exit(1)
    
    # u_landmark_distances_squared = np.sum(landmark_distances_squared, axis=1) / (len(landmark_distances_squared))
    
    delta_u = np.mean(delta_i_sq, axis=0)
    # delta_u = np.sum(delta_i_sq, axis=1) / len(delta_i)

    # print '#'*100
    # print u_landmark_distances_squared
    # print '-'*100
    # print np.mean(landmark_distances_squared, axis=1)
    # print '='*100
    # print landmark_positions_inv
    # print '$'*100
    # print landmark_positions.shape
    # print distances.shape

    # print '@'*100
    # print np.matrix(landmark_positions_inv).shape
    # print distances_squared.reshape((-1,1)).shape
    # print '@'*100
    # print np.matrix((distances_squared - u_landmark_distances_squared).reshape((-1,1))).shape

    # print '@'*100

    # x = -0.5 * np.matrix(landmark_positions_inv) * np.matrix((delta_a_sq - delta_u).reshape((-1,1)))
    x = -0.5 * L_inv * np.matrix((delta_a_sq - delta_u).reshape((-1,1)))
    # print '!'*100
    # print x.shape
    # import sys; sys.exit(1)
    # pass
    return x

def infer_cleaner_positions(ctx):
    logger.info('#'*100)
    logger.info('#'*25 + ' Inferring cleaner camera positions using shortest paths... ' + '#'*25)
    logger.info('#'*100)
    data = ctx.data
    images = sorted(data.images())
    np_images = np.array(images)
    num_fns = len(images)
    seq_cost_factor = ctx.sequence_cost_factor
    
    if data.reconstruction_exists('reconstruction_gt.json'):
        reverse_image_mapping_gt = {}
        recon_gt = data.load_reconstruction('reconstruction_gt.json')[0]
        num_fns_gt = len(recon_gt.shots.keys())
        positions_gt = np.zeros((num_fns_gt,3))
        for i,k in enumerate(sorted(recon_gt.shots.keys())):
            reverse_image_mapping_gt[i] = k
            positions_gt[i,:] = recon_gt.shots[k].pose.get_origin()

        distances_gt = euclidean_distances(positions_gt)

    for sp_label in ['rm-cost']:#, 'rm-seq-cost-{}'.format(seq_cost_factor)]:
        distances_baseline = -1 * np.ones((len(images), len(images)))
        shortest_paths = data.load_shortest_paths(label=sp_label)
        cached_features = {}
        image_mapping = {}
        reverse_image_mapping = {}
        im_closest_images = {}

        for i,im1 in enumerate(images):
            image_mapping[im1] = i
            reverse_image_mapping[i] = im1

            im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
            p1, f1, c1 = data.load_features(im1)
            for j,im2 in enumerate(images):
                if i == j:
                    distances_baseline[i,j] = 0.0
                else:
                    if im1 in shortest_paths and im2 in shortest_paths[im1]:
                        path = shortest_paths[im1][im2]['path']
                    else:
                        path = shortest_paths[im2][im1]['path']
                    shortest_path_distance = 0.0
                    for k,p in enumerate(path):
                        im1_, im2_ = p.split('---')
                        if im1_ in cached_features:
                            p1_ = cached_features[im1_]
                        else:
                            p1_, _, _ = data.load_features(im1_)
                            cached_features[im1_] = p1_
                        if im2_ in cached_features:
                            p2_ = cached_features[im2_]
                        else:
                            p2_, _, _ = data.load_features(im2_)
                            cached_features[im2_] = p2_

                        # if path[p]['rmatches'] == 0:
                        #     print (path)
                        #     print ('{} : {}    /    {} : {}'.format(im1, len(p1_), im2, len(p2_)))
                        
                        # shortest_path_distance += 1.0 / (path[p]['rmatches'] / min(len(p1_), len(p2_)))
                        shortest_path_distance += path[p]['cost']
                    distances_baseline[i,j] = shortest_path_distance
                distances_baseline[j,i] = distances_baseline[i,j]

        
        # distances_baseline = np.array([
        #     [0.0,       1.0,      1.0,        2.0],
        #     [1.0,       0.0,      1.4142,     2.236],
        #     [1.0,       1.4142,   0.0,        1.0],
        #     [2.0,       2.236,    1.0,        0.0]
        #     ])
        if False:
            positions = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                ])

            images = images[0:len(positions)]
            num_fns = len(images)
            distances_baseline = euclidean_distances(positions)
            distances_gt = distances_baseline[:,:]

        distances_baseline_pruned = distances_baseline[:,:]
        if data.reconstruction_exists('reconstruction_gt.json'):
            distances_gt_pruned = distances_gt[:,:]

        # noisiest_images = []
        mds_images = images[:]
        logger.info('Total images: {}'.format(len(mds_images)))
        # for i in range(0, int(math.floor(0.5*len(images)))): # remove maximum 50% of images
        # for i in range(0, int(math.floor(0.2*len(images)))): # remove maximum 20% of images
        # for i in range(0, int(math.floor(0.1*len(images)))): # remove maximum 20% of images
        for i in range(0, 1): # remove maximum 50% of images
            # r, c = np.where(distances_baseline_pruned > 100.0/1.0)
            r, c = np.where(distances_baseline_pruned > 1.0/15.0)
            # r, c = np.where(distances_baseline_pruned > 1.0/100.0)
            if len(r) == 0:
                logger.info('\tRemoved all "noisy" images. Images left to perform MDS: {}'.format(distances_baseline_pruned.shape[0]))
                break
            unique_images, image_counts = np.unique(r, return_counts=True)
            mapped_images = dict(zip(unique_images, image_counts))
            noisiest_image = unique_images[np.argmax(image_counts)]
            distances_baseline_pruned = np.delete(distances_baseline_pruned,(noisiest_image), axis=0)
            distances_baseline_pruned = np.delete(distances_baseline_pruned,(noisiest_image), axis=1)
            if data.reconstruction_exists('reconstruction_gt.json'):
                distances_gt_pruned = np.delete(distances_gt_pruned,(noisiest_image), axis=0)
                distances_gt_pruned = np.delete(distances_gt_pruned,(noisiest_image), axis=1)
            # noisiest_images.append(noisiest_image)
            del mds_images[noisiest_image]


        # Since we've pruned greedily, we need to figure out what are landmark indices and pruned indices (compared to the original matrix)
        images_pruned = []
        landmark_indices = []
        triangulated_indices = []
        offset = 0
        for i in range(0,len(images)):
            if i - offset >= len(mds_images) or images[i] != mds_images[i - offset]:
                images_pruned.append({'image': images[i], 'index': i})
                triangulated_indices.append(i)
                offset += 1
            else:
                landmark_indices.append(i)
        landmark_indices = np.array(landmark_indices)
        
        options = {'solve_mds_plot_freq': 1, 'max_iterations': 20, 'shortest_path_label': sp_label, 'lmds': True}
        if data.reconstruction_exists('reconstruction_gt.json'):
            # distances_gt_pruned_sq = np.multiply(distances_gt_pruned, distances_gt_pruned)
            # distances_baseline_pruned_sq = np.multiply(distances_baseline_pruned, distances_baseline_pruned)
            mds_embedding_positions, mds_embedding_positions_gt, positions_inferred, positions_inferred_gt, _ = solve_mds(data=data, distances_gt=distances_gt_pruned, distances=distances_baseline_pruned, num_fns=len(mds_images), opts=options, debug=False)
        else:
            mds_embedding_positions, positions_inferred, _ = solve_mds(data=data, distances_gt=None, distances=distances_baseline_pruned, num_fns=len(mds_images), opts=options, debug=False)

        # print (mds_embedding_positions)
        # import sys; sys.exit(1)
        use_embedding_positions = True
        if use_embedding_positions:
            all_positions_inferred = mds_embedding_positions[:,:]
            if data.reconstruction_exists('reconstruction_gt.json'):
                all_positions_inferred_gt = mds_embedding_positions_gt[:,:]
        else:
            all_positions_inferred = positions_inferred[:,:]
            if data.reconstruction_exists('reconstruction_gt.json'):
                all_positions_inferred_gt = positions_inferred_gt[:,:]


        logger.info('Triangulating pruned cameras using distances from landmark cameras...')
        for ip in images_pruned:
            logger.info('\tImage being added: {}'.format(reverse_image_mapping[ip['index']]))
            confident_indices = distances_baseline[ip['index'], landmark_indices].argsort()[:max(3, int(len(images)*0.2))]
            # confident_indices = distances_baseline[ip['index'], landmark_indices].argsort()[:3]
            x = distance_based_triangulation(data, mds_embedding_positions[confident_indices], distances_baseline[ip['index'], landmark_indices][confident_indices])
            # x = distance_based_triangulation(data, mds_embedding_positions, distances_baseline[ip['index'], landmark_indices])
            
            # all_positions_inferred = np.append(all_positions_inferred, x.reshape((1,-1)), axis=0)
            all_positions_inferred = np.insert(all_positions_inferred,ip['index'],x.reshape((1,-1)), axis=0) 

            if data.reconstruction_exists('reconstruction_gt.json'):
                x_gt = distance_based_triangulation(data, mds_embedding_positions_gt, distances_gt[ip['index'], landmark_indices])
                # all_positions_inferred_gt = np.append(all_positions_inferred_gt, x_gt.reshape((1,-1)), axis=0)
                all_positions_inferred_gt = np.insert(all_positions_inferred_gt,ip['index'],x_gt.reshape((1,-1)), axis=0) 


        options['triangulated-indices'] = triangulated_indices

        if False and use_embedding_positions:
            seed = np.random.RandomState(seed=3)
            nmds = manifold.MDS(n_components=3, metric=False, max_iter=3000, eps=1e-9,
                        dissimilarity="precomputed", random_state=seed, n_jobs=1,
                        n_init=1)
            clf = PCA(n_components=2)
            all_positions_inferred = nmds.fit_transform(distances_baseline, init=all_positions_inferred)
            all_positions_inferred = clf.fit_transform(all_positions_inferred)
            scale = np.max(np.abs(all_positions_inferred))
            all_positions_inferred = all_positions_inferred / scale
            if data.reconstruction_exists('reconstruction_gt.json'):
                seed_gt = np.random.RandomState(seed=3)
                nmds_gt = manifold.MDS(n_components=3, metric=False, max_iter=3000, eps=1e-9,
                            dissimilarity="precomputed", random_state=seed_gt, n_jobs=1,
                            n_init=1)
                clf_gt = PCA(n_components=2)
                all_positions_inferred_gt = nmds_gt.fit_transform(distances_gt, init=all_positions_inferred_gt)
                all_positions_inferred_gt = clf_gt.fit_transform(all_positions_inferred_gt)
                scale_gt = np.max(np.abs(all_positions_inferred_gt))
                all_positions_inferred_gt = all_positions_inferred_gt / scale_gt
        else:
            all_positions_inferred = np.array(all_positions_inferred)
            if data.reconstruction_exists('reconstruction_gt.json'):
                all_positions_inferred_gt = np.array(all_positions_inferred_gt)

        # print (all_positions_inferred)
        updated_distances = euclidean_distances(all_positions_inferred)
        plot_mds(data=data, pos_gt=None, pos=all_positions_inferred, iteration=1, opts=options)
        if data.reconstruction_exists('reconstruction_gt.json'):
            plot_mds(data=data, pos_gt=all_positions_inferred_gt, pos=None, iteration=-1, opts=options)

        for i in range(0,num_fns):
            order = np.argsort(np.array(updated_distances[:,i]))
            im_closest_images[reverse_image_mapping[i]] = np_images[order].tolist()

        data.save_closest_images(im_closest_images, label='{}-lmds-{}'.format(sp_label, options['lmds']))

    # import sys; sys.exit(1)
    if data.reconstruction_exists('reconstruction_gt.json'):
        im_closest_images_gt = {}
        np_images_gt = np.array(sorted(recon_gt.shots.keys()))
        for i in range(0,num_fns_gt):
            order_gt = np.argsort(np.array(distances_gt[:,i]))
            im_closest_images_gt[reverse_image_mapping_gt[i]] = np_images_gt[order_gt].tolist()
        data.save_closest_images(im_closest_images_gt, label='gt-lmds-{}'.format(options['lmds']))

        # recon_inferred = create_inferred_reconstruction(positions_inferred, reverse_image_mapping)
        # relevant_reconstructions = [[recon_gt, recon_inferred, 'path']]
        # opensfm.commands.validate_results.ransac_based_ate_evaluation(data, relevant_reconstructions)

def infer_positions(ctx):
    logger.info('#'*100)
    logger.info('#'*25 + ' Inferring camera positions using shortest paths... ' + '#'*25)
    logger.info('#'*100)
    data = ctx.data
    images = sorted(data.images())
    np_images = np.array(images)
    num_fns = len(images)
    seq_cost_factor = ctx.sequence_cost_factor
    
    if data.reconstruction_exists('reconstruction_gt.json'):
        reverse_image_mapping_gt = {}
        recon_gt = data.load_reconstruction('reconstruction_gt.json')[0]
        num_fns_gt = len(recon_gt.shots.keys())
        positions_gt = np.zeros((num_fns_gt,3))
        for i,k in enumerate(sorted(recon_gt.shots.keys())):
            reverse_image_mapping_gt[i] = k
            positions_gt[i,:] = recon_gt.shots[k].pose.get_origin()

        distances_gt = euclidean_distances(positions_gt)

    for sp_label in ['rm-cost', 'rm-seq-cost-{}'.format(seq_cost_factor)]:
        distances_baseline = -1 * np.ones((len(images), len(images)))
        shortest_paths = data.load_shortest_paths(label=sp_label)
        cached_features = {}
        image_mapping = {}
        reverse_image_mapping = {}
        im_closest_images = {}

        for i,im1 in enumerate(images):
            image_mapping[im1] = i
            reverse_image_mapping[i] = im1

            im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
            p1, f1, c1 = data.load_features(im1)
            for j,im2 in enumerate(images):
                if i == j:
                    distances_baseline[i,j] = 0.0
                else:
                    if im1 in shortest_paths and im2 in shortest_paths[im1]:
                        path = shortest_paths[im1][im2]['path']
                    else:
                        path = shortest_paths[im2][im1]['path']
                    shortest_path_distance = 0.0
                    for k,p in enumerate(path):
                        im1_, im2_ = p.split('---')
                        if im1_ in cached_features:
                            p1_ = cached_features[im1_]
                        else:
                            p1_, _, _ = data.load_features(im1_)
                            cached_features[im1_] = p1_
                        if im2_ in cached_features:
                            p2_ = cached_features[im2_]
                        else:
                            p2_, _, _ = data.load_features(im2_)
                            cached_features[im2_] = p2_

                        # if path[p]['rmatches'] == 0:
                        #     print (path)
                        #     print ('{} : {}    /    {} : {}'.format(im1, len(p1_), im2, len(p2_)))
                        
                        # shortest_path_distance += 1.0 / (path[p]['rmatches'] / min(len(p1_), len(p2_)))
                        shortest_path_distance += path[p]['cost']
                    distances_baseline[i,j] = shortest_path_distance
                distances_baseline[j,i] = distances_baseline[i,j]

        options = {'solve_mds_plot_freq': 1, 'max_iterations': 20, 'shortest_path_label': sp_label, 'lmds': False}
        if data.reconstruction_exists('reconstruction_gt.json'):
            mds_embedding_positions, mds_embedding_positions_gt, positions_inferred, positions_inferred_gt, updated_distances = solve_mds(data=data, distances_gt=distances_gt, distances=distances_baseline, num_fns=num_fns, opts=options, debug=False)
        else:
            mds_embedding_positions, positions_inferred, updated_distances = solve_mds(data=data, distances_gt=None, distances=distances_baseline, num_fns=num_fns, opts=options, debug=False)

        for i in range(0,num_fns):
            order = np.argsort(np.array(updated_distances[:,i]))
            im_closest_images[reverse_image_mapping[i]] = np_images[order].tolist()

        data.save_closest_images(im_closest_images, label='{}-lmds-{}'.format(sp_label, options['lmds']))

    if data.reconstruction_exists('reconstruction_gt.json'):
        im_closest_images_gt = {}
        np_images_gt = np.array(sorted(recon_gt.shots.keys()))
        for i in range(0,num_fns_gt):
            order_gt = np.argsort(np.array(distances_gt[:,i]))
            im_closest_images_gt[reverse_image_mapping_gt[i]] = np_images_gt[order_gt].tolist()
        data.save_closest_images(im_closest_images_gt, label='gt-lmds-{}'.format(options['lmds']))

        # recon_inferred = create_inferred_reconstruction(positions_inferred, reverse_image_mapping)
        # relevant_reconstructions = [[recon_gt, recon_inferred, 'path']]
        # opensfm.commands.validate_results.ransac_based_ate_evaluation(data, relevant_reconstructions)

def calculate_track_map(image_coordinates, track_lengths, grid_size):
    entropy = 0.0
    tmap = np.zeros((grid_size*grid_size,)).astype(np.float)
    if len(image_coordinates) == 0:
        return tmap.reshape((grid_size, grid_size))
    # dmap_entropy = np.zeros((grid_size*grid_size,)).astype(np.float)
    denormalized_image_coordinates = features.denormalized_image_coordinates(image_coordinates, grid_size, grid_size)
    indx = denormalized_image_coordinates[:,0].astype(np.int32)
    indy = denormalized_image_coordinates[:,1].astype(np.int32)
    for i in xrange(0,len(indx)):
        tmap[indy[i]*grid_size + indx[i]] += track_lengths[i]
        # dmap_entropy[indy[i]*grid_size + indx[i]] += 1.0
    return tmap.reshape((grid_size, grid_size))

def create_tracks_map(ctx):
    logger.info('#'*100)
    logger.info('#'*25 + ' Creating track map... ' + '#'*25)
    logger.info('#'*100)
    data = ctx.data
    graph = data.load_tracks_graph('tracks.csv')
    for im in data.images():
        image_coordinates = []
        track_lengths = []
        if im in graph:
            tracks = graph[im]
            for t in tracks:
                # tid = graph[im][t]['feature_id']
                x,y = graph[im][t]['feature']
                image_coordinates.append([x,y])
                track_lengths.append(len(graph[t]))
        track_map = calculate_track_map(np.array(image_coordinates), track_lengths, 224)
        data.save_track_map(im, track_map)
        # print np.sum(np.array(track_lengths))
        # print np.sum(track_map)
        # # print '{} : {}  -  {}'.format(im, t, graph[t])
        # return
