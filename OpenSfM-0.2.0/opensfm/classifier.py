import cv2
import glob
import json
import logging
import math
import numpy as np
import os
import pickle
import pyopengv
import random
import re

from opensfm import features, multiview
from opensfm import context
from multiprocessing import Pool
from sklearn.externals import joblib
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

    dmap = np.zeros((grid_size*grid_size,1)).astype(np.float)
    indx = ((image_coordinates[:,0] + 1)/2 * grid_size).astype(np.int32)
    indy = ((image_coordinates[:,1] + 1)/2 * grid_size).astype(np.int32)
    for i in xrange(0,len(indx)):
        dmap[indy[i]*grid_size + indx[i]] += 1.0

    prob_map = dmap / np.sum(dmap) + epsilon
    entropy = np.sum(prob_map * np.log2(prob_map))

    return round(-entropy/np.log2(np.sum(dmap)),4)

def next_best_view_score(image_coordinates):
  # Based on the paper Structure-from-Motion Revisited - https://demuc.de/papers/schoenberger2016sfm.pdf
  # Get a score based on number of common tracks and spatial distribution of the tracks in the image
  grid_sizes = [2, 4, 8, 16, 32, 64]
  score = 0

  for grid_size in grid_sizes:
    dmap = np.zeros((grid_size*grid_size,1))
    indx = ((image_coordinates[:,0] + 1)/2 * grid_size).astype(np.int32)
    indy = ((image_coordinates[:,1] + 1)/2 * grid_size).astype(np.int32)
    dmap[indy*grid_size + indx] = 1
    score += np.sum(dmap) * grid_size
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
        nbvs_im1, nbvs_im2, te_histogram, ch_im1, ch_im2, vt_rank_percentage_im1_im2, vt_rank_percentage_im2_im1, labels, \
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
        ),
    axis=1)
    
    y = labels

    # Fit regression model
    if regr is None:
        regr = GradientBoostingClassifier(max_depth=options['max_depth'], n_estimators=options['n_estimators'], subsample=1.0, random_state=rng)
        regr.fit(X, y)

    # Predict
    y_ = regr.predict_proba(X)[:,1]
    return fns, num_rmatches, regr, y_, labels

def relative_pose(arg):
    im1, im2, p1, p2, cameras, exifs, rmatches, threshold = arg

    p1_ = p1[rmatches[:, 0].astype(int)]
    p2_ = p2[rmatches[:, 1].astype(int)]
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

        for im2 in im1_all_matches:
            rmatches = im1_all_matches[im2]
            
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

            entropy_im1_8 = calculate_spatial_entropy(p1_, 8)
            entropy_im2_8 = calculate_spatial_entropy(p2_, 8)
            entropy_im1_16 = calculate_spatial_entropy(p1_, 16)
            entropy_im2_16 = calculate_spatial_entropy(p2_, 16)
            if im1 not in entropies:
                entropies[im1] = {}
            entropies[im1][im2] = {
                'entropy_im1_8': entropy_im1_8, 'entropy_im2_8': entropy_im2_8, \
                'entropy_im1_16': entropy_im1_16, 'entropy_im2_16': entropy_im2_16 \
                }
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

def warp_image(Ms, triangle_pts_img1, triangle_pts_img2, img1, img2, im1, im2, patchdataset, flags, colors):
  img2_o_final = 255 * np.ones(img1.shape, dtype = img1.dtype)
  
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
    img2_cropped = cv2.warpAffine( img1_cropped, M, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT_101 )

    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(img2_tri_cropped), (1.0, 1.0, 1.0), 16, 0);
    img2_cropped = img2_cropped * mask

    # Output image is set to white
    img2_o = 255 * np.ones(img1.shape, dtype = img1.dtype)
    img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_cropped
    img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_o_final[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_cropped

    if flags['draw_points']:
      for i, _ in enumerate(pts_img1[0]):
        color_index = random.randint(0,len(colors)-1)
        cv2.circle(img1, (int(pts_img1[0][i][0]), int(pts_img1[0][i][1])), 2, colors[color_index], 2)
        cv2.circle(img2, (int(pts_img2[0][i][0]), int(pts_img2[0][i][1])), 2, colors[color_index], 2)

    if flags['draw_triangles']:
      color_index = random.randint(0,len(colors)-1)
      cv2.polylines(img1,[triangle_pts_img1[s]],True,colors[color_index])
      cv2.polylines(img2,[triangle_pts_img2[s]],True,colors[color_index])

  h1,w1,c1 = img1_cropped.shape
  h2,w2,c2 = img2_cropped.shape
  img1_cropped = cv2.resize(img1_cropped, (max(w1,w2), max(h1,h2)), interpolation = cv2.INTER_CUBIC)
  img2_cropped = cv2.resize(img2_cropped, (max(w1,w2), max(h1,h2)), interpolation = cv2.INTER_CUBIC)
  img2_or = cv2.resize(img2_o_final, (max(w1,w2), max(h1,h2)), interpolation = cv2.INTER_CUBIC)

  imgs = np.concatenate((img1_cropped,img2_cropped, img2_or),axis=1)
  cv2.imwrite(os.path.join(patchdataset,'photometric-data/photometric-warped-image-{}-{}.png'.format(os.path.basename(im1), os.path.basename(im2))), img2_o_final)

def calculate_photometric_error_histogram(errors, im1, im2, patchdataset, debug):
  histogram, bins = np.histogram(np.array(errors), bins=51, range=(0,250))
  if debug:
    width = 0.9 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, histogram, align='center', width=width)
    plt.xlabel('L2 Error (LAB)', fontsize=16)
    plt.ylabel('Error count', fontsize=16)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(patchdataset,'photometric-data/photometric-histogram-{}-{}.png'.format(os.path.basename(im1), os.path.basename(im2))))
  return histogram

def calculate_convex_hull(img1_o, denormalized_p1_points, img2_o, denormalized_p2_points, debug):
  try:
    hull_img1 = ConvexHull( [ (x,y) for x,y in denormalized_p1_points[:,0:2].tolist()] )
    hull_img2 = ConvexHull( [ (x,y) for x,y in denormalized_p2_points[:,0:2].tolist()] )

    if debug:
      for v, vertex in enumerate(hull_img1.vertices):
        if v == 0:
          continue
        cv2.line(img1_o, ( int(denormalized_p1_points[hull_img1.vertices[v-1], 0]), int(denormalized_p1_points[hull_img1.vertices[v-1], 1]) ), \
          (  int(denormalized_p1_points[hull_img1.vertices[v], 0]), int(denormalized_p1_points[hull_img1.vertices[v], 1]) ), (255,0,0), 3)

      cv2.line(img1_o, ( int(denormalized_p1_points[hull_img1.vertices[v], 0]), int(denormalized_p1_points[hull_img1.vertices[v], 1]) ), \
          (  int(denormalized_p1_points[hull_img1.vertices[0], 0]), int(denormalized_p1_points[hull_img1.vertices[0], 1]) ), (255,0,0), 3)

      for v, vertex in enumerate(hull_img2.vertices):
        if v == 0:
          continue
        cv2.line(img2_o, ( int(denormalized_p2_points[hull_img2.vertices[v-1], 0]), int(denormalized_p2_points[hull_img2.vertices[v-1], 1]) ), \
          (  int(denormalized_p2_points[hull_img2.vertices[v], 0]), int(denormalized_p2_points[hull_img2.vertices[v], 1]) ), (255,0,0), 5)

      cv2.line(img2_o, ( int(denormalized_p2_points[hull_img2.vertices[v], 0]), int(denormalized_p2_points[hull_img2.vertices[v], 1]) ), \
          (  int(denormalized_p2_points[hull_img2.vertices[0], 0]), int(denormalized_p2_points[hull_img2.vertices[0], 1]) ), (255,0,0), 5)
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

def PolyArea2D(pts):
  lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
  area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
  return area

def tesselate_matches(im1, im2, matches, p1, p2, patchdataset, flags, ii, jj):
  n, outlier_threshold, debug, error_threshold, num_clusters, sample_matches = [flags['num_samples'], flags['outlier_threshold_percentage'], \
    flags['debug'], flags['lab_error_threshold'], flags['kmeans_num_clusters'], flags['use_kmeans']]

  # logger.info('Processing files: {} / {}'.format(im1, im2))
  t_total_get_triangle_points = 0.0
  t_total_photometric_error = 0.0
  total_pts = 0
  Ms = []
  all_errors = []
  triangle_pts_img1 = []
  triangle_pts_img2 = []
  colors = [(int(random.random()*255), int(random.random()*255), int(random.random()*255)) for i in xrange(0,100)]
  t_start_img_loading = timer()
  img1 = cv2.imread(im1,cv2.IMREAD_COLOR)
  img2 = cv2.imread(im2,cv2.IMREAD_COLOR)
  height_o, width_o, channels_o = img1.shape
 
  scale_factor = int(np.floor(width_o / 500.0))

  img1_o = cv2.resize(img1, (width_o/scale_factor, height_o/scale_factor), interpolation = cv2.INTER_CUBIC)
  img2_o = cv2.resize(img2, (width_o/scale_factor, height_o/scale_factor), interpolation = cv2.INTER_CUBIC)
  if debug:
    img1_w = cv2.resize(img1, (width_o/scale_factor, height_o/scale_factor), interpolation = cv2.INTER_CUBIC)
    img2_w = cv2.resize(img2, (width_o/scale_factor, height_o/scale_factor), interpolation = cv2.INTER_CUBIC)
  img1_original = cv2.cvtColor(img1_o, cv2.COLOR_BGR2LAB)
  img2_original = cv2.cvtColor(img2_o, cv2.COLOR_BGR2LAB)

  height, width, channels = img1_o.shape
  if debug:
    print '\t Images loading/reading time: {}'.format(timer()-t_start_img_loading)
    print '\t Matches shape: {}  /  {}'.format(matches[:,0].shape, matches[:,1].shape)
    print '\t p shape: {}  /  {}'.format(p1.shape, p2.shape)
  p1_points = p1[ matches[:,0].astype(np.int) ]
  p2_points = p2[ matches[:,1].astype(np.int) ]

  denormalized_p1_points = features.denormalized_image_coordinates(p1_points, width, height)
  denormalized_p2_points = features.denormalized_image_coordinates(p2_points, width, height)

  hull_img1, hull_img2 = calculate_convex_hull(img1_o, denormalized_p1_points, img2_o, denormalized_p2_points, debug)
  if hull_img1 is None or hull_img2 is None:
    return None, None, 0, np.array([])

  if sample_matches:
    indices = cluster_matches(denormalized_p1_points, k=num_clusters)
    tesselation_vertices_im1 = denormalized_p1_points[indices,0:2]
    tesselation_vertices_im2 = denormalized_p2_points[indices,0:2]
  else:
    tesselation_vertices_im1 = denormalized_p1_points[:,0:2]
    tesselation_vertices_im2 = denormalized_p2_points[:,0:2]

  if debug and flags['draw_matches']:
    for i, cc in enumerate(tesselation_vertices_im1):
      color_index = random.randint(0,len(colors)-1)
      cv2.circle(img1_o, (int(cc[0]), int(cc[1])), 2, colors[color_index], 2)
      cv2.circle(img2_o, (int(tesselation_vertices_im2[i, 0]), int(tesselation_vertices_im2[i, 1])), 2, colors[color_index], 2)

  try:
    # if len(tesselation_vertices_im1) > 500:
    #     logger.info('\tTesselation points: {}'.format(len(tesselation_vertices_im1)))
    #     print tesselation_vertices_im1
    triangles_img1 = Delaunay(tesselation_vertices_im1, qhull_options='Pp Qt')
  except:
    return None, None, 0, np.array([])

  if flags['sampling_method'] == 'sample_polygon_uniformly':
    t_start_sampling = timer()
    polygon, sampled_points_polygon_img1 = sample_points_polygon(denormalized_p1_points, hull_img1.vertices, n)
    if debug:
      print '\t Sampling time: {}'.format(timer()-t_start_sampling)

  h, w, c = img2_original.shape
  x = np.linspace(0, w - 1, w).astype(int)
  y = np.linspace(0, h - 1, h).astype(int)

  z_img1 = [None] * 3
  z_img2 = [None] * 3
  f_img1 = [None] * 3
  f_img2 = [None] * 3
  for i in xrange(0,3):
    z_img1[i] = img1_original[0:h,0:w,i]
    z_img2[i] = img2_original[0:h,0:w,i]
    f_img1[i] = interpolate.interp2d(x, y, z_img1[i], kind='cubic')
    f_img2[i] = interpolate.interp2d(x, y, z_img2[i], kind='cubic')

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
    
    if flags['sampling_method'] == 'sample_triangles_uniformly':
      sampled_points_img1 = sample_points_triangle(pts_img1_, n)
    elif flags['sampling_method'] == 'sample_polygon_uniformly':
      t_start_get_triangle_points = timer()
      sampled_points_img1 = get_triangle_points(pts_img1, sampled_points_polygon_img1)
      t_total_get_triangle_points += timer()-t_start_get_triangle_points
    total_pts += len(sampled_points_img1)
    if len(sampled_points_img1) == 0:
      continue


    sampled_points_transformed = (np.matrix(M) * np.matrix(np.concatenate((sampled_points_img1, np.ones((len(sampled_points_img1),1))), axis=1)).T).T
    t_start_photometric_error = timer()
    errors = get_photometric_error(f_img1, f_img2, sampled_points_img1.tolist(), img1_original, sampled_points_transformed.tolist(), img2_original, error_threshold=error_threshold)
    t_total_photometric_error += timer()-t_start_photometric_error
    all_errors.extend(errors)

    if debug and (flags['draw_outliers'] or flags['draw_points']):
      for i, sampled_pt_img1 in enumerate(sampled_points_img1):
        color_index_ = random.randint(0,len(colors)-1)
        if flags['draw_points']:
          cv2.circle(img1_o, ( int( sampled_pt_img1[0] ), int( sampled_pt_img1[1] ) ), 1, colors[color_index_], 2)
          cv2.circle(img2_o, ( int( sampled_points_transformed[i, 0] ), int( sampled_points_transformed[i, 1] ) ), 1, colors[color_index_], 2)
        elif flags['draw_outliers']:
          if not outliers[i]:
            continue
          cv2.circle(img1_o, ( int( sampled_pt_img1[0] ), int( sampled_pt_img1[1] ) ), 1, colors[color_index_], 2)
          cv2.circle(img2_o, ( int( sampled_points_transformed[i, 0] ), int( sampled_points_transformed[i, 1] ) ), 1, colors[color_index_], 2)

        
    if debug and flags['draw_triangles']:
      cv2.polylines(img1_o,[pts_img1],True,colors[color_index])
      cv2.polylines(img2_o,[pts_img2],True,colors[color_index])

  polygon_points = [( int(denormalized_p1_points[hull_img1.vertices[v], 0]), int(denormalized_p1_points[hull_img1.vertices[v], 1]) ) for v, vertex in enumerate(hull_img1.vertices)]
  polygon_area = PolyArea2D(polygon_points)
  polygon_area_percentage = 100.0 * polygon_area/(h * w)
  if debug:
    print '\t Getting triangle points time: {}'.format(t_total_get_triangle_points)
    print '\t Photometric error time: {}'.format(t_total_photometric_error)
    print '\t Main triangle loop time: {}'.format(timer()-t_start_triangle_loop)
    print '\t Total points: {}  Total samples: {}  Polygon area: {}  Polygon %: {}'\
      .format(total_pts, flags['num_samples'], polygon_area, polygon_area_percentage)
    print '\n'

  if debug:
    imgs = np.concatenate((img1_o,img2_o),axis=1)
    cv2.imwrite(os.path.join(patchdataset,'photometric-data/photometric-delaunay-{}-{}.png'.format(os.path.basename(im1), os.path.basename(im2))), imgs)

  if debug:
    warp_image(Ms, triangle_pts_img1, triangle_pts_img2, img1_w, img2_w, im1, im2, patchdataset, flags, colors)
  histogram = calculate_photometric_error_histogram(all_errors, im1, im2, patchdataset, debug)
  # logger.info('Finished processing files: {}({}) / {}({})'.format(im1, ii, im2, jj))
  return polygon_area, polygon_area_percentage, len(triangles_img1.simplices), histogram

def calculate_photometric_error_convex_hull(arg):
  ii, jj, patchdataset, data, im1, im2, matches, flags = arg
  
  if flags['masked_tags']:
    p1, f1, c1 = data.load_features_masked(im1)
    p2, f2, c2 = data.load_features_masked(im2)
  else:
    p1, f1, c1 = data.load_features(im1)
    p2, f2, c2 = data.load_features(im2)
  
  polygon_area, polygon_area_percentage, total_triangles, histogram = tesselate_matches(os.path.join(data.data_path,'images',im1), \
      os.path.join(data.data_path,'images',im2), matches, p1, p2, patchdataset, flags, ii, jj)
  return im1, im2, polygon_area, polygon_area_percentage, total_triangles, histogram.tolist()

def calculate_photometric_errors(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = config['processes']
    args = []
    flags = {'masked_tags': False, 'num_samples': 500, 'outlier_threshold_percentage': 0.7, 'debug': False, 'lab_error_threshold': 50, \
        'use_kmeans': False, 'sampling_method': 'sample_polygon_uniformly', 'draw_matches': False, 'draw_triangles': False, 'draw_points': False, \
        'processes': processes, 'draw_outliers': False, 'kmeans_num_clusters': None }

    if data.photometric_errors_exists():
        logger.info('Photometric errors exist!')
        return
    logger.info('Calculating photometric errors...')
    for im1 in images:
        im1_all_matches, im1_valid_rmatches, im1_all_robust_matches = data.load_all_matches(im1)
        for im2 in im1_all_matches:
            rmatches = im1_all_matches[im2]
            if len(rmatches) == 0:
                continue
            args.append([0, 1, None, ctx.data, im1, im2, rmatches[:, 0:2].astype(int), flags])
    t_start = timer()
    p_results = []
    results = {}
    p = Pool(processes)
    logger.info('Using {} thread(s)'.format(processes))
    if processes == 1:
        for a, arg in enumerate(args):
            logger.info('Finished processing photometric errors: {} / {} : {} / {}'.format(a, len(args), arg[4], arg[5]))
            p_results.append(calculate_photometric_error_convex_hull(arg))
    else:
        p_results = p.map(calculate_photometric_error_convex_hull, args)
        p.close()

    for r in p_results:
        im1, im2, polygon_area, polygon_area_percentage, total_triangles, histogram = r
        if polygon_area is None or polygon_area_percentage is None:
            continue

        element = {'polygon_area': polygon_area, 'polygon_area_percentage': polygon_area_percentage, \
          'total_triangles': total_triangles, 'histogram': histogram}
        if im1 not in results:
            results[im1] = {}
        results[im1][im2] = element

    data.save_photometric_errors(results)

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

        for im2 in im1_all_matches:
            rmatches = im1_all_matches[im2]
            
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

def triplet_arguments(fns, Rs):
    unique_filenames = sorted(list(set(fns[:,0]).union(set(fns[:,1]))))
    for i, fn1 in enumerate(unique_filenames):
        yield unique_filenames, fns, Rs, i, fn1

def triplet_pairwise_arguments(fns, t_fns, t_errors, processes):
    fns_ = []
    args = []
    for i, (im1,im2) in enumerate(fns):
        fns_.append([im1, im2])
        if len(fns_) >= math.ceil(len(fns) / processes):
            args.append([None, fns_, t_fns, t_errors, False])
            fns_ = []
    
    if len(fns_) > 0:
        args.append([None, fns_, t_fns, t_errors, False])

    return args

def calculate_rotation_triplet_errors(arg):
  unique_filenames, fns, Rs, i, fn1 = arg
  results = {}

  t_start = timer()
  for j, fn2 in enumerate(unique_filenames):
    if j <= i:
      continue

    if fn2 not in results:
      results[fn2] = {}
    fn12_ris = np.where((fns[:,0] == fn1) & (fns[:,1] == fn2) | (fns[:,0] == fn2) & (fns[:,1] == fn1))[0]
    if len(fn12_ris) == 0:
      continue

    if fns[fn12_ris, 0][0] == fn1:
      R12 = np.matrix(Rs[fn12_ris].reshape(3,3))
    else:
      R12 = np.matrix(Rs[fn12_ris].reshape(3,3)).T

    for k, fn3 in enumerate(unique_filenames):
      if k <= j:
        continue

      fn23_ris = np.where((fns[:,0] == fn2) & (fns[:,1] == fn3) | (fns[:,0] == fn3) & (fns[:,1] == fn2))[0]
      if len(fn23_ris) == 0:
        continue

      if fns[fn23_ris, 0][0] == fn2:
        R23 = np.matrix(Rs[fn23_ris].reshape(3,3))
      else:
        R23 = np.matrix(Rs[fn23_ris].reshape(3,3)).T

      fn13_ris = np.where((fns[:,0] == fn1) & (fns[:,1] == fn3) | (fns[:,0] == fn3) & (fns[:,1] == fn1))[0]

      if len(fn13_ris) == 0:
        continue
      if fns[fn13_ris, 0][0] == fn1:
        R13 = np.matrix(Rs[fn13_ris].reshape(3,3))
      else:
        R13 = np.matrix(Rs[fn13_ris].reshape(3,3)).T

      R11 = R12 * R23 * R13.T
      error = np.arccos((np.trace(R11) - 1.0)/2.0)
      if np.isnan(error):
        error = -1.0

      results[fn2][fn3] = {'error': error, 'R11': R11.tolist(), 'triplet': '{}--{}--{}'.format(fn1,fn2,fn3)}
  return fn1, results

def calculate_triplet_error_histogram(errors, im1, im2, output_dir, debug):
  histogram, bins = np.histogram(np.array(errors), bins=80, range=(-2.0,78))
  epsilon = 0.000000001
  return histogram.tolist(), np.round((1.0 * histogram / (np.sum(histogram) + epsilon)), 2).tolist(), bins.tolist()

def flatten_triplets(triplets):
  flattened_triplets = {}
  fns1 = []
  fns2 = []
  fns3 = []
  errors = []
  for t1 in triplets:
      for t2 in triplets[t1]:
        for t3 in triplets[t1][t2]:
          triplet = [t1, t2, t3]
          flattened_triplets['--'.join(triplet)] = triplets[t1][t2][t3]['error']
          fns1.append(t1)
          fns2.append(t2)
          fns3.append(t3)
          errors.append(triplets[t1][t2][t3]['error'])

  return np.array(fns1), np.array(fns2), np.array(fns3), np.array(errors)

def calculate_triplet_pairwise_errors(arg):
  output_dir, fns, t_fns, t_errors, debug = arg
  errors = {}
  histograms = {}
  histograms_list = []
  stime = timer()
  for i, (im1,im2) in enumerate(fns):
    relevant_indices = np.where((t_fns[:,0] == im1) & (t_fns[:,1] == im2) | (t_fns[:,0] == im1) & (t_fns[:,2] == im2) | \
      (t_fns[:,1] == im1) & (t_fns[:,2] == im2))[0]

    histogram_counts, histogram, bins = calculate_triplet_error_histogram(t_errors[relevant_indices], im1, im2, output_dir, debug=debug)
    if im1 not in histograms:
        histograms[im1] = {}
    histograms[im1][im2] = { 'im1': im1, 'im2': im2, 'histogram': histogram, 'histogram-counts': histogram_counts, 'bins': bins }
    histograms_list.append({ 'im1': im1, 'im2': im2, 'histogram': histogram, 'histogram-counts': histogram_counts, 'bins': bins })
  return histograms, histograms_list

def calculate_triplet_errors(ctx):
    data = ctx.data
    cameras = data.load_camera_models()
    images = data.images()
    exifs = ctx.exifs
    config = data.config
    processes = ctx.data.config['processes']
    threshold = config['robust_matching_threshold']
    cached_p = {}
    fns = []
    Rs = []
    transformations = {}
    
    if data.triplet_errors_exists():
        logger.info('Triplet errors exist!')
        return

    if data.transformations_exists():
        transformations = data.load_transformations()
    else:
        transformations, t_num_pairs = calculate_transformations(ctx)

    for im1 in transformations:
        for im2 in transformations[im1]:
            R = transformations[im1][im2]['rotation']
            fns.append(np.array([im1, im2]))
            Rs.append(np.array(R).reshape((1,-1)))
    
    logger.info('Calculating triplet errors...')
    args = triplet_arguments(np.array(fns), np.array(Rs))
    triplet_results = {}
    p = Pool(processes)
    if processes > 1:
        t_results = p.map(calculate_rotation_triplet_errors, args)
    else:
        t_results = []
        for arg in args:
            t_results.append(calculate_rotation_triplet_errors(arg))
    for r in t_results:
        fn1, triplets = r
        triplet_results[fn1] = triplets
    data.save_triplet_errors(triplet_results)

    logger.info('Calculating triplet pairwise errors...')
    t_fns1, t_fns2, t_fns3, t_errors = flatten_triplets(triplet_results)
    t_fns = np.concatenate((t_fns1.reshape(-1,1), t_fns2.reshape(-1,1), t_fns3.reshape(-1,1)), axis=1)
    args = triplet_pairwise_arguments(np.array(fns), t_fns, t_errors, processes)
    p = Pool(processes)
    p_results = []
    t_start = timer()
    if processes == 1:
      for arg in args:
        p_results.append(calculate_triplet_pairwise_errors(arg))
    else:
      p_results = p.map(calculate_triplet_pairwise_errors, args)
    p.close()
    triplet_pairwise_results = {}
    for i, r in enumerate(p_results):
      histograms, histograms_list = r
      for k in histograms:
        if k not in triplet_pairwise_results:
            triplet_pairwise_results[k] = histograms[k]
        else:
            triplet_pairwise_results[k].update(histograms[k])
    data.save_triplet_pairwise_errors(triplet_pairwise_results)

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
            sequence_ranks[im1][im2] = j
    
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
        element = [data, im, data.load_reconstruction('reconstruction_gt.json'), 1.0,\
            config.get('error_inlier_threshold'), config.get('error_outlier_threshold')]
        args.append(element)

    p = Pool(processes)
    if processes == 1:    
        for arg in args:
            compute_matches_using_gt_reconstruction(arg)
    else:
        p.map(compute_matches_using_gt_reconstruction, args)

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

def rmatches_adapter(data, options={}):
    im_all_rmatches = {}
    im_num_rmatches = {}

    for img in data.images():
        _, _, rmatches = data.load_all_matches(img)
        im_all_rmatches[img] = rmatches

    for im1 in im_all_rmatches:
        if im1 not in im_num_rmatches:
            im_num_rmatches[im1] = {}
        for im2 in im_all_rmatches[im1]:
            im_num_rmatches[im1][im2] = len(im_all_rmatches[im1][im2])

    return im_num_rmatches

def vocab_tree_adapter(data, options={}):
    vtranks, vtscores = data.load_vocab_ranks_and_scores()
    vt_rank_scores_mean = {}
    vt_rank_scores_min = {}
    vt_rank_scores_max = {}
    # total_images = len(vt_ranks.keys())

    for im1 in vtranks:
        if im1 not in vt_rank_scores_mean:
            vt_rank_scores_mean[im1] = {}
            vt_rank_scores_min[im1] = {}
            vt_rank_scores_max[im1] = {}

        for im2 in vtranks[im1]:
            vt_rank_scores_mean[im1][im2] = 0.5 * vtranks[im1][im2] + 0.5*vtranks[im2][im1]
            vt_rank_scores_min[im1][im2] = min(vtranks[im1][im2], vtranks[im2][im1])
            vt_rank_scores_max[im1][im2] = max(vtranks[im1][im2], vtranks[im2][im1])

    return vt_rank_scores_mean, vt_rank_scores_min, vt_rank_scores_max

def sequence_rank_adapter(data, options={}):
    sequence_ranks = data.load_sequence_ranks()
    sequence_rank_scores_mean = {}
    sequence_rank_scores_min = {}
    sequence_rank_scores_max = {}
    total_images = len(sequence_ranks.keys())

    for im1 in sequence_ranks:
        if im1 not in sequence_rank_scores_mean:
            sequence_rank_scores_mean[im1] = {}
            sequence_rank_scores_min[im1] = {}
            sequence_rank_scores_max[im1] = {}

        for im2 in sequence_ranks[im1]:
            sequence_rank_scores_mean[im1][im2] = \
                0.5 * (total_images - sequence_ranks[im1][im2]) / total_images + \
                0.5 * (total_images - sequence_ranks[im2][im1]) / total_images
            sequence_rank_scores_min[im1][im2] = min(\
                (total_images - sequence_ranks[im1][im2]) / total_images,
                (total_images - sequence_ranks[im2][im1]) / total_images
                )
            sequence_rank_scores_max[im1][im2] = max(\
                (total_images - sequence_ranks[im1][im2]) / total_images,
                (total_images - sequence_ranks[im2][im1]) / total_images
                )

    return sequence_rank_scores_mean, sequence_rank_scores_min, sequence_rank_scores_max

def triplet_errors_adapter(data, options={}):
    triplet_errors = data.load_triplet_pairwise_errors()
    triplet_scores = {}

    for im1 in triplet_errors:
        if im1 not in triplet_scores:
            triplet_scores[im1] = {}
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

            cum_error = \
                np.sum( \
                    np.array(triplet_errors[im1][im2]['histogram-counts'][2:]) * \
                    np.power(0.5 + np.array(triplet_errors[im1][im2]['bins'][2:-1]), 1) \
                )

            # cum_error = \
            #     np.sum( \
            #         np.array(triplet_errors[im1][im2]['histogram-counts'][2:]) * \
            #         np.power(2.0, 0.5 * np.array(triplet_errors[im1][im2]['bins'][2:-1])) \
            #     )

            triplet_scores[im1][im2] = 1.0 / (cum_error + 1.0)

            if im1 == 'DSC_1140.JPG':
                if im1 in options['scores_gt'] and im2 in options['scores_gt'][im1]:
                    gts = options['scores_gt'][im1][im2]
                else:
                    gts = 0.0
                print '{}-{} : {}  {}'.format(im1, im2, cum_error, gts)

    return triplet_scores

def groundtruth_image_matching_results_adapter(data):
    gt_results = data.load_groundtruth_image_matching_results()
    scores_gt = {}
    for im1 in gt_results:
        if im1 not in scores_gt:
            scores_gt[im1] = {}
        for im2 in gt_results[im1]:
            scores_gt[im1][im2] = gt_results[im1][im2]['score']
    return scores_gt