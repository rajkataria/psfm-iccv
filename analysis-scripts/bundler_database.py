import gzip
import sys
import glob
import json
import numpy as np
import os
import pickle
import argparse
import shutil
from PIL import Image

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def load_rmatches(rmatches_fn, rmatch_order):
    im_rmatches = {}
    with open(rmatches_fn, 'r') as f:
        counter = 0
        datum = f.readlines()
        for i,d in enumerate(datum):
            if d.strip() == '':
                continue
            
            indices = [int(x) for x in d.strip().split(' ')]
            rmatches = np.zeros((len(indices)/2,2))
            for jj, j in enumerate(indices):
                rmatches[int(jj/2), jj % 2] = indices[jj]

            im1 = rmatch_order[counter][0]
            im2 = rmatch_order[counter][1]
            if im1 not in im_rmatches:
                im_rmatches[im1] = {}
            im_rmatches[im1][im2] = rmatches
            counter += 1
    return im_rmatches

def load_matches(matches_fn, cameras):
    im_matches = {}
    match_images = []
    with open(matches_fn, 'r') as f:
        datum = f.readlines()
        matches_finished = True
        for i,d in enumerate(datum):
            if matches_finished:
                im1_id, im2_id = [int(x) for x in d.split(' ')]
                matches = []
                nmatches = -1 # not yet retrieved
                matches_finished = False
            else:
                if nmatches < 0:
                    nmatches = int(d)
                else:
                    matches.append([int(x) for x in d.split(' ')])
                    if len(matches) == nmatches:
                        matches_finished = True
                        im1 = cameras[im1_id]['name']
                        im2 = cameras[im2_id]['name']
                        # print cameras
                        # print '{} - {}'.format(im1_id, im2_id)
                        if im1 not in im_matches:
                            im_matches[im1] = {}
                        im_matches[im1][im2] = np.array(matches)
                        match_images.append([im1, im2])
    return match_images, im_matches

def load_cameras_and_features(data, image_list_fn):
    cameras = {}
    im_features = {}
    with open(image_list_fn, 'r') as f:
        datum = f.readlines()
        for i,d in enumerate(datum):
            name, _, focal_length = d.split(' ')
            image_name = os.path.basename(name)

            img = Image.open(os.path.join(data.data_path, 'images', image_name))
            cameras[i] = {
                'id': i,
                'name': image_name, 
                'width': img.width,
                'height': img.height,
                'focal_length': float(focal_length)
            }

            with gzip.open(os.path.join(data.data_path, image_name.split('.')[0] + '.key.gz'), 'r') as kin:
                feature_data = kin.readlines()
                keypoints = []
                descs = []
                desc = []
                for j,feature_datum in enumerate(feature_data):
                    if j == 0:
                        nmatches, _ = feature_datum.strip().split(' ')
                    else:
                        if j % 8 == 0:
                            converted_feature_datum = [int(kk) for kk in feature_datum.strip().split(' ')]
                            desc.extend(converted_feature_datum)
                            descs.append(desc)
                            desc = []
                        elif j % 8 == 1:
                            y,x,scale,orientation = feature_datum.split(' ')
                            keypoints.append([float(x),float(y)])
                        else:
                            converted_feature_datum = [int(kk) for kk in feature_datum.strip().split(' ')]
                            desc.extend(converted_feature_datum)

            im_features[image_name] = {
                'points': np.array(keypoints),
                'desc': np.array(descs),
                'colors': None,
                'width': img.width,
                'height': img.height
                }
    return cameras, im_features

def load_nrmatches(nrmatches_fn, cameras):
    rmatch_order = []
    num_rmatches = {}

    with open(nrmatches_fn, 'r') as f:
        datum = f.readlines()
        for i, d in enumerate(datum):
            if i == 0:
                continue
            else:
                im1 = cameras[i-1]['name']
                if im1 not in num_rmatches:
                    num_rmatches[im1] = {}

                rmatches = d.strip().split(' ')
                for rr, r in enumerate(rmatches):
                    if rr <= i-1:
                        continue
                    im2 = cameras[rr]['name']
                    num_rmatches[im1][im2] = int(r)
                    if num_rmatches[im1][im2] > 0:
                        rmatch_order.append([im1,im2])
    return rmatch_order, num_rmatches

def write_nrmatches(data, nrmatches_fn, cameras, num_rmatches):
    with open(nrmatches_fn, 'w') as f:
        f.write('{}\n'.format(len(data.images())))
        for i,im1 in enumerate(sorted(data.images())):
            for j,im2 in enumerate(sorted(data.images())):
                if j <= i:
                    f.write('0 ')
                    continue
                f.write('{} '.format(num_rmatches[im1][im2]))
            if i < len(data.images()) - 1:
                f.write('\n')

def write_rmatches(data, rmatches_fn, cameras, num_rmatches, im_rmatches):
    with open(rmatches_fn, 'w') as f:
        for i,im1 in enumerate(sorted(data.images())):
            for j,im2 in enumerate(sorted(data.images())):
                if j <= i:
                    f.write('\n')
                    continue
                if num_rmatches[im1][im2] > 0:
                    for m in im_rmatches[im1][im2]:
                        idx1, idx2 = m[0], m[1]
                        f.write('{} {} '.format(int(idx1), int(idx2) ))
                    f.write('\n')

def matches_osfm_to_bundler(database_path, threshold=0.5):
    data = dataset.DataSet(database_path)
    image_matching_results = data.load_image_matching_results()
    image_list_fn = os.path.join(data.data_path, 'list.txt')
    matches_fn = os.path.join(data.data_path, 'matches.init.txt')
    
    rmatches_fn = os.path.join(data.data_path, 'matches.ransac.txt')
    rmatches_fn_thresholded = os.path.join(data.data_path, 'matches.ransac-thresholded.txt')

    nrmatches_fn = os.path.join(data.data_path, 'nmatches.ransac.txt')
    nrmatches_fn_thresholded = os.path.join(data.data_path, 'nmatches.ransac-thresholded.txt')

    cameras, im_features = load_cameras_and_features(data, image_list_fn)
    match_images, im_matches = load_matches(matches_fn, cameras)
    rmatch_order, num_rmatches = load_nrmatches(nrmatches_fn, cameras)
    im_rmatches = load_rmatches(rmatches_fn, rmatch_order)

    for im1 in num_rmatches:
        for im2 in num_rmatches[im1]:
            if num_rmatches[im1][im2] > 0:
                if image_matching_results[im1][im2]['score'] < threshold:
                    num_rmatches[im1][im2] = 0
                    im_rmatches[im1][im2] = 0

    write_nrmatches(data, nrmatches_fn_thresholded, cameras, num_rmatches)
    write_rmatches(data, rmatches_fn_thresholded, cameras, num_rmatches, im_rmatches)

def matches_bundler_to_osfm(database_path):

    data = dataset.DataSet(database_path)
    mkdir_p(os.path.join(data.data_path, 'images'))

    for i in glob.glob(os.path.join(data.data_path,'*.jpg')):
        shutil.copy(i, os.path.join(os.path.dirname(i), 'images', os.path.basename(i)))

    image_list_fn = os.path.join(data.data_path, 'list.txt')
    matches_fn = os.path.join(data.data_path, 'matches.init.txt')
    rmatches_fn = os.path.join(data.data_path, 'matches.ransac.txt')
    nrmatches_fn = os.path.join(data.data_path, 'nmatches.ransac.txt')

    cameras, im_features = load_cameras_and_features(data, image_list_fn)
    match_images, im_matches = load_matches(matches_fn, cameras)
    rmatch_order, num_rmatches = load_nrmatches(nrmatches_fn, cameras)
    im_rmatches = load_rmatches(rmatches_fn, rmatch_order)

    for im in im_features:
        points, _, _ = features.mask_and_normalize_features(im_features[im]['points'], im_features[im]['desc'], im_features[im]['colors'], im_features[im]['width'], im_features[im]['height'], mask=None)
        im_features[im]['normalized_points'] = points

    for im in im_features:
        data.save_features(im, im_features[im]['points'], im_features[im]['desc'], im_features[im]['colors'])
        if im not in im_matches:
            im_matches[im] = {}
        if im not in im_rmatches:
            im_rmatches[im] = {}

        if len(im_matches[im].keys()) > 0:
            data.save_matches(im, im_rmatches[im])
            data.save_all_matches(im, im_matches[im], None, im_rmatches[im])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="bundler run path")
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    args = parser.parse_args()
    
    if not args.opensfm_path in sys.path:
        sys.path.insert(1, args.opensfm_path)
    from opensfm import features, dataset, matching, classifier, reconstruction, types, io
    global features
    
    # matches_bundler_to_osfm(args.database_path)
    matches_osfm_to_bundler(args.database_path)
