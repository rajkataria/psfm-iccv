import cv2
import glob
import gzip
import numpy as np
import os
import pickle
import random
import sys
from argparse import ArgumentParser

# from patchdataset import load_feature_matching_dataset, load_datasets

def mkdir_p(path):
  '''Make a directory including parent directories.
  '''
  try:
    os.makedirs(path)
  except os.error as exc:
    pass

def draw_matches(im1, p1, im2, p2, rmatches, colors, label=None):
    height,width,channels = im1.shape
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,height-50)
    fontScale              = 3
    fontColor              = (0,0,255)
    lineType               = 10

    if len(rmatches) > 0:
        pts_normalized_img1 = p1[rmatches[:,0].astype(int),0:2]
        pts_normalized_img2 = p2[rmatches[:,1].astype(int),0:2]
        
        pts_denormalized_img1 = features.denormalized_image_coordinates(pts_normalized_img1, width, height)
        pts_denormalized_img2 = features.denormalized_image_coordinates(pts_normalized_img2, width, height)

        for i, _ in enumerate(pts_denormalized_img1):
            # Draw points
            cv2.circle(im1, (int(pts_denormalized_img1[i][0]), int(pts_denormalized_img1[i][1])), 20, colors[i%30], -1)
            cv2.circle(im2, (int(pts_denormalized_img2[i][0]), int(pts_denormalized_img2[i][1])), 20, colors[i%30], -1)

    if label is not None:
      im_text = 'Total robust matches: {} ; Label: {}'.format(len(rmatches), label)
    else:
      im_text = 'Total robust matches: {}'.format(len(rmatches))
    cv2.putText(im1, im_text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def iterate_gt_matches(dset, colors, im1_filters, im2_filters):
  # fm_dsets, fm_fns, [fm_np_indices_1, fm_np_indices_2, fm_np_distances, fm_np_errors, fm_np_size1, fm_np_size2, \
  #   fm_np_angle1, fm_np_angle2, fm_np_rerr1, fm_np_rerr2, fm_np_labels] = \
  #   load_datasets(load_feature_matching_dataset, [patchdataset], load_file_names=True)  
  data = dataset.DataSet(dset)
  fm_fns, [fm_indices_1, fm_indices_2, fm_lowes_ratio_1, fm_lowes_ratio_2, fm_errors, fm_size1, fm_size2, fm_angle1, fm_angle2, fm_rerr1, \
    fm_rerr2, fm_labels] = data.load_feature_matching_dataset(lowes_threshold=0.8)

  im_inliers = {}
  im_outliers = {}
  im_inliers_p1s = {}
  im_inliers_p2s = {}
  labels = {}

  for i, (a,b) in enumerate(fm_fns):
    if a < b:
      im = a
      key = b
    else:
      im = b
      key = a

    if not im_inliers.has_key(im):
      im_inliers[im] = {key: 0}
      im_outliers[im] = {key: 0}
      im_inliers_p1s[im] = {key: []}
      im_inliers_p2s[im] = {key: []}
    if not im_inliers[im].has_key(key):
      im_inliers[im][key] = 0
      im_outliers[im][key] = 0
      im_inliers_p1s[im][key] = []
      im_inliers_p2s[im][key] = []

    if fm_labels[i] >= 1:
      im_inliers[im][key] = im_inliers[im][key] + 1
      im_inliers_p1s[im][key].append(fm_indices_1[i].astype(np.int))
      im_inliers_p2s[im][key].append(fm_indices_2[i].astype(np.int))
    else:
      im_outliers[im][key] = im_outliers[im][key] + 1

  # print sorted(im_inliers['DSC_1761.JPG'].keys())
  # print '='*100
  # print im_inliers['DSC_1761.JPG']['DSC_1762.JPG']
  # print '#'*100
  # print im_inliers['DSC_1761.JPG']['DSC_1804.JPG']
  # print '$'*100
  # import sys; sys.exit(1)
  for im1_fn in im_inliers.keys():
    if os.path.basename(im1_fn) not in im1_filters:
      continue
    print 'Processing file: {}'.format(im1_fn)
    for im2_fn in im_inliers[im1_fn].keys():
      if len(im2_filters) > 0 and os.path.basename(im2_fn) not in im2_filters:
        continue
      # p1, f1, c1 = load_features(os.path.join(dset, 'features', im1_fn))
      # p2, f2, c2 = load_features(os.path.join(dset, 'features', im2_fn))
      p1, f1, c1 = data.load_features(im1_fn)
      p2, f2, c2 = data.load_features(im2_fn)
      im1_indices = np.array(im_inliers_p1s[im1_fn][im2_fn])
      im2_indices = np.array(im_inliers_p2s[im1_fn][im2_fn])

      matches = np.concatenate((im1_indices.reshape(-1,1), im2_indices.reshape(-1,1)), axis=1)
      im1 = cv2.imread(os.path.join(dset,'images',im1_fn))
      im2 = cv2.imread(os.path.join(dset,'images',im2_fn))
      draw_matches(im1, p1, im2, p2, matches, colors)
      viz_filename = os.path.join(dset, 'match_visualizations_gt', '{}---{}.jpeg'.format(os.path.basename(im1_fn).split('.')[0], \
        os.path.basename(im2_fn).split('.')[0]))
      # imgs = np.concatenate((cv2.resize(im1, (225, 225)),cv2.resize(im2, (225, 225))),axis=1)

      im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
      imgs = np.concatenate((im1,im2),axis=1)
      cv2.imwrite(viz_filename, cv2.resize(imgs, (2560, 960)) )

def iterate_matches(dset, colors, im1_filters, im2_filters):
  data = dataset.DataSet(dset)
  for matches_filename in glob.glob(os.path.join(dset,'matches','*')):
    # if os.path.basename(matches_filename) not in ['2017-11-22_17-51-50_494.jpeg_matches.pkl.gz', \
    #     '2017-11-22_17-51-47_591.jpeg_matches.pkl.gz', '2017-11-22_17-51-07_202.jpeg_matches.pkl.gz']:
    #     continue
    # if os.path.basename(matches_filename) not in ['2017-11-22_17-53-35_744.jpeg_matches.pkl.gz', \
    #     '2017-11-22_17-54-40_215.jpeg_matches.pkl.gz', '2017-11-22_17-52-56_108.jpeg_matches.pkl.gz',
    #     '2017-11-22_17-53-28_363.jpeg_matches.pkl.gz', '2017-11-22_17-54-32_708.jpeg_matches.pkl.gz']:
    #     continue
    im1_fn = os.path.basename(matches_filename).split('_matches.')[0]
    if len(im1_filters) != 0:
      if os.path.basename(im1_fn) not in im1_filters:
          continue
    p1, f1, c1 = data.load_features(im1_fn)
    print 'Processing file: {}'.format(im1_fn)
    with gzip.open(matches_filename, 'rb') as fin:
      matches = pickle.load(fin)

      for im2_fn in matches:
        if len(im2_filters) > 0 and os.path.basename(im2_fn) not in im2_filters:
          continue
        im1 = cv2.imread(os.path.join(dset,'images',im1_fn))
        im2 = cv2.imread(os.path.join(dset,'images',im2_fn))
        p2, f2, c2 = data.load_features(im2_fn)
        draw_matches(im1, p1, im2, p2, matches[im2_fn], colors)
        viz_filename = os.path.join(dset, 'match_visualizations', '{}---{}.jpeg'.format(os.path.basename(im1_fn).split('.')[0], \
          os.path.basename(im2_fn).split('.')[0]))
        # imgs = np.concatenate((cv2.resize(im1, (225, 225)),cv2.resize(im2, (225, 225))),axis=1)
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
        imgs = np.concatenate((im1,im2),axis=1)
        cv2.imwrite(viz_filename, cv2.resize(imgs, (2560, 960)) )

  # print all_matches
def main():
    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('-d', '--dataset', help='dataset to scan')
    # parser.add_argument('-p', '--patchdataset', help='dataset to scan')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')

    parser.add_argument('--debug', dest='debug', action='store_true', help='show mask')
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import features, dataset, matching, classifier, reconstruction, types, io
    global features, matching, classifier, dataset

    mkdir_p(os.path.join(parser_options.dataset,'match_visualizations'))
    mkdir_p(os.path.join(parser_options.dataset,'match_visualizations_gt'))

    colors = [(int(random.random()*255), int(random.random()*255), int(random.random()*255)) for i in xrange(0,30)]
    
    # filters = [
    #   '2017-11-22_17-51-36_646.jpeg', '2017-11-22_17-51-55_732.jpeg', '2017-11-22_17-52-00_904.jpeg','2017-11-22_17-51-24_734.jpeg', # FNs
    #   '2017-11-22_17-53-35_744.jpeg', '2017-11-22_17-54-40_215.jpeg', '2017-11-22_17-52-56_108.jpeg','2017-11-22_17-53-28_363.jpeg', '2017-11-22_17-54-32_708.jpeg' # FPs
    #   ]
    
    # filters = [
    #   '2017-11-22_18-11-40_276.jpeg', '2017-11-22_18-16-13_417.jpeg'
    # ]

    # filters = [
    #     '2017-11-22_17-53-14_116.jpeg', '2017-11-22_18-14-54_337.jpeg', '2017-11-22_17-50-12_935.jpeg', '2017-11-22_18-07-18_812.jpeg'
    # ]

    # filters = [
    #     '000014.jpg', '000193.jpg', '000261.jpg', '000247.jpg', '000419.jpg'
    # ]

    # exhibition_hall
    # im1_filters = ['DSC_1761.JPG', 'DSC_1770.JPG']
    # im2_filters = ['DSC_1762.JPG', 'DSC_1804.JPG']
    im1_filters = ['DSC_1744.JPG']
    im2_filters = ['DSC_1760.JPG', 'DSC_1780.JPG', 'DSC_1800.JPG', 'DSC_1811.JPG']
    # im1_filters = ['DSC_1745.JPG']
    # im2_filters = ['DSC_1802.JPG']
    # im1_filters = ['DSC_1800.JPG']
    # im2_filters = ['DSC_1807.JPG', 'DSC_1812.JPG']
    # im1_filters = ['DSC_1770.JPG']
    # im2_filters = ['DSC_1773.JPG', 'DSC_1812.JPG', 'DSC_1794.JPG']
    # im1_filters = ['DSC_1755.JPG']
    # im2_filters = ['DSC_1795.JPG']
    # im1_filters = ['DSC_1744.JPG']
    # im2_filters = ['DSC_1760.JPG']
    # im1_filters = ['DSC_1761.JPG']
    # im2_filters = ['DSC_1780.JPG']
    # im1_filters = ['DSC_1744.JPG']
    # im2_filters = ['DSC_1780.JPG']
    

    # im1_filters = ['DSC_1744.JPG', 'DSC_1803.JPG', 'DSC_1746.JPG', 'DSC_1785.JPG']
    # im2_filters = ['DSC_1803.JPG', 'DSC_1746.JPG', 'DSC_1785.JPG', 'DSC_1800.JPG']
    im1_filters = ['DSC_1744.JPG', 'DSC_1784.JPG']
    im2_filters = ['DSC_1784.JPG', 'DSC_1800.JPG']

    im1_filters = ['DSC_1744.JPG', u'DSC_1766.JPG', u'DSC_1785.JPG', u'DSC_1803.JPG', u'DSC_1802.JPG', 'DSC_1800.JPG']
    im2_filters = [u'DSC_1766.JPG', u'DSC_1785.JPG', u'DSC_1803.JPG', u'DSC_1802.JPG', u'DSC_1800.JPG']

    # courtyard
    # im1_filters = ['DSC_0291.JPG']
    # im2_filters = ['DSC_0309.JPG']

    iterate_gt_matches(parser_options.dataset, colors, im1_filters, im2_filters)
    iterate_matches(parser_options.dataset, colors, im1_filters, im2_filters)
 
if __name__ == '__main__':
    main()