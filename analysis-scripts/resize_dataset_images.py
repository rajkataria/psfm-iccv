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

def resize_dataset(dset):
    data = dataset.DataSet(dset)
    original_image_folder = 'images'
    resized_image_folder = 'images-square'
    mkdir_p(os.path.join(dset, resized_image_folder))
    for im in data.images():
        img = cv2.imread(os.path.join(dset, original_image_folder, im))
        new_image_size = min(img.shape[1], img.shape[0])
        new_img = cv2.resize(img, (new_image_size, new_image_size))
        cv2.imwrite(os.path.join(dset, resized_image_folder, im), new_img)

  
def main():
    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('-d', '--dataset', help='dataset to scan')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')

    parser.add_argument('--debug', dest='debug', action='store_true', help='')
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import features, dataset, matching, classifier, reconstruction, types, io
    global features, matching, classifier, dataset

    resize_dataset(parser_options.dataset)
 
if __name__ == '__main__':
    main()