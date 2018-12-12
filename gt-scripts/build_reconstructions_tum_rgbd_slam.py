import cv2
import glob
import logging
import math
import numpy as np
import os
import pyquaternion
import sys

from build_reconstructions_eth3d import quat2aa
from argparse import ArgumentParser
from pyquaternion import Quaternion 
from shutil import copyfile


logger = logging.getLogger(__name__)

def mkdir_p(path):
  '''Make a directory including parent directories.
  '''
  try:
    os.makedirs(path)
  except os.error as exc:
    pass

def parse_log_file(log_file, sample_rate):
    rotations = []
    translations = []
    subsampled_images = []
    timestamps = []
    image_counter = -1
    with open(log_file,'r') as fin:
        data = fin.readlines()
        for i,d in enumerate(data):
            image_counter += 1
            if image_counter % sample_rate != 0:
                continue

            rgb_ts, rgb_img, gt_ts, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            translations.append([float(tx), float(ty), float(tz)])
            # rotations.append(quat2aa(float(qw), float(qx),float(qy),float(qz)))
            q = Quaternion(np.array([float(qw), float(qx), float(qy), float(qz)]))
            rotations.append(q.rotation_matrix)

            subsampled_images.append(rgb_img)
            timestamps.append(rgb_ts)
            # sys.exit(1)
    return rotations, translations, subsampled_images, timestamps

def build_camera():
    camera = types.PerspectiveCamera()
    camera.id = "v2 unknown unknown 640 480 perspective 0"
    camera.width = 640
    camera.height = 480
    camera.focal_prior = 0.85
    camera.focal = 0.8203
    camera.k1 = 0.0
    camera.k2 = 0.0
    camera.k1_prior = 0.0
    camera.k2_prior = 0.0
    return camera

def generate_dataset(dataset_path, subsampled_images):
    mkdir_p(os.path.join(dataset_path, 'images'))
    for i in subsampled_images:
        src = os.path.join(dataset_path, i)
        dest = os.path.join(dataset_path, 'images', os.path.basename(i))
        copyfile(src, dest)

def build_reconstruction(opensfm_path, log_file, dataset_path):
    if not opensfm_path in sys.path:
      sys.path.insert(1, opensfm_path)

    from opensfm import dataset, matching, reconstruction, types, io
    from opensfm.reconstruction import TrackTriangulator
    # from opensfm import learners
    from opensfm import log
    global types


    Rs, ts, subsampled_images, timestamps = parse_log_file(log_file, sample_rate=10)
    generate_dataset(dataset_path, subsampled_images)
    
    recon = types.Reconstruction()
    camera = build_camera()

    recon.add_camera(camera)
    for i, rotation in enumerate(Rs):
        pose = types.Pose()
        # pose.rotation = Rs[i]
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        pose.set_rotation_matrix(Rs[i])
        # pose.set_rotation_matrix(-pose.get_rotation_matrix())
        pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        pose.set_origin(np.array(ts[i]))
        # pose.translation = 20.0 * pose.translation
        # pose.set_rotation_matrix(np.random.rand(3,3))
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        # pose.set_origin(np.array(ts[i]))
        # pose.translation = np.array(ts[i])
        # pose.translation = ts[i]
        # pose.translation = 20.0 * pose.translation

        shot = types.Shot()
        shot.camera = camera
        shot.pose = pose
        shot.id = '{}.png'.format(timestamps[i])

        sm = types.ShotMetadata()
        sm.orientation = 1
        sm.gps_position = [0.0, 0.0, 0.0]
        sm.gps_dop = 999999.0
        shot.metadata = sm

        # add shot to reconstruction
        recon.add_shot(shot)

    data = dataset.DataSet(dataset_path)
    data.save_reconstruction([recon],'reconstruction_gt.json')

def main():
    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('-s', '--opensfm_path', help='opensfm path')
    # parser.add_argument('-l', '--log_file', help='input log file that contains transformation matrix')
    parser.add_argument('-o', '--dataset_path', help='path of the dataset')

    parser.add_argument('--debug', dest='debug', action='store_true', help='show mask')
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    logfile = os.path.join(parser_options.dataset_path, 'rgb-gt.txt' )
    build_reconstruction(parser_options.opensfm_path, logfile, parser_options.dataset_path)
    # build_reconstruction(parser_options.opensfm_path, parser_options.log_file, parser_options.dataset_path)

if __name__ == '__main__':
    main()
