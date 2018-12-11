import cv2
import glob
import json
import logging
import math
import numpy as np
import os
import sys
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def parse_dataset(dataset_path):
    transformations = []
    poses = sorted(glob.glob(dataset_path + '/poses/*.json'))
    images = sorted(glob.glob(dataset_path + '/images/*.png'))
    for fn in poses:
        with open(fn,'r') as fin:
            data = json.load(fin)
            T = np.array(data['extrinsic'])
            transformations.append(T)
    return images, transformations


def build_camera():
    camera = types.PerspectiveCamera()
    camera.id = "v2 unknown unknown 810 540 perspective 0"
    camera.width = 810
    camera.height = 540

    camera.focal_prior = 0.85
    camera.focal = 0.6265742216284438
    camera.k1 = 0.019126724353228715
    camera.k2 = -0.0030189241102539393
    camera.k1_prior = 0.0
    camera.k2_prior = 0.0
    return camera

def build_reconstruction(opensfm_path, dataset_path):
    if not opensfm_path in sys.path:
      sys.path.insert(1, opensfm_path)

    from opensfm import dataset, matching, reconstruction, types, io
    from opensfm.reconstruction import TrackTriangulator
    from opensfm import log
    global types

    images, T = parse_dataset(dataset_path)
    
    recon = types.Reconstruction()
    camera = build_camera()

    recon.add_camera(camera)
    offset = None

    opose = types.Pose()
    pose0 = types.Pose()
    for i, transformation in enumerate(T):
        pose = types.Pose()
        
        R = np.array(transformation[0:3,0:3])
        t = np.array(transformation[0:3,3])
  
        if np.linalg.det(R) < 0:
            R = R * -1.0

        pose.set_rotation_matrix(np.array(R))
        pose.translation = t
        
        if i == 0:
            pose0.rotation = pose.rotation
            pose0.translation = pose.translation
        pose.set_origin(pose.get_origin() - pose0.get_origin())
                
        shot = types.Shot()
        shot.camera = camera
        shot.pose = pose
        shot.id = os.path.basename(images[i])

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

    build_reconstruction(parser_options.opensfm_path, parser_options.dataset_path)

if __name__ == '__main__':
    main()