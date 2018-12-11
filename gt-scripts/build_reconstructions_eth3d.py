
import os
import sys
import getopt
import math
import numpy as np

from argparse import ArgumentParser

#=============== Quaternion to AxisAngle ===============#
# convert quaternion to aa:
#   http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
#
# convert aa to compact aa:
#   https://stackoverflow.com/questions/12933284/rodrigues-into-eulerangles-and-vice-versa
#   https://stackoverflow.com/questions/38844493/transforming-quaternion-to-camera-rotation-matrix-opencv
def quat2aa(qw,qx,qy,qz):
    
    # normalize the quaternion
    if qw > 1:
        bottom = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw /= bottom
        qx /= bottom
        qy /= bottom
        qz /= bottom

    # angle
    angle = 2 * math.acos(qw)

    # axis
    s = math.sqrt(1-qw*qw)
    if s < 0.001: # avoid divide by zero
        x = 1
        y = 0
        z = 0
    else:
        x = qx / s
        y = qy / s
        z = qz / s

    # compact
    compact = [angle*x, angle*y, angle*z]
    return compact

#============= End Quaternion to AxisAngle =============#

#=============== Convert to Recon JSONs ===============#
def convertToReconJSONs(dataset_path):
    gt_folder = 'dslr_calibration_undistorted'

    images_file = os.path.join(dataset_path, gt_folder,'images.txt')
    cameras_file = os.path.join(dataset_path, gt_folder,'cameras.txt')
    points_file = os.path.join(dataset_path, gt_folder,'points3D.txt')
    convertToReconJSON(images_file, cameras_file, points_file, dataset_path)

def convertToReconJSON(images_file, cameras_file, points_file, dataset_path):

    # variables
    extrinsic_line = True

    # create reconstruction
    new_recon = types.Reconstruction()

    # read cameras file
    camfile = open(cameras_file,'r')
    for line in camfile:

        # split line into tokens
        tokens = line.split()

        # skip comments
        if tokens[0] == '#':
            continue

        # convert tokens into camera
        if tokens[1] == 'PINHOLE':
            new_cam = types.PerspectiveCamera()
            new_cam.id = tokens[0]
            new_cam.width = int(tokens[2])
            new_cam.height = int(tokens[3])
            new_cam.focal = 0.5 * ( float(tokens[4]) + float(tokens[5]) ) / max(int(tokens[2]), int(tokens[3]))
            new_recon.add_camera(new_cam)
        elif tokens[1] == 'THIN_PRISM_FISHEYE':
            new_cam = types.PerspectiveCamera()
            new_cam.id = tokens[0]
            new_cam.width = int(tokens[2])
            new_cam.height = int(tokens[3])
            new_cam.focal = 0.5 * ( float(tokens[4]) + float(tokens[5]) ) / max(int(tokens[2]), int(tokens[3]))
            new_cam.k1 = float( tokens[9] )
            new_cam.k2 = float( tokens[10] )
            new_recon.add_camera(new_cam)
        else:
            print 'unknown camera model: ', tokens[1]

    # read images file
    imfile = open(images_file,'r') 
    for line in imfile:

        # split line into tokens
        tokens = line.split()

        # skip comments
        if tokens[0] == '#':
            continue

        # every odd line after 4th is camera line
        if extrinsic_line:
            
            # split line into tokens
            tokens = line.split()
            qw = float(tokens[1])
            qx = float(tokens[2])
            qy = float(tokens[3])
            qz = float(tokens[4])
            tx = float(tokens[5])
            ty = float(tokens[6])
            tz = float(tokens[7])

            # convert tokens into pose
            new_pose = types.Pose()
            new_pose.translation = np.array( [tx, ty, tz] )
            new_pose.rotation = np.array( quat2aa(qw,qx,qy,qz) )

            # convert tokens into shot
            new_shot = types.Shot()
            new_shot.camera = new_recon.get_camera( tokens[8] )
            new_shot.pose = new_pose
            new_shot.id = os.path.basename(tokens[9])

            # add shot to reconstruction
            new_recon.add_shot(new_shot)

            # turn off extrinsic_line
            extrinsic_line = False

        # every even line after 4th is observations
        else:
            extrinsic_line = True

    # save as json
    data = dataset.DataSet(dataset_path)
    data.save_reconstruction([new_recon],'reconstruction_gt.json')

#============= End Convert to Recon JSONs =============#

#=============================================================================#
#================================ End Functions ==============================#
#=============================================================================#





#=============================================================================#
#==================================== Main ===================================#
#=============================================================================#

#=============== main ===============#
def main():
    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('-s', '--opensfm_path', help='opensfm path')
    parser.add_argument('-o', '--dataset_path', help='path of the dataset')

    parser.add_argument('--debug', dest='debug', action='store_true', help='show mask')
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
      sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, reconstruction, types, io

    global dataset, types

    convertToReconJSONs(parser_options.dataset_path)

if __name__ == "__main__":
    main()