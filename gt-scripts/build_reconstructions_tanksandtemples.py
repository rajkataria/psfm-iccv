import cv2
import logging
import math
import numpy as np
import os
import sys
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def parse_log_file(log_file):
    transformations = []
    with open(log_file,'r') as fin:
        data = fin.readlines()
        num_cameras = len(data) / 5
        
        for i,d in enumerate(data):
            camera = int(i / 5)
            if i % 5 == 0:
                if i > 0:
                    transformations.append(T)
                    # return transformations
                T = np.zeros((4, 4))
                continue
            else:
                T[i % 5 - 1,:] = d.split(' ')
    return transformations

def parse_trans_file(trans_file):
    T_g = np.zeros((4, 4))
    with open(trans_file,'r') as fin:
        data = fin.readlines()
        
        for i,d in enumerate(data):
            T_g[i,:] = d.split(' ')
    return T_g

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def build_camera():
    # new_cam = types.PerspectiveCamera()
    # new_cam.id = tokens[0]
    # new_cam.width = int(tokens[2])
    # new_cam.height = int(tokens[3])
    # new_cam.focal = 0.5 * ( float(tokens[4]) + float(tokens[5]) )
    # new_recon.add_camera(new_cam)

    camera = types.PerspectiveCamera()
    camera.id = "v2 sony a7sm2 1920 1080 perspective 0.5833"
    camera.width = 1920
    camera.height = 1080
    # camera.focal = 0.7 * camera.width
    camera.focal_prior = 0.5833333333333334
    camera.focal = 0.6069623805001559
    # camera.k1 = -0.06184428051526924
    # camera.k2 = 0.04138070097075985
    camera.k1 = 0.0
    camera.k2 = 0.0
    camera.k1_prior = 0.0
    camera.k2_prior = 0.0
    return camera
    # {
    # "v2 sony a7sm2 1920 1080 perspective 0.5833": {
    #     "focal_prior": 0.5833333333333334, 
    #     "width": 1920, 
    #     "k1": 0.0, 
    #     "k2": 0.0, 
    #     "k1_prior": 0.0, 
    #     "k2_prior": 0.0, 
    #     "projection_type": "perspective", 
    #     "focal": 0.5833333333333334, 
    #     "height": 1080
    #     }
    # }

def build_reconstruction(opensfm_path, log_file, trans_file, dataset_path):
    if not opensfm_path in sys.path:
      sys.path.insert(1, opensfm_path)

    from opensfm import dataset, matching, reconstruction, types, io
    from opensfm.reconstruction import TrackTriangulator
    # from opensfm import learners
    from opensfm import log
    global types

    T = parse_log_file(log_file)
    T_g = parse_trans_file(trans_file)
    pose_g = types.Pose()
    pose_g.set_rotation_matrix(np.matrix(T_g[0:3, 0:3]) )
    pose_g.set_rotation_matrix(np.matrix(T_g[0:3, 0:3]).T )
    pose_g.set_origin(T_g[0:3, 3])
    
    recon = types.Reconstruction()
    camera = build_camera()

    recon.add_camera(camera)
    for i, transformation in enumerate(T):
        pose = types.Pose()
        
        pose.set_rotation_matrix(np.matrix(transformation[0:3, 0:3]) )
            

        pose.set_rotation_matrix(np.matrix(transformation[0:3, 0:3]).T )
        # pose.translation = np.array( (transformation[0:3, 3]).tolist() )
        
        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = pose.rotation[0], pose.rotation[1], pose.rotation[2]
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        pose.set_origin(transformation[0:3, 3])
        pose = pose.compose(pose_g)
        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = -pose.rotation[2] - 3*np.pi/2, pose.rotation[0]  - np.pi/2.0, -pose.rotation[1] + 3*np.pi/2.0
        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = pose.rotation[0]- np.pi/2.0, pose.rotation[1] - np.pi/2.0, pose.rotation[2]

        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = -pose.rotation[2], pose.rotation[0], -pose.rotation[1]
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        # pose.set_origin([-transformation[2,3], transformation[0,3], -transformation[1,3]])
        
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = pose.rotation[0] + np.pi/2.0, pose.rotation[1] + np.pi/2.0, pose.rotation[2] + np.pi/2.0
        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = pose.rotation[0] , pose.rotation[1] , pose.rotation[2] 
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        # pose.set_origin(transformation[0:3,3])


        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = -pose.rotation[2], pose.rotation[0], -pose.rotation[1]
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        # pose.set_origin([-transformation[2,3], transformation[0,3], -transformation[1,3]])


        # pose.rotation[0], pose.rotation[1], pose.rotation[2] = -pose.rotation[2], pose.rotation[0], -pose.rotation[1]
        # pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        # pose.set_origin([-transformation[2,3], transformation[0,3], -transformation[1,3]])

        # pose.set_origin(transformation[0:3,3])

        # pose.translation[0], pose.translation[1], pose.translation[2] = -pose.translation[2], pose.translation[0], -pose.translation[1]

        # pose.translation = -pose.translation
        # pose.translation[1], pose.translation[2] = pose.translation[2], -pose.translation[1]
        # o = pose.get_origin()
        # pose.set_origin([o[1], -o[2], o[0]])
        # pose.set_origin([o[0], o[2], o[1]])
        # pose.translation[0], pose.translation[1], pose.translation[2] = -pose.translation[2], pose.translation[0], -pose.translation[1]
        # M = pose.get_rotation_matrix()
        # M[:,1], M[:,2] = M[:,2], M[:,1]
        # M[1,:], M[2,:] = M[2,:], M[1,:]
        # pose.set_rotation_matrix(M)
        # pose.translation[1], pose.translation[2] = pose.translation[2], pose.translation[1]
        
        # R = pose.rotation
        # print R
        # R[1], R[2] = -R[1], -R[2]  # Reverse y and z
        # R[1], R[2] = -R[2], -R[1]  # Reverse y and z
        # pose.rotation = R.T
        # print pose.rotation
        # print '='*100

        # t = pose.translation
        # t = -pose.get_rotation_matrix() * np.matrix(pose.translation.reshape((3,1)))

        # pose.translation = t.reshape((3,)).tolist()[0]
        # print pose.translation
        
        shot = types.Shot()
        shot.camera = camera
        shot.pose = pose
        shot.id = '{}.jpg'.format(str(i + 1).zfill(6))

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


    logfile = os.path.join(parser_options.dataset_path, '{}_COLMAP_SfM.log'.format(os.path.basename(os.path.normpath(parser_options.dataset_path))) )
    transfile = os.path.join(parser_options.dataset_path, '{}_trans.txt'.format(os.path.basename(os.path.normpath(parser_options.dataset_path))) )
    # print (logfile)
    # sys.exit(1)
    build_reconstruction(parser_options.opensfm_path, logfile, transfile, parser_options.dataset_path)

if __name__ == '__main__':
    main()