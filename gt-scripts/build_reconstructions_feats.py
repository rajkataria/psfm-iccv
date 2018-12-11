import cv2
import logging
import math
import numpy as np
import os
import sys

from build_reconstructions_eth3d import quat2aa
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def parse_log_file(log_file):
    rotations = []
    translations = []
    subsampled_images = []
    with open(log_file,'r') as fin:
        data = fin.readlines()
        for i,d in enumerate(data):
            if i == 0:
                continue
            # image, timestamp, tx, ty, tz, qw, qx, qz, qy = d.split(' ')
            # translations.append([float(tx), float(tz), float(ty)])
            

            # image, timestamp, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            # translations.append([-float(tx), float(tz), -float(ty)])
            # rotations.append(quat2aa(-float(qw),-float(qx), float(qz),-float(qy)))

            # image, timestamp, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            # translations.append([-float(tz), float(ty), -float(tx)])
            # rotations.append(quat2aa(float(qw),-float(qz), float(qy),-float(qx)))

            # image, timestamp, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            # translations.append([float(tx), float(tz), float(tx)])
            # rotations.append(quat2aa(float(qw),-float(qx), -float(qz),-float(qy)))
            

            # image, timestamp, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            # translations.append([-float(tz), float(ty), -float(tx)])
            # rotations.append(quat2aa(float(qw),-float(qz), float(qx),-float(qy)))
            
            # 2000 pts
            # image, timestamp, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            # translations.append([-float(tz), -float(ty), -float(tx)])
            # aa = quat2aa(float(qw),-float(qz), -float(qy),-float(qx))
            # rotations.append(aa)

            image, timestamp, tx, ty, tz, qx, qy, qz, qw = d.split(' ')
            translations.append([-float(tz), -float(ty), -float(tx)])
            aa = quat2aa(float(qw),-float(qz), -float(qy),-float(qx))
            # aa[1] = aa[1] + math.pi/2.0
            # aa[1] = aa[1] - 90.0


            # aa[0] = math.fabs(aa[0])
            # aa[1] = math.fabs(aa[1])
            # aa[2] = math.fabs(aa[2])
            # print aa
            rotations.append(aa)

            # if aa[0] < 0 and i < 40:
            #     print 'hey'
            #     aa[0] = -1 * aa[0]
            # print aa 

            if False:
                qw = float(qw)
                qx = float(qx)
                qy = float(qy)
                qz = float(qz)
                R = np.zeros((3,3));
                R[0,0] = math.pow(qw,2) + math.pow(qx,2) - math.pow(qy,2) - math.pow(qz,2)
                R[0,1] = 2*qx*qy - 2*qz*qw;
                R[0,2] = 2*qx*qz + 2*qy*qw;

                R[1,0] = 2*qx*qy + 2*qz*qw;
                R[1,1] = math.pow(qw,2) - math.pow(qx,2) + math.pow(qy,2) - math.pow(qz,2);
                R[1,2] = 2*qy*qz - 2*qx*qw;

                R[2,0] = 2*qx*qz - 2*qy*qw;
                R[2,1] = 2*qx*qy + 2*qx*qw;
                R[2,2] = math.pow(qw,2) - math.pow(qx,2) - math.pow(qy,2) + math.pow(qz,2);
                
                rotations.append(R)
            # rotations.append([float(qx),float(qy),float(qz)])
            subsampled_images.append(image)

    return rotations, translations, subsampled_images

def build_camera():
    camera = types.PerspectiveCamera()
    camera.id = "v2 unknown unknown 752 480 perspective 0"
    camera.width = 752
    camera.height = 480
    # camera.focal = 0.7 * camera.width
    camera.focal_prior = 0.85
    # camera.focal = 0.85
    camera.focal = 1.2189825983284073
    # camera.k1 = 0.0
    # camera.k2 = 0.0
    camera.k1 = -0.35737618070251304
    camera.k2 = -0.009274326820465998
    camera.k1_prior = 0.0
    camera.k2_prior = 0.0
    return camera

def build_reconstruction(opensfm_path, log_file, dataset_path):
    if not opensfm_path in sys.path:
      sys.path.insert(1, opensfm_path)

    from opensfm import dataset, matching, reconstruction, types, io
    from opensfm.reconstruction import TrackTriangulator
    # from opensfm import learners
    # from opensfm import log
    global types


    Rs, ts, subsampled_images = parse_log_file(log_file)
    
    recon = types.Reconstruction()
    camera = build_camera()

    recon.add_camera(camera)

    offset = None
    pose0 = types.Pose()
    pose0_recon = types.Pose()
    for i, _ in enumerate(Rs):
        pose = types.Pose()
        pose.rotation = Rs[i]
        # pose.set_rotation_matrix(Rs[i])
        
        # print pose.get_rotation_matrix()

        pose.set_rotation_matrix(pose.get_rotation_matrix().T)
        pose.set_origin(np.array(ts[i]))


        if False and i == 0:
            print subsampled_images[i]
            # pose0 = types.Pose()
            # pose0_recon = types.Pose()

            pose0.rotation = pose.rotation
            pose0.translation = pose.translation

            pose0_recon.rotation = [1.46327114203856,  0.6520934519442041, -0.7289951219890223]
            pose0_recon.translation = [-151.62008675764042, 7.551077656334444, 32.03538718382186]

        
        if False:
            R_ = np.matrix(pose.get_rotation_matrix()) * np.matrix(pose0.get_rotation_matrix()).T * np.matrix(pose0_recon.get_rotation_matrix())
            pose.set_rotation_matrix(R_)
            
            # Bad cases
            if subsampled_images[i] == '1476939075123622.jpg':
                print '-'*100
                print '{} : {}'.format(subsampled_images[i], pose.rotation)
                pose.rotation[2] = math.pi + pose.rotation[2]
            # Good cases
            if subsampled_images[i] == '1476939074934112.jpg':
                print '+'*100
                print '{} : {}'.format(subsampled_images[i], pose.rotation)
            # if offset is None:
            #     offset = pose.get_origin() - pose0_recon.get_origin()

            # pose.set_origin(pose.get_origin() - offset)
            # pose.translation = pose.translation * 0.1

            # pose.translation = np.array(ts[i])
            # print pose.get_origin()
            
            # print pose.get_rotation_matrix()
            # print pose.get_rotation_matrix()
            # print pose.get_origin()
            # sys.exit(1)
            
            # t = pose.translation
            # t[0] = -t[0]
            # t[1] = -t[1]
            # t[2] = -t[2]
            # t[1],t[2] = t[2],t[1]
            # pose.translation = t

            # print pose.rotation
            # print pose.translation
            # print '#'*100
            # sys.exit(1)
            # R = pose.get_rotation_matrix()
            # R[:,1], R[:,2] = R[:,2], R[:,1]
            # pose.set_rotation_matrix(R)
            
            # R = pose.get_rotation_matrix()
            # R[:,1], R[:,2] = R[:,2], R[:,1]
            # pose.set_rotation_matrix(R.T)
            # t = pose.translation
            # t = pose.get_rotation_matrix() * np.matrix(pose.translation.reshape((3,1)))
            # print pose.translation
        
        pose.translation = 20.0 * pose.translation
        # print pose.translation
        # sys.exit(1)

        shot = types.Shot()
        shot.camera = camera
        shot.pose = pose
        shot.id = subsampled_images[i]

        sm = types.ShotMetadata()
        sm.orientation = 1
        sm.gps_position = [0.0, 0.0, 0.0]
        sm.gps_dop = 999999.0
        # sm.capture_time = 0.0
        shot.metadata = sm

        # add shot to reconstruction
        recon.add_shot(shot)

    data = dataset.DataSet(dataset_path)
    data.save_reconstruction([recon],'reconstruction_gt.json')

def main():
    parser = ArgumentParser(
        description='test apriltag Python bindings')

    parser.add_argument('-s', '--opensfm_path', help='opensfm path')
    parser.add_argument('-l', '--log_file', help='input log file that contains transformation matrix')
    parser.add_argument('-o', '--dataset_path', help='path of the dataset')

    parser.add_argument('--debug', dest='debug', action='store_true', help='show mask')
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    build_reconstruction(parser_options.opensfm_path, parser_options.log_file, parser_options.dataset_path)

if __name__ == '__main__':
    main()