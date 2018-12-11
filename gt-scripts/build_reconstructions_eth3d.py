#!/usr/bin/python
#! -*- encoding: utf-8 -*-
#
# eth3DToReconJSON.py
#
# python script to convert ETH3D data to opensfm reconstruction
# written by Joe DeGol
#
# usage : python eth3DToReconJSON.py <options>
#
# Options:
# -h  --help   help message
# -i  --indir  path to directory with images.txt, cameras.txt, and points3D.txt inside (e.g. ./courtyard)





#=============================================================================#
#================================== Preamble =================================#
#=============================================================================#

#=============== imports ===============#

# system
import os
import sys
import getopt
import math
import numpy as np

# opensfm
path =  './pipelines/baseline/OpenSfM-0.2.0-VT-Faster-BA/'
if not path in sys.path:
  sys.path.insert(1, path)
del path
from opensfm import dataset, matching, reconstruction, types, io

#============= End Imports =============#

#=============== Globals ===============#

myDataPath = ""

#============= End Globals =============#

#=============================================================================#
#================================ End Preamble ===============================#
#=============================================================================#





#=============================================================================#
#================================= Functions =================================#
#=============================================================================#


#=============== Intro ===============#
def intro():
    print('========================================')
    print('= eth3DToReconJSON.py ')
    print('========================================')
    print('=')
#============= End Intro =============#


#=============== Outro ===============#
def outro():
    print('=')
    print('========================================')
    print('= End eth3DToReconJSON.py ')
    print('========================================')
#============= End Outro =============#


#=============== Message ===============#
def message(str_param, newline_param=1, continued_param=0):

    #if continued, dont write start character
    if continued_param == 0:
        sys.stdout.write('= ')

    #write
    sys.stdout.write(str_param)

    #newline if requested
    if newline_param > 0:
        sys.stdout.write('\n')

    #flush
    sys.stdout.flush()
#============= End Message =============#


#=============== Help ===============#
def help():
    print('========================================')
    print('= Help Menu')
    print('========================================')
    message('usage:')
    message('python eth3DToReconJSON.py <options>')
    message('')
    message('-h  --help    :  this help menu')
    message('-i  --indir  path to directory with images.txt, cameras.txt, and points3D.txt inside')
    print('========================================')
#============= End Help =============#


#=============== Parse Args ===============#
def parse_args(argv):

    # variables
    global myDataPath
    inputPathGiven = False

    # try to getopt inputs
    try:
        opts, args = getopt.getopt(argv, "hi:c:", ["indir=","rchar="])
    except getopt.GetoptError:
        help()
        outro()
        sys.exit(2)

    # for each argument
    for opt, arg in opts:

        #help
        if opt in ('-h', "--help"):
            help()
            outro()
            sys.exit()

        # root dir
        if opt in ('-i', "--indir"):
            myDataPath = arg
            inputPathGiven = True

    if inputPathGiven is False:
        message('Error: No input directory given. Use -i option.')
        outro()
        sys.exit()

#============= End Parse Args =============#

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
def convertToReconJSONs():

    # variables
    global myDataPath

    # iterate file tree
    for root, dirs, files in os.walk(myDataPath):
        for file in files:
            if file == 'images.txt':
                images_file = os.path.join(root,file)
                cameras_file = os.path.join(root,'cameras.txt')
                points_file = os.path.join(root,'points3D.txt')
                message( 'processing ' + images_file )
                convertToReconJSON(images_file, cameras_file, points_file, root)

def convertToReconJSON(images_file, cameras_file, points_file, root):

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
    data = dataset.DataSet(root)
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

    #===== Setup =====#

    # intro
    intro()

    # parse arguments
    parse_args(sys.argv[1:])

    #=== End Setup ===#


    #===== Run =====#

    # iterate file tree and convertToReconJSON
    convertToReconJSONs()

    #=== End Run ===#


    #===== Shutdown =====#

    # outro
    outro()

    #=== End Shutdown ===#

#============= end main =============#

#=============== conditional script ===============#
if __name__ == "__main__":
    main()
#============= end conditional script =============#

#=============================================================================#
#=================================== End Main ================================#
#=============================================================================#
