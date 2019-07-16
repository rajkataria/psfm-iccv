import json
import math
import numpy as np
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import rcParams
import seaborn as sns; sns.set()
import scipy
import sys
import torch

from argparse import ArgumentParser
import matching_classifiers # import load_classifier, calculate_per_image_mean_auc, calculate_dataset_auc, mkdir_p
import convnet

def plot_mds_rank_vs_gt_ranks(datasets, options={}):
    data_folder = 'data/mds-rank-analysis'
    matching_classifiers.mkdir_p(data_folder)
    gt_ranks = []
    mds_ranks = []
    options = {'shortest_path_label': 'rm-cost', 'lmds': False, 'PCA-n_components': 2, 'MDS-n_components': 3 ,'debug': True}
    for i,t in enumerate(datasets):
        data = dataset.DataSet(t)
        images = sorted(data.images())
        # im_closest_images = data.load_closest_images('rm-cost-lmds-False')
        # im_closest_images_gt = data.load_closest_images('gt-lmds-False')

        # print im_closest_images_gt
        # import sys; sys.exit(1)
        for im1 in images:
            im_closest_images = data.load_closest_images(im1, '{}-PCA_n_components-{}-MDS_n_components-{}-lmds-{}'.format(options['shortest_path_label'], options['PCA-n_components'], options['MDS-n_components'], options['lmds']))
            im_closest_images_gt = data.load_closest_images(im1, 'gt-PCA_n_components-{}-MDS_n_components-{}-lmds-{}'.format(options['PCA-n_components'], options['MDS-n_components'], options['lmds']))
            for im2 in im_closest_images_gt:
                try:
                    distance_rank_percentage_im1_im2_gt = 100.0 * im_closest_images_gt.index(im2) / len(data.images())
                    mds_rank_percentage_im1_im2 = 100.0 * im_closest_images.index(im2) / len(data.images())

                    gt_ranks.append(distance_rank_percentage_im1_im2_gt)
                    mds_ranks.append(mds_rank_percentage_im1_im2)
                    if distance_rank_percentage_im1_im2_gt > 100.0 or mds_rank_percentage_im1_im2 > 100.0:
                        print '#'*100
                        print t
                        print '{} / {}'.format(distance_rank_percentage_im1_im2_gt, mds_rank_percentage_im1_im2)
                        print im_closest_images_gt[im1].index(im2)
                        print len(data.images())
                        print '#'*100
                        import sys; sys.exit(1)
                except:
                    continue

    plt.hist2d(np.array(gt_ranks), np.array(mds_ranks), bins=25)
    plt.xlabel('Ground-truth Rank Percentiles')
    plt.ylabel('MDS Rank Percentiles')
    plt.title('MDS Rank Analysis')
    plt.savefig(os.path.join(data_folder, 'mds-rank-analysis.png'))
    # plt.show()

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    # parser.add_argument('-m', '--image_match_classifier_min_match', help='')
    # parser.add_argument('-x', '--image_match_classifier_max_match', help='')
    # parser.add_argument('-c', '--image_match_classifier_file', help='classifier to run')
    # parser.add_argument('--classifier', help='classifier type - BDT/CONVNET')
    # parser.add_argument('--convnet_checkpoint', help='checkpoint file for convnet')
  
    parser_options = parser.parse_args()

    if not parser_options.opensfm_path in sys.path:
        sys.path.insert(1, parser_options.opensfm_path)
    from opensfm import dataset, matching, classifier, reconstruction, types, io
    global matching, classifier, dataset

    options = {
        # 'classifier': parser_options.classifier, \
        # 'image_match_classifier_file': parser_options.image_match_classifier_file, \
        # 'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        # 'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        # 'feature_selection': False,
        # 'convnet_checkpoint': parser_options.convnet_checkpoint
    }

    training_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Barn',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Caterpillar',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Church',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courthouse',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ignatius',
    
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/facade',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/kicker',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/meadow',
    
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_360',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_desk2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_floor',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_plant',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_room',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg1_teddy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_360_hemisphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_coke',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_desk_with_person',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_dishes',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_no_loop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_large_with_loop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_metallic_sphere2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_360',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam2',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg2_pioneer_slam3',
            
    ]


    testing_datasets = [
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Meetingroom',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Truck',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lecture_room',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/living_room',

        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static',
        '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz',
    ]

    temp_datasets = [
        '/hdd/Research/psfm-iccv/data/temp-recons/exhibition_hall', 
    ]
    # plot_mds_rank_vs_gt_ranks(training_datasets + testing_datasets, options)
    plot_mds_rank_vs_gt_ranks(temp_datasets, options)

if __name__ == '__main__':
    main(sys.argv)
