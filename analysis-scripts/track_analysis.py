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
# import matching_classifiers # import load_classifier, calculate_per_image_mean_auc, calculate_dataset_auc, mkdir_p
import convnet

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def track_inlier_analysis(datasets, options):
    data_folder = 'data/track_analysis'
    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_{}.json'.format(os.path.basename(t))) # used for reading only
        dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_{}.json'.format(os.path.basename(t)))

        if os.path.isfile(dataset_track_inlier_analysis_fn):
            continue

        data = dataset.DataSet(t)
        graph = data.load_tracks_graph('tracks.csv')
        with open(dataset_track_analysis_fn, 'r') as fi:
            track_data = json.load(fi)
            all_robust_matches = {}
            track_degrees = {}
            total_tracks_dataset = 0
            for _, str_track_length in enumerate(sorted(track_data['total_tracks'])):
                if int(str_track_length) == 2:
                    continue
                print ('\tTrack length: {} -- Number of tracks: {}'.format(str_track_length, track_data['total_track_count'][str_track_length]))
                for ii, t in enumerate(track_data['total_tracks'][str_track_length]):
                    if str_track_length not in track_degrees:
                        track_degrees[str_track_length] = {}
                    # print '#'*100
                    # print ('\tTrack #{}/{}'.format(ii, total_tracks_dataset))
                    # print json.dumps(graph[t], sort_keys=True, indent=4, separators=(',', ': '))
                    total_matches = 0
                    for i, im1 in enumerate(sorted(graph[t].keys())):
                        fid1 = graph[t][im1]['feature_id']
                        if im1 not in all_robust_matches:
                            _, _, im1_all_robust_matches = data.load_all_matches(im1)
                            all_robust_matches[im1] = im1_all_robust_matches
                        for j, im2 in enumerate(sorted(graph[t].keys())):
                            if j <= i:
                                continue
                            if im2 not in all_robust_matches[im1]:
                                continue

                            fid2 = graph[t][im2]['feature_id']
                            rmatches = all_robust_matches[im1][im2]
                            if len(rmatches) == 0:
                                continue

                            if len(rmatches) < 20:
                                continue

                            if [fid1, fid2] in rmatches[:,0:2].tolist():
                                total_matches += 1

                    track_degrees[str_track_length][t] = total_matches

            with open(dataset_track_inlier_analysis_fn, 'w') as fout:
                json.dump(track_degrees, fout, sort_keys=True, indent=4, separators=(',', ': '))


    inlier_track_degree_sum = {}
    inlier_track_degree_count = {}
    outlier_track_degree_sum = {}
    outlier_track_degree_count = {}
    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_{}.json'.format(os.path.basename(t))) # used for reading only
        dataset_track_inlier_analysis_fn = os.path.join(data_folder, 'track_inlier_analysis_{}.json'.format(os.path.basename(t)))

        with open(dataset_track_analysis_fn, 'r') as fi:
            track_data = json.load(fi)
            with open(dataset_track_inlier_analysis_fn, 'r') as fi:
                track_degrees = json.load(fi)

                for str_track_length in track_degrees:
                    if str_track_length not in inlier_track_degree_sum:
                        inlier_track_degree_sum[str_track_length] = 0
                        inlier_track_degree_count[str_track_length] = 0
                    if str_track_length not in outlier_track_degree_sum:
                        outlier_track_degree_sum[str_track_length] = 0
                        outlier_track_degree_count[str_track_length] = 0

                    for t in track_degrees[str_track_length]:
                        if str_track_length in track_data['triangulated_tracks'] and t in track_data['triangulated_tracks'][str_track_length]:
                            inlier_track_degree_sum[str_track_length] += int(track_degrees[str_track_length][t])
                            inlier_track_degree_count[str_track_length] += 1
                        else:
                            outlier_track_degree_sum[str_track_length] += int(track_degrees[str_track_length][t])
                            outlier_track_degree_count[str_track_length] += 1


    inlier_track_degree_mean = []
    inlier_track_degree_lengths = []
    outlier_track_degree_mean = []
    outlier_track_degree_lengths = []

    epsilon = 0.0000001
    for track_length in sorted([int(d) for d in track_degrees.keys()]):
        inlier_track_degree_mean.append(1.0*inlier_track_degree_sum[str(track_length)] / (inlier_track_degree_count[str(track_length)] + epsilon))
        inlier_track_degree_lengths.append(track_length)
        outlier_track_degree_mean.append(1.0*outlier_track_degree_sum[str(track_length)] / (outlier_track_degree_count[str(track_length)] + epsilon))
        outlier_track_degree_lengths.append(track_length)
    legend = []
    
    fig = plt.figure()
    
    plt.ylabel('Mean matches in a track')
    plt.xlabel('Track lengths')
    plt.plot(inlier_track_degree_lengths, inlier_track_degree_mean, color='g')
    plt.plot(outlier_track_degree_lengths, outlier_track_degree_mean, color='r')

    
    # ax1=fig.add_subplot(111, label="1")
    # ax2=fig.add_subplot(111, label="2", frame_on=False)

    # ax1.set_xlim(2,max(track_lengths))
    # ax1.set_ylim(0,100.0)
    # ax1.plot(np.array(track_lengths), np.array(track_inlier_percentages))
    # ax1.set_xlabel('Track length')
    # ax1.set_ylabel('Inlier % (Percentage reconstructed)', color="b")
    # ax1.tick_params(axis='y', colors="b")
    # ax1.yaxis.tick_left()

    # ax2.set_xlim(2,max(track_lengths))
    # ax2.set_ylim(0,max(track_inlier_counts))
    # ax2.plot(np.array(track_lengths), np.array(track_inlier_counts), color='g')
    # ax2.yaxis.set_label_position('right')
    # ax2.yaxis.tick_right()
    # ax2.tick_params(axis='y', colors="g")
    # ax2.set_xlabel('Track length')
    # ax2.set_ylabel('Inlier Count (Percentage reconstructed)', color="g")
    
    plt.title('Track inlier degrees vs length')
    fig.set_size_inches(13.875, 7.875)
    plt.savefig(os.path.join(data_folder, 'track-degree-inliers.png'))


    import sys; sys.exit(1)


def triangulate_gt_reconstruction(data, graph):
    recon = data.load_reconstruction('reconstruction_gt.json')[0]
    reconstruction.retriangulate(graph, recon, data.config)
    reconstruction.paint_reconstruction(data, graph, recon)
    data.save_reconstruction([recon], filename='reconstruction_gt_triangulated.json')
    return recon

def track_length_analysis(datasets, options={}):
    data_folder = 'data/track_analysis'
    mkdir_p(data_folder)

    for i,t in enumerate(datasets):
        print ('Processing dataset: {}'.format(os.path.basename(t)))
        dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_{}.json'.format(os.path.basename(t)))

        if os.path.isfile(dataset_track_analysis_fn):
            continue

        total_track_count = {}
        triangulated_track_count = {}
        total_tracks = {}
        triangulated_tracks = {}

        data = dataset.DataSet(t)
        graph = data.load_tracks_graph('tracks.csv')
        if data.reconstruction_exists('reconstruction_gt_triangulated.json'):
            recon_gt_triangulated = data.load_reconstruction('reconstruction_gt_triangulated.json')[0]
        else:
            recon_gt_triangulated = triangulate_gt_reconstruction(data, graph)
           
        tracks, images = matching.tracks_and_images(graph)
        for t in tracks:
            track_length = len(graph[t].keys())
            total_track_count[track_length] = total_track_count.get(track_length, 0) + 1
            if track_length not in total_tracks:
                total_tracks[track_length] = []
            total_tracks[track_length].append(t)

            if t in recon_gt_triangulated.points.keys():
                triangulated_track_count[track_length] = triangulated_track_count.get(track_length, 0) + 1
                if track_length not in triangulated_tracks:
                    triangulated_tracks[track_length] = []
                triangulated_tracks[track_length].append(t)

        with open(dataset_track_analysis_fn, 'w') as fout:
            json.dump({'total_track_count': total_track_count, 'triangulated_track_count': triangulated_track_count, 'total_tracks': total_tracks, 'triangulated_tracks': triangulated_tracks}, fout, sort_keys=True, indent=4, separators=(',', ': '))

    total_track_count = {}
    triangulated_track_count = {}
    for i,t in enumerate(datasets):
        print ('Aggregating results - dataset: {}'.format(os.path.basename(t)))
        dataset_track_analysis_fn = os.path.join(data_folder, 'track_analysis_{}.json'.format(os.path.basename(t)))

        with open(dataset_track_analysis_fn, 'r') as fin:
            datum = json.load(fin)
            for k in datum['total_track_count'].keys():
                total_track_count[int(k)] = total_track_count.get(int(k),0) + int(datum['total_track_count'].get(k,0))
                triangulated_track_count[int(k)] = triangulated_track_count.get(int(k),0) + int(datum['triangulated_track_count'].get(k,0))

    track_lengths = []
    track_inlier_percentages = []
    track_inlier_counts = []
    for k in sorted(total_track_count.keys()):
        track_inlier_percentage = round(100.0 * triangulated_track_count.get(k,0) / total_track_count.get(k,0), 2)

        track_lengths.append(k)
        track_inlier_counts.append(triangulated_track_count.get(k,0))
        track_inlier_percentages.append(track_inlier_percentage)
        print ('\t{}:  {} / {} = {}'.format(k, triangulated_track_count.get(k,0), total_track_count.get(k,0), track_inlier_percentage))

    # Plotting track inlier graph
    fig = plt.figure()
    ax1=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax1.set_xlim(2,max(track_lengths))
    ax1.set_ylim(0,100.0)
    ax1.plot(np.array(track_lengths), np.array(track_inlier_percentages))
    ax1.set_xlabel('Track length')
    ax1.set_ylabel('Inlier % (Percentage reconstructed)', color="b")
    ax1.tick_params(axis='y', colors="b")
    ax1.yaxis.tick_left()

    ax2.set_xlim(2,max(track_lengths))
    ax2.set_ylim(0,max(track_inlier_counts))
    ax2.plot(np.array(track_lengths), np.array(track_inlier_counts), color='g')
    ax2.plot(np.array(track_lengths), np.array(total_track_count), color='r')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', colors="g")
    ax2.set_xlabel('Track length')
    ax2.set_ylabel('Inlier Count (Percentage reconstructed)', color="g")
    
    plt.title('Track inliers vs track length')
    plt.legend(['% track inliers', 'Inlier track count', 'Total track count'], loc='lower left', shadow=True, fontsize=10)

    fig.set_size_inches(13.875, 7.875)
    plt.savefig(os.path.join(data_folder, 'track-length-inliers.png'))



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
    # from opensfm.commands import create_tracks
    global dataset, matching, reconstruction

    options = {
        # 'classifier': parser_options.classifier, \
        # 'image_match_classifier_file': parser_options.image_match_classifier_file, \
        # 'image_match_classifier_min_match': int(parser_options.image_match_classifier_min_match), \
        # 'image_match_classifier_max_match': int(parser_options.image_match_classifier_max_match), \
        # 'feature_selection': False,
        # 'convnet_checkpoint': parser_options.convnet_checkpoint
    }

    datasets = [
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Barn',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Caterpillar',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Church',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Courthouse',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Ignatius',
    
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Meetingroom',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TanksAndTemples/Truck',

        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/courtyard',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/delivery_area',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/electro',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/facade',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/kicker',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/meadow',

        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/exhibition_hall',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/lecture_room',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/ETH3D/living_room',

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

        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_cabinet',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_large_cabinet',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_long_office_household',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_far',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_notexture_near_withloop',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_far',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_nostructure_texture_near_withloop',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_halfsphere',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_rpy',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_static',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_sitting_xyz',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_far',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_notexture_near',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_far',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_structure_texture_near',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_teddy',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_halfsphere',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_rpy',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_static',
        # '/hdd/Research/psfm-iccv/data/classifier-datasets-bruteforce/TUM_RGBD_SLAM/rgbd_dataset_freiburg3_walking_xyz',
    ]

    track_length_analysis(datasets, options)

    # track length analysis needs to run before this to create a list of inliers
    track_inlier_analysis(datasets, options)

if __name__ == '__main__':
    main(sys.argv)
