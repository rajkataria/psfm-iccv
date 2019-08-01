import json
import logging
import math
import networkx as nx
import numpy as np
import opensfm
import operator
import os
import sys
from timeit import default_timer as timer

from opensfm import dataset
from opensfm import io
from opensfm import matching
from opensfm import types

logger = logging.getLogger(__name__)

class Command:
    name = 'yan'
    help = "Implementation of the paper 'Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context' - http://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        tracks_graph = data.load_tracks_graph('tracks.csv')

        G, iconic_images, non_iconic_images = network_construction(data, tracks_graph) # path network
        # print ('*'*100)
        # print ('Iconic images: {}'.format(iconic_images))
        # print ('#'*100)
        # print ('Non-iconic images: {}'.format(non_iconic_images))
        # print ('='*100)
        # import sys; sys.exit(1)
        tracks_graph_regenerated = track_regeneration(data, G, tracks_graph, iconic_images, non_iconic_images)
        data.save_tracks_graph(tracks_graph_regenerated, 'tracks-yan.csv')
        

def iconic_set_points(data, tracks_graph, C):
    T_A = set()
    for image in C:
        T_A = T_A.union(set(tracks_graph[image].keys()))
    return T_A

def confusing_points(data, tracks_graph, C):
    D = set()
    for i in C:
        for j in C:
            if i == j:
                continue
            D = D.union(set(tracks_graph[i].keys()).intersection(set(tracks_graph[j].keys())))
    return D

# def geodetically_consistent(G, tracks_graph, i, j, p):
#     paths = nx.all_simple_paths(G, source=i, target=j)
#     if paths == 0:
#         return -1
#     for path in paths:
#         consistent = True
#         # Make sure every image in the path from image i to image j sees the track p
#         for node in path:
#             if node not in tracks_graph[p]:
#                 consistent = False
#         if consistent:
#             return 1
#     return -1

# def objective(G, tracks_graph):
#     tracks, _ = matching.tracks_and_images(tracks_graph)
#     tracks_prime, _ = matching.tracks_and_images(tracks_graph_prime)
#     for t in tracks:
#         continue

#     pass

def track_regeneration(data, G, V, iconic_images, non_iconic_images):
    current_track_counter = 1
    image_tracks = {}
    old_to_new_track_mapping = {}
    new_to_old_track_mapping = {}

    # tracks, images = matching.tracks_and_images(V)
    tracks, images = matching.tracks_and_images(V)
    # images = list(images)
    # tracks = list(tracks)

    V_prime = nx.Graph()
    V_prime.add_nodes_from(images, bipartite=0)
    # V_prime.add_nodes_from(tracks, bipartite=1)

    # print '{} / {}'.format(len(images), len(tracks))
    # tracks_, images_ = matching.tracks_and_images(V)
    # print ('{}_ / {}_'.format(len(images_), len(tracks_)))
    
    # tracks_prime_, images_prime_ = matching.tracks_and_images(V_prime)
    # print '#'*100
    # print ('{}_ / {}_'.format(len(images_prime_), len(tracks_prime_)))
    
    # for i in V_prime.nodes(data=True):
        # print i
    # import sys; sys.exit(1)

    cum_common_tracks_prime = 0
    cum_common_tracks = 0
    # source = sorted(G.nodes())[0]
    # for i,j in nx.bfs_edges(G, source):
    # for count in range(0,3):
    

    for i in sorted(G.nodes()):
        # if i != 'P1010141.jpg' and i != 'P1010159.jpg':
        #     continue
        for j in sorted(G.neighbors(i)):
            # if j != 'P1010145.jpg':
            #     continue

            # print ('{} - {}'.format(i,j))
            common_tracks = len(set(V[i].keys()).intersection(set(V[j].keys())))
            cum_common_tracks += common_tracks
            # print ('\tCommon tracks: {}'.format(common_tracks))

            if i not in image_tracks:
                image_tracks[i] = {}
            if j not in image_tracks:
                image_tracks[j] = {}
            for t in V[i]:
                if t not in V[j]:
                    continue
                # print 'hello'
                # print '{} / {}'.format(j,t)
                # print V[j][t]
                # import sys; sys.exit(1)


                if t not in old_to_new_track_mapping:
                    old_to_new_track_mapping[t] = [] # old track could map to multiple new tracks
                
                # if len(set(old_to_new_track_mapping[t]).intersection(V_prime[i].keys())) == 0:
                if t not in image_tracks[i]:
                    if t in image_tracks[j]:
                        current_track_id =  image_tracks[j][t]
                    else:
                        current_track_id = current_track_counter
                        current_track_counter += 1
                    # Create new track and add it to image i
                    old_to_new_track_mapping[t].append(current_track_id)
                    new_to_old_track_mapping[current_track_id] = t

                    image_tracks[i][t] = current_track_id

                    V_prime.add_nodes_from([current_track_id], bipartite=1)
                    V_prime.add_edge(i,current_track_id)
                    V_prime[i][current_track_id] = {
                        'feature_id': V[i][t]['feature_id'], 
                        'feature_color': V[i][t]['feature_color'], 
                        'feature': V[i][t]['feature']
                    }
                    

                if t not in image_tracks[j]:
                    new_track_id = image_tracks[i][t]
                    # Add track to image j
                    V_prime.add_edge(j,new_track_id)
                    V_prime[j][new_track_id] = {
                        'feature_id': V[j][t]['feature_id'], 
                        'feature_color': V[j][t]['feature_color'], 
                        'feature': V[j][t]['feature']
                    }
                    image_tracks[j][t] = new_track_id

            common_tracks_prime = len(set(V_prime[i].keys()).intersection(set(V_prime[j].keys())))
            cum_common_tracks_prime += common_tracks_prime
            # print ('\tCommon tracks: {}'.format(common_tracks_prime))

        
        # import sys; sys.exit(1)
    print ('Original tracks graph: {} edges  --  Regenerated tracks graph: {} edges'.format(len(V.edges()), len(V_prime.edges())))
    print ('Common Tracks: {} / {}'.format(cum_common_tracks, cum_common_tracks_prime))

    tracks_, images_ = matching.tracks_and_images(V)
    print ('{}_ / {}_'.format(len(images_), len(tracks_)))
    tracks_, images_ = matching.tracks_and_images(V_prime)
    print ('{}_ / {}_'.format(len(images_), len(tracks_)))
    # tracks_, images_ = nx.bipartite.sets(V)
    # print ('{}_ / {}_'.format(len(images_), len(tracks_)))
    # tracks_, images_ = nx.bipartite.sets(V_prime)
    # print ('{}_ / {}_'.format(len(images_), len(tracks_)))


    # for o in old_to_new_track_mapping:
    #     if len(old_to_new_track_mapping[o]) > 1:

    #         print '{} : {}'.format(o, len(old_to_new_track_mapping[o]))
    return V_prime
    # print 'Done'
        # continue
        # neighbors = G.neighbors(n1)


    # for i in remaining_images:
    #     continue

def network_construction(data, tracks_graph):
    if not data.iconic_image_list_exists() or not data.non_iconic_image_list_exists():
        iconic_images, non_iconic_images = scene_sampling_phase(data, tracks_graph)
        data.save_iconic_image_list(iconic_images)
        data.save_non_iconic_image_list(non_iconic_images)
    else:
        iconic_images, non_iconic_images = scene_sampling_phase(data, tracks_graph)
        data.save_iconic_image_list(iconic_images)
        data.save_non_iconic_image_list(non_iconic_images)

        # iconic_images = data.load_iconic_image_list()
        # non_iconic_images = data.load_non_iconic_image_list()

    # print ('Raj: uncommont scene_sampling_phase. Currently debugging!')
    # iconic_images = [ "DSC_0845.JPG", "DSC_0846.JPG", "DSC_0848.JPG", "DSC_0850.JPG", "DSC_0851.JPG", "DSC_0853.JPG", "DSC_0854.JPG", "DSC_0856.JPG", "DSC_0858.JPG", "DSC_0859.JPG", "DSC_0861.JPG", "DSC_0872.JPG", "DSC_0873.JPG", "DSC_0874.JPG", "DSC_0876.JPG", "DSC_0880.JPG", "DSC_0881.JPG", "DSC_0883.JPG"]
    # non_iconic_images = ["DSC_0844.JPG", "DSC_0879.JPG", "DSC_0857.JPG", "DSC_0852.JPG", "DSC_0860.JPG", "DSC_0862.JPG", "DSC_0871.JPG", "DSC_0875.JPG", "DSC_0847.JPG", "DSC_0877.JPG", "DSC_0849.JPG", "DSC_0855.JPG"]
    G = path_growth_phase(data, tracks_graph, iconic_images, non_iconic_images)
    return G, iconic_images, non_iconic_images

def path_growth_phase(data, tracks_graph, iconic_images, non_iconic_images):
    epsilon = 5
    draw_graph = True
    G = nx.Graph()
    H = nx.Graph()
    J = nx.Graph()
    G.add_nodes_from(iconic_images, bipartite=0)
    G.add_nodes_from(non_iconic_images, bipartite=1)
    # H.add_nodes_from(iconic_images)
    # J.add_nodes_from(iconic_images)

    T_A = iconic_set_points(data, tracks_graph, iconic_images)
    D = confusing_points(data, tracks_graph, iconic_images)
    U = T_A - D

    # T_A_non_iconic = iconic_set_points(data, tracks_graph, non_iconic_images)
    # D_non_iconic = confusing_points(data, tracks_graph, non_iconic_images)
    # U_non_iconic = T_A_non_iconic - D_non_iconic

    for c in iconic_images:
        for n in non_iconic_images:
            U_c = set(tracks_graph[c].keys()).intersection(U) # unique points for an image in the iconic set
            # common_unique_pts = U_c.intersection(tracks_graph[n].keys())
            # U_n = set(tracks_graph[n].keys()) # unique points for an image in the non-iconic set
            
            U_n = set(tracks_graph[n].keys()).intersection(U)
            # U_n = set(tracks_graph[n].keys()).intersection(U_non_iconic)
            # U_n = set(tracks_graph[n].keys())
            common_unique_pts = U_c.intersection(U_n)

            if c == 'P1010141.jpg' and n == 'P1010145.jpg' or c == 'P1010159.jpg' and n == 'P1010145.jpg':
            # if True:
                # print '#'*100
                # print common_unique_pts
                # print '='*100
                print '{}: {} - {}: {} => {}'.format(c, len(U_c), n, len(U_n), len(common_unique_pts))
                # p1, _, _ = data.load_features(c)
                # p2, _, _ = data.load_features(n)
                # im_rmatches = data.load_matches(c)
                # rmatches = im_rmatches[n]

                for cup in common_unique_pts:
                    # fid1 = tracks_graph[cup][c]['feature_id']
                    # fid2 = tracks_graph[cup][n]['feature_id']
                    # ri = np.where((rmatches[:,0] == fid1) & (rmatches[:,1] == fid2))[0]
                    print '\t {} - {}'.format(tracks_graph[cup][c], tracks_graph[cup][n])
                    # print '\t\t {} - {}  :  {}'.format(fid1, fid2, rmatches[ri])

                # print common_unique_pts
                
            if len(common_unique_pts) > epsilon:
                print '{} / {} : {}'.format(len(common_unique_pts), max(len(U_c), len(U_n)), 1.0 * len(common_unique_pts) / max(len(U_c), len(U_n)))
                # if 1.0 * len(common_unique_pts) / max(len(U_c), len(U_n)) >= 0.1:
                if True:
                    G.add_edge(c,n,weight=len(common_unique_pts)/100.0)

    # import sys; sys.exit(1)
    # # For debugging purposes only
    # for i in iconic_images:
    #     for j in iconic_images:
    #         if i <= j:
    #             continue
    #         if len(set(tracks_graph[i].keys()).intersection(set(tracks_graph[j].keys()))) > 0:
    #             H.add_edge(i, j, weight=0.2)

    # # For debugging purposes only
    # for i in iconic_images:
    #     for j in iconic_images:
    #         if i <= j:
    #             continue
    #         U_i = set(tracks_graph[i].keys()).intersection(U)
    #         U_j = set(tracks_graph[j].keys()).intersection(U)
    #         if len(U_i.intersection(U_j)) > 0:
    #             J.add_edge(i, j, weight=1.0)

    if draw_graph:
        pos = dict()
        pos.update( (n, (1, i)) for i, n in enumerate(iconic_images) )
        pos.update( (n, (2, i)) for i, n in enumerate(non_iconic_images) )
        opensfm.commands.formulate_graphs.draw_graph(G, filename=os.path.join(data.data_path,'yan/yan-path.png'), layout=pos)
        # opensfm.commands.formulate_graphs.draw_graph(H, filename=os.path.join(data.data_path,'yan/yan-iconic-images.png'), layout='spring')
        # opensfm.commands.formulate_graphs.draw_graph(J, filename=os.path.join(data.data_path,'yan/yan-iconic-images-intersections.png'), layout='spring')
    return G

def scene_sampling_phase(data, tracks_graph):
    alpha = 0.1
    C = {} # images in iconic set
    tracks, remaining_images = matching.tracks_and_images(tracks_graph)

    while True:
        R_i = {}
        delta = {}
        T_A_ = {}
        T_A = iconic_set_points(data, tracks_graph, C)
        D = confusing_points(data, tracks_graph, C)
        U = T_A - D
        completeness = len(T_A)
        distinctiveness = len(D)
        R = completeness - alpha * distinctiveness
        # print '='*100
        # print 'R: {} T_A: {} D: {}'.format(R, completeness, distinctiveness)
        # print 'C: {}'.format(C.keys())
        for i in remaining_images:
            C_i = C.copy()
            C_i[i] = True
            T_A_[i] = iconic_set_points(data, tracks_graph, C_i)
            D_i = confusing_points(data, tracks_graph, C_i)
            U_i = T_A_[i] - D_i
            completeness_i = len(T_A_[i])
            distinctiveness_i = len(D_i)
            R_i[i] = completeness_i - alpha * distinctiveness_i
            delta[i] = R_i[i] - R

        best_image = max(delta.iteritems(), key=operator.itemgetter(1))[0]

        # print '!'*100
        # print json.dumps(delta, sort_keys=True, indent=4, separators=(',', ': '))
        # print best_image
        # print '!'*100

        # if 1.0*len(T_A_[best_image])/len(tracks) < 0.6:
        if delta[best_image] > 0:
            C[best_image] = True
            remaining_images.remove(best_image)
            if len(remaining_images) == 0:
                break
        else:
            break
    
    # print '='*100    
    # print json.dumps(C, sort_keys=True, indent=4, separators=(',', ': '))
    # print '#'*100
    # print json.dumps(remaining_images, sort_keys=True, indent=4, separators=(',', ': '))
    return sorted(C.keys()), sorted(remaining_images)

def write_report(data, graph,
                 features_time, matches_time, tracks_time):
    tracks, images = matching.tracks_and_images(graph)
    image_graph = bipartite.weighted_projected_graph(graph, images)
    view_graph = []
    for im1 in data.images():
        for im2 in data.images():
            if im1 in image_graph and im2 in image_graph[im1]:
                weight = image_graph[im1][im2]['weight']
                view_graph.append((im1, im2, weight))

    report = {
        "wall_times": {
            "load_features": features_time,
            "load_matches": matches_time,
            "compute_tracks": tracks_time,
        },
        "wall_time": features_time + matches_time + tracks_time,
        "num_images": len(images),
        "num_tracks": len(tracks),
        "view_graph": view_graph
    }
    data.save_report(io.json_dumps(report), 'yan.json')
