import logging
import subprocess

from timeit import default_timer as timer
from subprocess import call

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import sklearn

from opensfm import dataset
from opensfm import classifier
from opensfm import features
from opensfm import io
from opensfm import log
from opensfm.context import parallel_map

logger = logging.getLogger(__name__)


class Command:
    name = 'formulate_graphs'
    help = 'Formulate graphs based on a variety of similarity metrics'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        images = data.images()

        start = timer()

        # ground-truth match graph
        scores_gt = classifier.groundtruth_image_matching_results_adapter(data)
        G_gt = formulate_graph([data, images, scores_gt, 'gt'])
        auc_gt = calculate_graph_auc(G_gt, G_gt)

        # Criterias for match graph edges
        num_rmatches = classifier.rmatches_adapter(data)
        vt_scores_mean, vt_scores_min, vt_scores_max = classifier.vocab_tree_adapter(data)
        triplet_scores = classifier.triplet_errors_adapter(data, options={'scores_gt': scores_gt})
        sequence_scores_mean, sequence_scores_min, sequence_scores_max = classifier.sequence_rank_adapter(data)

        graphs = [
            [num_rmatches, 'rm'],
            [vt_scores_min, 'vt-min'],
            [vt_scores_max, 'vt-max'],
            [vt_scores_mean, 'vt-mean'],
            [triplet_scores, 'te'],
            [sequence_scores_min, 'sq-min'],
            [sequence_scores_max, 'sq-max'],
            [sequence_scores_mean, 'sq-mean'],
        ]

        results = {'gt': auc_gt}

        for criteria, label in graphs:
            G = formulate_graph([data, images, criteria, label])            
            auc = calculate_graph_auc(G, G_gt)
            draw_graph(G, \
                filename=os.path.join(data.data_path,'graph-{}.png'.format(label.replace(' ','-'))), \
                highlighted_nodes=[], \
                layout='shell', \
                title='{} match graph'.format(label))

            results[label] = round(auc,2)

        print (json.dumps(results, sort_keys=True, indent=4, separators=(',', ': ')))
        data.save_match_graph_results(results)

        end = timer()
        with open(data.profile_log(), 'a') as fout:
            fout.write('formulate_graphs: {0}\n'.format(end - start))

        self.write_report(data, end - start)

    def write_report(self, data, wall_time):
        report = {
            "wall_time": wall_time
        }
        data.save_report(io.json_dumps(report), 'similarity-graphs.json')

def calculate_graph_auc(G, G_gt):
    scores_gt = []
    scores = []
    for i,im1 in enumerate(sorted(G_gt.nodes())):
        for j,im2 in enumerate(sorted(G_gt.nodes())):
            if j <= i:
                continue
            if G_gt.has_edge(im1,im2):
                scores_gt.append(1.0)
            else:
                scores_gt.append(0.0)

            if G.has_edge(im1,im2):
                scores.append(G.get_edge_data(im1,im2)['weight'])
            else:
                scores.append(0.0)

    precision, recall, threshs = sklearn.metrics.precision_recall_curve(scores_gt, scores)
    auc = sklearn.metrics.average_precision_score(scores_gt, scores)
    # plt.step(recall, precision, color=color, alpha=0.2 * width,
    #     where='post')
    # print auc
    return auc
    
def draw_graph(G, filename, highlighted_nodes=[], layout='shell', title=None):
    plt.clf()
    if layout == 'spring':
        pos=nx.spring_layout(G, iterations=70) # positions for all nodes
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.shell_layout(G)

    highlighted_exlarge = [(u, v) for (u, v, d) in G.edges(data=True) if u in highlighted_nodes and v in highlighted_nodes and d['weight'] >= 0.8]
    highlighted_elarge = [(u, v) for (u, v, d) in G.edges(data=True) if u in highlighted_nodes and v in highlighted_nodes and d['weight'] >= 0.6 and d['weight'] < 0.8]
    highlighted_emedium = [(u, v) for (u, v, d) in G.edges(data=True) if u in highlighted_nodes and v in highlighted_nodes and d['weight'] >= 0.4 and d['weight'] < 0.6]
    highlighted_esmall = [(u, v) for (u, v, d) in G.edges(data=True) if u in highlighted_nodes and v in highlighted_nodes and d['weight'] >= 0.2 and d['weight'] < 0.4]
    highlighted_exsmall = [(u, v) for (u, v, d) in G.edges(data=True) if u in highlighted_nodes and v in highlighted_nodes and d['weight'] >= 0.0 and d['weight'] < 0.2]

    exlarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.8]
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.6 and d['weight'] < 0.8]
    emedium = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.4 and d['weight'] < 0.6]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.2 and d['weight'] < 0.4]
    exsmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.0 and d['weight'] < 0.2]

    # if pagerank is not None:
    # pageranks = nx.get_node_attributes(G,'pagerank')
    node_labels = {}
    for n in G.nodes():
        # if len(highlighted_nodes) > 0 and (n1 not in highlighted_nodes or n2 not in highlighted_nodes):
        #     continue
        # print '#'*100
        # print G[n]
        # print '='*100
        node_labels[n] = '{}\npr: {}\nlcc: {}'.format(n, round(G.node[n]['pagerank'], 2), round(G.node[n]['lcc'], 2))
    # print node_labels
    

    weights = nx.get_edge_attributes(G,'weight')
    edge_weights = {}
    for n1,n2 in weights:
        if len(highlighted_nodes) > 0 and (n1 not in highlighted_nodes or n2 not in highlighted_nodes):
            continue
        edge_weights[(n1,n2)] = round(weights[(n1,n2)], 2)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_weights)
    
    if len(highlighted_nodes) == 0:
        alpha = 1.0
    else:
        alpha = 0.10

    nx.draw_networkx_nodes(G,pos,node_size=4000, alpha=alpha)
    nx.draw_networkx_nodes(G,pos,nodelist=highlighted_nodes, node_size=4000, alpha=1.0)

    edgetypes = {
        'highlighted_exlarge': {'edges': highlighted_exlarge, 'color': 'green', 'width': 13, 'alpha': 1.0},
        'highlighted_elarge': {'edges': highlighted_elarge, 'color': 'yellow', 'width': 10, 'alpha': 1.0},
        'highlighted_emedium': {'edges': highlighted_emedium, 'color': 'purple', 'width': 7, 'alpha': 1.0},
        'highlighted_esmall': {'edges': highlighted_esmall, 'color': 'orange', 'width': 4, 'alpha': 1.0},
        'highlighted_exsmall': {'edges': highlighted_exsmall, 'color': 'red', 'width': 1, 'alpha': 1.0},
        'exlarge': {'edges': exlarge, 'color': 'green', 'width': 13, 'alpha': alpha},
        'elarge': {'edges': elarge, 'color': 'yellow', 'width': 10, 'alpha': alpha},
        'emedium': {'edges': emedium, 'color': 'purple', 'width': 7, 'alpha': alpha},
        'esmall': {'edges': esmall, 'color': 'orange', 'width': 4, 'alpha': alpha},
        'exsmall': {'edges': exsmall, 'color': 'red', 'width': 1, 'alpha': alpha}
    }
    for k in edgetypes.keys():
        nx.draw_networkx_edges(G, pos, \
            edgelist=edgetypes[k]['edges'], \
            width=edgetypes[k]['width'], \
            edge_color=edgetypes[k]['color'], \
            alpha=edgetypes[k]['alpha'], \
            with_labels=False
        )

    # nx.draw_networkx_labels(G,pos,font_size=8,font_family='sans-serif')
    nx.draw_networkx_labels(G,pos,node_labels, font_size=8,font_family='sans-serif')
    if title:
        plt.title(title)

    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(filename)

def formulate_graph(args):
    log.setup()

    data, images, scores, criteria = args
    start = timer()
    G = nx.Graph()
    for i,img1 in enumerate(sorted(images)):
        for j,img2 in enumerate(sorted(images)):
            if j <= i:
                continue
            if img1 in scores and img2 in scores[img1]:
                G.add_edge(img1, img2, weight=scores[img1][img2])

    pagerank = nx.pagerank(G, alpha=0.9)
    lcc = nx.clustering(G, nodes=G.nodes())
    for n in G.nodes():
        G.node[n]['pagerank'] = pagerank[n]
        G.node[n]['lcc'] = lcc[n]

    end = timer()
    report = {
        "wall_time": end - start,
    }
    data.save_report(io.json_dumps(report),
                     'similarity-graphs.json')
    return G

