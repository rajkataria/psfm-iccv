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

from multiprocessing import Pool

logger = logging.getLogger(__name__)


class Command:
    name = 'formulate_graphs'
    help = 'Formulate graphs based on a variety of similarity metrics'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    
    def graph_arguments(self, data, graphs, images, G_gt):
        args = []
        for criteria, label in graphs:
            args.append([data, images, criteria, label, G_gt])
        return args

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        images = data.images()
        processes = 4 #data.config['processes']
        edge_threshold = 0

        start = timer()
        # ground-truth match graph
        scores_gt = classifier.groundtruth_image_matching_results_adapter(data)
        G_gt = formulate_graph([data, images, scores_gt, 'gt', edge_threshold])
        auc_gt = calculate_graph_auc(G_gt, G_gt)

        # Criterias for match graph edges
        num_rmatches, num_rmatches_cost = classifier.rmatches_adapter(data)
        vt_rank_scores_mean, vt_rank_scores_min, vt_rank_scores_max, vt_scores_mean, vt_scores_min, vt_scores_max = classifier.vocab_tree_adapter(data)
        triplet_scores_counts, triplet_scores_cumsum = classifier.triplet_errors_adapter(data, options={'scores_gt': scores_gt})
        photometric_scores_counts, photometric_scores_cumsum = classifier.photometric_errors_adapter(data, options={'scores_gt': scores_gt})
        
        sequence_scores_mean, sequence_scores_min, sequence_scores_max, sequence_distance_scores = \
            data.sequence_rank_adapter()

        graphs = [
            [num_rmatches, 'rm'],
            [num_rmatches_cost, 'rm-cost'],
            [vt_rank_scores_min, 'vt-rank-min'],
            [vt_rank_scores_max, 'vt-rank-max'],
            [vt_rank_scores_mean, 'vt-rank-mean'],
            [vt_scores_min, 'vt-scores-min'],
            [vt_scores_max, 'vt-scores-max'],
            [vt_scores_mean, 'vt-scores-mean'],
            [triplet_scores_counts, 'te-counts'],
            [triplet_scores_cumsum, 'te-cumsum'],
            [photometric_scores_counts, 'pe-counts'],
            [photometric_scores_cumsum, 'pe-cumsum'],
            [sequence_scores_min, 'sq-min'],
            [sequence_scores_max, 'sq-max'],
            [sequence_scores_mean, 'sq-mean'],
            [sequence_distance_scores, 'sq-distance']
        ]

        results = {'gt': auc_gt}
        args = self.graph_arguments(data, graphs, images, G_gt)
        p = Pool(processes)
        if processes > 1:
            m_results = p.map(evaluate_metric, args)
        else:
            m_results = []
            for arg in args:
                m_results.append(evaluate_metric(arg))
        for r in m_results:
            results.update(r)

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

def evaluate_metric(arg):
    data, images, criteria, label, G_gt = arg
    results = {}
    edge_threshold = 0

    G = formulate_graph([data, images, criteria, label, edge_threshold])            
    auc = calculate_graph_auc(G, G_gt)
    if False:
        draw_graph(G, \
            filename=os.path.join(data.data_path,'graph-{}.png'.format(label.replace(' ','-'))), \
            highlighted_nodes=[], \
            layout='shell', \
            title='{} match graph'.format(label))

    results[label] = round(auc,2)
    data.save_graph(G, label, edge_threshold)
    return results

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
    return auc
    
def draw_graph(G, filename, highlighted_nodes=[], layout='spring', title=None):
    plt.clf()
    if layout == 'spring':
        pos=nx.spring_layout(G, iterations=200) # positions for all nodes
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = layout

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

    node_labels = {}
    for n in G.nodes():
        if 'lcc' in G.node[n] and 'pagerank' in G.node[n]:
            node_labels[n] = '{}\npr: {}\nlcc: {}'.format(n, round(G.node[n]['pagerank'], 2), round(G.node[n]['lcc'], 2))
        elif 'lcc' in G.node[n]:
            node_labels[n] = '{}\nlcc: {}'.format(n, round(G.node[n]['lcc'], 2))
        elif 'pagerank' in G.node[n]:
            node_labels[n] = '{}\npr: {}'.format(n, round(G.node[n]['pagerank'], 2))
        else:
            node_labels[n] = '{}'.format(n)


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

    nx.draw_networkx_nodes(G,pos,node_size=1000, alpha=alpha)
    nx.draw_networkx_nodes(G,pos,nodelist=highlighted_nodes, node_size=1000, alpha=1.0)

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

    nx.draw_networkx_labels(G,pos,node_labels, font_size=8,font_family='sans-serif')
    if title:
        plt.title(title)

    fig = plt.gcf()
    fig.set_size_inches(37, 21)
    plt.savefig(filename)

def formulate_graph(args):
    log.setup()

    data, images, scores, criteria, edge_threshold = args
    start = timer()
    G = nx.Graph()
    for i,img1 in enumerate(sorted(images)):
        for j,img2 in enumerate(sorted(images)):
            if j <= i:
                continue
            if img1 in scores and img2 in scores[img1]:
                if 'cost' in criteria:
                    if scores[img1][img2] <= edge_threshold:
                        G.add_edge(img1, img2, weight=scores[img1][img2])
                else:
                    if scores[img1][img2] >= edge_threshold:
                        G.add_edge(img1, img2, weight=scores[img1][img2])

    try:
        pagerank = nx.pagerank(G, alpha=0.9)
    except:
        pagerank = {}
        for n in G.nodes():
            pagerank[n] = 1.0
    lcc = nx.clustering(G, nodes=G.nodes(), weight='weight')

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

def threshold_graph_edges(G, threshold, key='weight'):
    G_thresholded = nx.Graph()
    for i,(n1,n2,d) in enumerate(G.edges(data=True)):
        if d[key] >= threshold:
            G_thresholded.add_edge(n1,n2)

    for n in G.nodes():
        if not G_thresholded.has_node(n):
            G_thresholded.add_node(n)

    try:
        pagerank = nx.pagerank(G_thresholded, alpha=0.9)
    except:
        pagerank = {}
        for n in G_thresholded.nodes():
            pagerank[n] = 1.0
    lcc = nx.clustering(G_thresholded, nodes=G_thresholded.nodes())

    for n in G_thresholded.nodes():
        G_thresholded.node[n]['pagerank'] = pagerank[n]
        G_thresholded.node[n]['lcc'] = lcc[n]

    return G_thresholded

def invert_graph(G):
    S = nx.Graph()
    edges = {}
    for n1,n2 in G.edges():
        if n1 not in edges:
            edges[n1] = []
        if n2 not in edges:
            edges[n2] = []
        node = '{}---{}'.format(n1,n2)
        S.add_node(node, weight=G.get_edge_data(n1,n2)['weight'])

    for i,en1 in enumerate(sorted(S.nodes())):
        n1,n2 = en1.split('---')
        for j,en2 in enumerate(sorted(S.nodes())):
            if j <= i:
                continue
            n1_,n2_ = en2.split('---')
            common_nodes = list(set([n1,n2]).intersection([n1_,n2_]))
            if len(common_nodes) > 0:
                common_node = common_nodes[0]
                S.add_edge(en1, en2, weight=1.0, \
                    label=common_node, \
                    pagerank=G.node[common_node]['pagerank'] if 'pagerank' in G.node[common_node] else '-', \
                    lcc=G.node[common_node]['lcc'] if 'lcc' in G.node[common_node] else '-'
                    )
    return S
