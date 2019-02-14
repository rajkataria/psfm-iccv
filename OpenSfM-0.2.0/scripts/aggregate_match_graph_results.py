import csv
import datetime
import glob
import json
import os
import sys

from argparse import ArgumentParser

def get_results(root_directory, experiments):
    results = {}
    metadata = {}
    datasets = glob.glob(root_directory + '/*/*/')
    for d in datasets:
        dataset_name = d.split('/')[-3]
        sequence_name = d.split('/')[-2]
        results_folder = os.path.join(d, 'results')
        match_graph_results_fn = '{}/match_graph_results.json'.format(results_folder)
        # reconstruction_results_fn = '{}/reconstruction_results.json'.format(results_folder)

        if os.path.exists(match_graph_results_fn):
            with open(match_graph_results_fn, 'r') as fin:
                data = json.load(fin)
                for exp in experiments:
                    if exp not in data:
                        print ('\tDataset: "{}" missing results for experiment: "{}"'.format(d.split('/')[-2], exp))
                        continue
                    # sequence_name = data[exp]['trajectory']
                    r_key = '{}---{}'.format(dataset_name, sequence_name)
                    if r_key not in results:
                        results[r_key] = {}    
                        metadata[r_key] = {
                            # 'images': data[exp]["total images %f"],
                            'dataset': dataset_name
                        }
                    results[r_key][exp] = {
                        'sequence': sequence_name,
                        'AUC': data[exp]
                        }
    return metadata, results

def output_csv(metadata, experiments, results):
    with open('results-match-graphs.csv', mode='w') as csv_file:
        fieldnames = ['Dataset', 'Sequence']
        # for key in sorted([experiments[e]['key'] for e in experiments]):
        for e in sorted(experiments.keys()):
            fieldnames.extend([
                '{}'.format(e)
            ])

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r_key in sorted(results.keys()):
            sequence_name = r_key.split('---')[-1]
            row = {
                'Dataset': str(metadata[r_key]['dataset']),
                'Sequence': str(sequence_name),
                # 'Images': str(metadata[r_key]['images'])
            }

            for exp in results[r_key]:
                row.update({
                    # 'AUC - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['AUC'])
                    '{}'.format(exp): str(results[r_key][exp]['AUC'])
                })
            writer.writerow(row)

        descriptions = {}
        for exp in experiments:
            descriptions[exp] = experiments[exp]['desc']
        writer.writerow({'Dataset': json.dumps(descriptions, sort_keys=True, indent=4, separators=(',', ': '))})

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-r', '--root_directory', help='')
  
    parser_options = parser.parse_args()

    experiments = {
        'gt': \
            {'key': 0, 'desc': 'Ground-truth match graph'},
        'rm': \
            {'key': 1, 'desc': 'Robust matches as edge weights (baseline)'},
        'sq-distance': \
            {'key': 2, 'desc': 'Sequence distances as edge weights'},
        'sq-max': \
            {'key': 3, 'desc': 'Max of the sequence ranks as edge weights: max(rank(im1,im2), rank(im2,im1))'},
        'sq-mean': \
            {'key': 4, 'desc': 'Mean of the sequence ranks as edge weights: mean(rank(im1,im2), rank(im2,im1))'},
        'sq-min': \
            {'key': 5, 'desc': 'Min of the sequence ranks as edge weights: min(rank(im1,im2), rank(im2,im1))'},
        'te': \
            {'key': 6, 'desc': 'Triplet scores as edge weights: triplet_score = sum(cumsum(triplet error histogram))'},
        'vt-rank-max': \
            {'key': 7, 'desc': 'Max of the vocab tree ranks as edge weights: max(rank(im1,im2), rank(im2,im1))'},
        'vt-rank-mean': \
            {'key': 8, 'desc': 'Mean of the vocab tree ranks as edge weights: mean(rank(im1,im2), rank(im2,im1))'},
        'vt-rank-min': \
            {'key': 9, 'desc': 'Min of the vocab tree ranks as edge weights: min(rank(im1,im2), rank(im2,im1))'},
        'vt-scores-max': \
            {'key': 10, 'desc': 'Max of the vocab tree scores as edge weights: max(score(im1,im2), score(im2,im1))'},
        'vt-scores-mean': \
            {'key': 11, 'desc': 'Mean of the vocab tree scores as edge weights: mean(score(im1,im2), score(im2,im1))'},
        'vt-scores-min': \
            {'key': 12, 'desc': 'Min of the vocab tree scores as edge weights: min(score(im1,im2), score(im2,im1))'}
    }
    metadata, results = get_results(parser_options.root_directory, experiments)
    output_csv(metadata, experiments, results)

if __name__ == '__main__':
    main(sys.argv)
