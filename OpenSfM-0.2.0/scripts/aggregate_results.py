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
        results_folder = os.path.join(d, 'results')
        ate_results_fn = '{}/ate_results.json'.format(results_folder)
        reconstruction_results_fn = '{}/reconstruction_results.json'.format(results_folder)

        if os.path.exists(ate_results_fn):
            with open(ate_results_fn, 'r') as fin:
                data = json.load(fin)
                for exp in experiments:
                    sequence_name = data[exp]['trajectory']
                    r_key = '{}---{}'.format(dataset_name, sequence_name)
                    if r_key not in results:
                        results[r_key] = {}    
                        metadata[r_key] = {
                            'images': data[exp]["total images %f"],
                            'dataset': dataset_name
                        }
                    results[r_key][exp] = {
                        'sequence': sequence_name,
                        'ATE dm': data[exp]['images aligned %f dm'],
                        'ATE cm': data[exp]['images aligned %f cm'],
                        'ATE m': data[exp]['images aligned %f m'],
                        'images': data[exp]['total images %f'],
                        }
        
        if os.path.exists(reconstruction_results_fn):
            with open(reconstruction_results_fn, 'r') as fin:
                data = json.load(fin)
                for exp in experiments:
                    sequence_name = data[exp]['dataset']
                    r_key = '{}---{}'.format(dataset_name, sequence_name)
                    if r_key not in results:
                        results[r_key] = {}
                    if exp not in results[r_key]:
                         results[r_key][exp] = {}

                    results[r_key][exp].update({ \
                        'points triangulated': data[exp]['points triangulated '],
                        'registered images': data[exp]['registered images'],
                        'time': data[exp]['time']
                        })

    return metadata, results

def output_csv(metadata, experiments, results):
    with open('results.csv', mode='w') as csv_file:
        fieldnames = ['Dataset', 'Sequence', 'Images']
        for key in sorted([experiments[e]['key'] for e in experiments]):
            fieldnames.extend([
                'Cameras Registered - {}'.format(key), \
                'Points Triangulated - {}'.format(key), \
                'ATE (cm) - {}'.format(key), \
                'ATE (dm) - {}'.format(key), \
                'ATE (m) - {}'.format(key), \
                'Time - {}'.format(key), \
                'Visual Inspection - {}'.format(key)
            ])

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r_key in sorted(results.keys()):
            sequence_name = r_key.split('---')[-1]
            row = {
                'Dataset': str(metadata[r_key]['dataset']),
                'Sequence': str(sequence_name),
                'Images': str(metadata[r_key]['images'])
            }

            for exp in results[r_key]:
                row.update({
                    'Cameras Registered - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['registered images']),
                    'Points Triangulated - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['points triangulated']),
                    'ATE (cm) - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['ATE cm']),
                    'ATE (dm) - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['ATE dm']),
                    'ATE (m) - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['ATE m']),
                    'Time - {}'.format(experiments[exp]['key']): str(results[r_key][exp]['time']),
                    'Visual Inspection - {}'.format(experiments[exp]['key']): str("-")
                })
            writer.writerow(row)

        descriptions = {}
        for exp in experiments:
            descriptions[experiments[exp]['key']] = experiments[exp]['desc']
        writer.writerow({'Dataset': json.dumps(descriptions, sort_keys=True, indent=4, separators=(',', ': '))})

def main(argv):
    parser = ArgumentParser(description='')
    parser.add_argument('-r', '--root_directory', help='')
  
    parser_options = parser.parse_args()

    experiments = {
        'baseline': {'key': 0, 'desc': 'OpenSfM baseline'},
        'colmap': {'key': 1, 'desc': 'Colmap baseline'},
        'imc-True-wr-False-gm-False-wfm-False-imt-False': {'key': 2, 'desc': ''},
        'imc-True-wr-False-gm-False-wfm-True-imt-False': {'key': 3, 'desc': ''},
        'imc-True-wr-False-gm-True-wfm-False-imt-False': {'key': 4, 'desc': ''},
        'imc-True-wr-True-gm-False-wfm-False-imt-False': {'key': 5, 'desc': ''},
        'imc-True-wr-True-gm-False-wfm-True-imt-False': {'key': 6, 'desc': ''},
        'imc-True-wr-True-gm-True-wfm-False-imt-False': {'key': 7, 'desc': ''}
    }
    metadata, results = get_results(parser_options.root_directory, experiments)
    output_csv(metadata, experiments, results)

if __name__ == '__main__':
    main(sys.argv)
