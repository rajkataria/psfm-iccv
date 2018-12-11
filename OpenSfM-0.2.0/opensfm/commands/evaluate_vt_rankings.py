import logging
import subprocess

from timeit import default_timer as timer
from subprocess import call

import numpy as np

from opensfm import dataset
from opensfm import features
from opensfm import io
from opensfm import log
from opensfm.context import parallel_map

logger = logging.getLogger(__name__)


class Command:
    name = 'evaluate_vt_rankings'
    help = 'Compute vocabulary tree rankings for all images'

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        images = data.images()

        arguments = [images, data]
        start = timer()
        vt_rankings(arguments)
        end = timer()
        with open(data.profile_log(), 'a') as fout:
            fout.write('evaluate_vt_rankings: {0}\n'.format(end - start))

        self.write_report(data, end - start)

    def write_report(self, data, wall_time):
        report = {
            "wall_time": wall_time
        }
        data.save_report(io.json_dumps(report), 'rankings.json')

def vt_rankings(args):
    log.setup()

    images, data = args
    libvot = data.config['libvot']

    start = timer()
    subprocess.Popen("ls -d {}/images/* > {}/vt_image_list.txt".format(data.data_path, data.data_path), shell=True, stdout=subprocess.PIPE).stdout.read()
    subprocess.Popen("{}/build/bin/libvot_feature -output_folder {}/sift/ {}/vt_image_list.txt".format(libvot, data.data_path, data.data_path), shell=True, stdout=subprocess.PIPE).stdout.read()
    subprocess.Popen("ls -d {}/sift/*.sift > {}/vt_sift_list.txt".format(data.data_path, data.data_path), shell=True, stdout=subprocess.PIPE).stdout.read()
    subprocess.Popen("{}/build/bin/image_search {}/vt_sift_list.txt {}/vocab_out".format(libvot, data.data_path, data.data_path), shell=True, stdout=subprocess.PIPE).stdout.read()

    end = timer()
    report = {
        "wall_time": end - start,
    }
    data.save_report(io.json_dumps(report),
                     'rankings.json')
