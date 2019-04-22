import json
import logging
import math
import numpy as np
import os
import sys
from timeit import default_timer as timer

from opensfm import dataset
from opensfm import io

logger = logging.getLogger(__name__)

class Command:
    name = 'convert_nvm'
    help = "Convert NVMs (colmap/theia) to reconstruction"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        colmap_fn = os.path.join(data.data_path, 'sparse_converted', 'colmap-output.nvm')
        theia_fn = os.path.join(data.data_path, 'theia', 'theia-output.nvm')
        for fn, label in [[colmap_fn, 'colmap'], [theia_fn, 'theia']]:
            reconstruction_fn = os.path.join(data.data_path, 'reconstruction_{}.json'.format(label))
            logger.info('Converting "{}" to "{}":'.format(os.path.basename(fn), os.path.basename(reconstruction_fn)))
            io.reconstruction_from_nvm(data.data_path, fn, reconstruction_fn)
        



