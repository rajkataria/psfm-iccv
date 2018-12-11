import json
import logging
import math
import numpy as np
import os
# import pyquaternion
import sys
from timeit import default_timer as timer

# from networkx.algorithms import bipartite

from opensfm import dataset
# from opensfm import evaluate_ate_scale, associate
from opensfm import io
# from opensfm import matching
# from opensfm import types
# from pyquaternion import Quaternion

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
# from matplotlib.patches import Ellipse

logger = logging.getLogger(__name__)

class Command:
    name = 'convert_colmap'
    help = "Convert colmap output(NVM) to reconstruction"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        colmap_fn = os.path.join(data.data_path, 'sparse_converted', 'colmap-output.nvm')
        reconstruction_fn = os.path.join(data.data_path, 'reconstruction_colmap.json')
        logger.info('Converting "{}" to "{}":'.format(os.path.basename(colmap_fn), os.path.basename(reconstruction_fn)))
        io.reconstruction_from_nvm(data.data_path, colmap_fn, reconstruction_fn)
        



