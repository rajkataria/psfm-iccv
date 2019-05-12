# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

# This script is based on an original implementation by True Price.

import sys
import sqlite3
import numpy as np
import os
import argparse
import shutil


IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def image_id_to_name(self, image_id):
        cursor = self.execute(
            "SELECT name FROM images WHERE image_id=?", (image_id,))
        image_name = (cursor.fetchone())[0]
        return image_name

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))


def matches_osfm_to_colmap(database_path, threshold=0.5):
    # copy the original database to the new one and start modifying the new database
    database_orig = os.path.join(database_path, 'database.db')
    database = os.path.join(database_path, 'database-{}.db'.format(threshold))
    shutil.copy(database_orig, database)
    db = COLMAPDatabase.connect(database)

    cameras = {}
    data = dataset.DataSet(database_path)
    image_matching_results = data.load_image_matching_results()

    camera_rows = db.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras")
    rmatches_rows = db.execute("SELECT pair_id, rows, cols, data FROM inlier_matches")

    for camera_id, model, width, height, params, prior_focal_length in camera_rows:
        cameras[camera_id] = {
            'image': db.image_id_to_name(camera_id),
            'width': width,
            'height': height,
            'params': blob_to_array(params, np.float64),
            'prior_focal_length': prior_focal_length
        }

    for pair_id, rows, cols, m in rmatches_rows:
        im1_id, im2_id = pair_id_to_image_ids(pair_id)
        im1 = cameras[im1_id]['image']
        im2 = cameras[im2_id]['image']

        if rows > 0:
            if image_matching_results[im1][im2]['score'] < threshold:
                print ('DELETING... {} : {} / {} -- {}  |  score: {}'.format(pair_id, im1, im2, rows, image_matching_results[im1][im2]['score']))
                db.execute('DELETE from inlier_matches where pair_id=?', (pair_id,))

    db.commit()
    db.close()

def matches_colmap_to_osfm(database_path):
    data = dataset.DataSet(database_path)
    database = os.path.join(database_path, 'database.db')
    db = COLMAPDatabase.connect(database)
    cameras = {}
    im_features = {}
    im_matches = {}
    im_rmatches = {}

    camera_rows = db.execute("SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras")
    matches_rows = db.execute("SELECT pair_id, rows, cols, data FROM matches")
    rmatches_rows = db.execute("SELECT pair_id, rows, cols, data FROM inlier_matches")
    keypoint_rows = db.execute("SELECT image_id, rows, cols, data FROM keypoints")
    descriptors_rows = db.execute("SELECT image_id, rows, cols, data FROM descriptors")


    for camera_id, model, width, height, params, prior_focal_length in camera_rows:
        cameras[camera_id] = {
            'image': db.image_id_to_name(camera_id),
            'width': width,
            'height': height,
            'params': blob_to_array(params, np.float64),
            'prior_focal_length': prior_focal_length
        }

    for image_id, rows, cols, k in keypoint_rows:
        im = cameras[image_id]['image']
        width = cameras[image_id]['width']
        height = cameras[image_id]['height']

        keypoints = blob_to_array(k, np.float32, (-1, cols))
        im_features[im] = {
            'points': keypoints,
            'desc': None,
            'colors': None,
            'width': width,
            'height': height
            }
    for image_id, rows, cols, d in descriptors_rows:
        im = cameras[image_id]['image']
        im_features[im]['desc'] = blob_to_array(d, np.uint8, (-1, cols))

    for pair_id, rows, cols, m in matches_rows:
        im1_id, im2_id = pair_id_to_image_ids(pair_id)
        im1 = cameras[im1_id]['image']
        im2 = cameras[im2_id]['image']

        if im1 not in im_matches:
            im_matches[im1] = {}

        if m:
            matches = blob_to_array(m, np.uint32, (-1, 2))
        else:
            matches = np.array([])
        im_matches[im1][im2] = matches
    
    for pair_id, rows, cols, m in rmatches_rows:
        im1_id, im2_id = pair_id_to_image_ids(pair_id)
        im1 = cameras[im1_id]['image']
        im2 = cameras[im2_id]['image']

        if im1 not in im_rmatches:
            im_rmatches[im1] = {}

        if m:
            rmatches = blob_to_array(m, np.uint32, (-1, 2))
        else:
            rmatches = np.array([])
        im_rmatches[im1][im2] = rmatches

    for im in im_features:
        points, _, _ = features.mask_and_normalize_features(im_features[im]['points'], im_features[im]['desc'], im_features[im]['colors'], im_features[im]['width'], im_features[im]['height'], mask=None)
        im_features[im]['normalized_points'] = points

    for im in im_features:
        data.save_features(im, im_features[im]['points'], im_features[im]['desc'], im_features[im]['colors'])
        if im not in im_matches:
            im_matches[im] = {}
        if im not in im_rmatches:
            im_rmatches[im] = {}

        data.save_matches(im, im_rmatches[im])
        data.save_all_matches(im, im_matches[im], None, im_rmatches[im])
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    parser.add_argument('-l', '--opensfm_path', help='opensfm path')
    args = parser.parse_args()
    
    if not args.opensfm_path in sys.path:
        sys.path.insert(1, args.opensfm_path)
    from opensfm import features, dataset, matching, classifier, reconstruction, types, io
    global features
    # example_usage()
    # matches_colmap_to_osfm(args.database_path)
    matches_osfm_to_colmap(args.database_path)
