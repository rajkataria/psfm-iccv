
from . import extract_metadata
from . import detect_features
from . import evaluate_vt_rankings
from . import match_features
from . import calculate_features
from . import classify_images
from . import create_tracks
from . import create_tracks_classifier
from . import reconstruct
from . import convert_colmap
from . import validate_results
from . import mesh
from . import undistort
from . import compute_depthmaps
from . import export_ply
from . import export_openmvs
from . import export_visualsfm
from . import export_geocoords
from . import create_submodels
from . import align_submodels


opensfm_commands = [
    extract_metadata,
    detect_features,
    evaluate_vt_rankings,
    match_features,
    calculate_features,
    classify_images,
    create_tracks,
    create_tracks_classifier,
    reconstruct,
    convert_colmap,
    validate_results,
    mesh,
    undistort,
    compute_depthmaps,
    export_ply,
    export_openmvs,
    export_visualsfm,
    export_geocoords,
    create_submodels,
    align_submodels,
]
