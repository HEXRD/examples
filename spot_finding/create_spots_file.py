from pathlib import Path

import h5py

from hexrd import imageseries

from spot_finder import SpotFinder
from write_spots import write_spots

# These are the user-modifiable settings
settings = {
    'threshold': 25,
    'min_area': 2,
    'max_area': 1000,
}

root_dir = Path(__file__).parent.parent.parent.parent
images_dir = root_dir / 'NIST_ruby/multiruby_dexelas/imageseries'

# Filenames
image_files = {
    'ff1': images_dir / 'mruby-0129_000004_ff1_000012-cachefile.npz',
    'ff2': images_dir / 'mruby-0129_000004_ff2_000012-cachefile.npz',
}
spots_filename = Path('spots_file.h5')

# Load in the raw imageseries for each Dexela detector
raw_ims_dict = {}
for k, filename in image_files.items():
    raw_ims_dict[k] = imageseries.open(filename, 'frame-cache')

# Create and write the spots file
finder = SpotFinder(**settings)
with h5py.File(spots_filename, 'w') as f:
    write_spots(raw_ims_dict, finder, f)
