# %%
import numpy as np
import matplotlib.pyplot as plt

# import lumicks
import lumicks.pylake as lk

# %matplotlib inline
from skimage.transform import rescale
import tifffile
import os
from cv2 import warpAffine, invertAffineTransform
from pathlib import Path
from cv2 import estimateAffine2D, estimateAffinePartial2D
from picasso import io
import json
from json import JSONEncoder
import argparse
import itertools
import math
import subprocess
import shutil


def norm_image(image, inverse=False):
    amin = image.min()
    amax = image.max()
    if inverse:
        return 1 - (image - amin) / (amax - amin)
    else:
        return (image - amin) / (amax - amin)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# %%
parser = argparse.ArgumentParser(
    description="""Scripts to align various channels based on reference beads.""",
    epilog="""""",
)
parser.add_argument("wt_file", help="WT tif file")
parser.add_argument("bf_file", help="BF tif file")

parser.add_argument(
    "-o",
    "--output-directory",
    default="output",
    help="Output directory. Default=output/",
)

parser.add_argument(
    "-f", "--fit-method", default="lq", help="Fit method for picasso.  Default=lq"
)
parser.add_argument(
    "-b", "--box-size", default=21, help="Box sized for picasso. Default=21"
)
parser.add_argument(
    "-g",
    "--min-gradient",
    default=70000,
    help="Minimum gradient for picasso. Default=70000",
)
parser.add_argument(
    "-e",
    "--max-pos-error",
    default=3.5,
    help="Maximum standard dev accepted for x and y position of spots. Default=3.5",
)
parser.add_argument("-p", "--max_photons", help="Maximum number of photons for spots.")

parser.add_argument(
    "-d",
    "--delete-temp-files",
    default=True,
    help="Delete temporary files made by picasso",
)

parser.add_argument(
    "-x1",
    help="X lower limit to consider in brightfield",
)

parser.add_argument(
    "-x2",
    help="X upper limit to consider in brightfield",
)

parser.add_argument(
    "-y1",
    help="Y lower limit to consider in brightfield",
)

parser.add_argument(
    "-y2",
    help="Y upper limit to consider in brightfield",
)

args = parser.parse_args()

bf_path = args.bf_file
wt_path = args.wt_file
output_path = (
    args.output_directory + "/"
)  # The trailing slash is in case it wasn't added by the user


# %%

# Check if the directory already exists
os.makedirs(output_path, exist_ok=True)
# %%

# Copy input files to output folder if calculating new matrix
shutil.copy2(bf_path, output_path + os.path.basename(bf_path))
shutil.copy2(wt_path, output_path + os.path.basename(wt_path))

bf_path = output_path + os.path.basename(bf_path)
wt_path = output_path + os.path.basename(wt_path)

# rename tiff to tif files
if bf_path.endswith(".tiff"):
    print(bf_path)
    os.rename(bf_path, bf_path[:-1])
    bf_path = bf_path[:-1]
if wt_path.endswith(".tiff"):
    print(wt_path)
    os.rename(wt_path, wt_path[:-1])
    wt_path = wt_path[:-1]

# %%
bf = lk.ImageStack(bf_path)  # Loading a stack.
wt = lk.ImageStack(wt_path)  # Loading a stack.

wt.export_tiff(
    output_path + Path(wt_path).stem + "_aligned.tif"
)  # Save aligned wt stack


# %%

# Get channels
wt_g = wt.get_image(channel="green")
# wt_r = wt.get_image(channel='red')  #not really used
# wt_b = wt.get_image(channel='blue') #not really used
bf_g = bf.get_image()
bf_g=np.swapaxes(bf_g,1,2)
bf_g=np.swapaxes(bf_g,0,1)
bf_g=bf_g[1]

# %%
# Get region of interest data

wt_metadata = wt._tiff_image_metadata()
wt_roi = wt_metadata["Alignment region of interest (x, y, width, height)"]

bf_metadata = bf._tiff_image_metadata()
bf_roi = bf_metadata[
    "Region of interest (x, y, width, height)"
]  # This is different because the wt was previously aligned I think. Can this cause issues?

# %%
# Padding is CANCELED. Once this is all working flawlessly I should fix the code to remove references to padding

padded_wt_filename = Path(wt_path).stem + "_padded.tif"
# wt_g_padded = np.pad(wt_g, [(int(wt_roi[0]), 0), (int(wt_roi[1]), 0)])
wt_g_padded = wt_g
tifffile.imwrite(output_path + padded_wt_filename, wt_g_padded)


padded_bf_filename = Path(bf_path).stem + "_padded.tif"
# bf_g_padded = np.pad(bf_g, [(int(bf_roi[0]), 0), (int(bf_roi[1]), 0)])
bf_g_padded = bf_g
tifffile.imwrite(output_path + padded_bf_filename, bf_g_padded)

# %%

transform_mat = []  # set to empty to check afterwards if I have a matrix

# if matrix wasnt provided, calculate it
run_string = (
    "python -m picasso localize "
    + output_path
    + padded_wt_filename
    + " --fit-method "
    + args.fit_method
    + " -b "
    + str(args.box_size)
    + " --gradient "
    + str(args.min_gradient)
)
subprocess.run(run_string)

run_string = (
    "python -m picasso localize "
    + output_path
    + padded_bf_filename
    + " --fit-method "
    + args.fit_method
    + " -b "
    + str(args.box_size)
    + " --gradient 500"   #Hardcoded

)
subprocess.run(run_string)

# %%

bf_locs_path = output_path + Path(padded_bf_filename).stem + "_locs.hdf5"
bf_locs, bf_info = io.load_locs(bf_locs_path)
wt_locs_path = output_path + Path(padded_wt_filename).stem + "_locs.hdf5"
wt_locs, wt_info = io.load_locs(wt_locs_path)

# %%
wt_locs = wt_locs[wt_locs["sx"] < args.max_pos_error]
wt_locs = wt_locs[wt_locs["sy"] < args.max_pos_error]
bf_locs = bf_locs[bf_locs["sx"] < args.max_pos_error]
bf_locs = bf_locs[bf_locs["sy"] < args.max_pos_error]

if args.max_photons:
    wt_locs = wt_locs[wt_locs["photons"] < args.max_photons]
    bf_locs = bf_locs[bf_locs["photons"] < args.max_photons]

if args.x1:
    bf_locs = bf_locs[bf_locs["x"] > int(args.x1)]
if args.x2:
    bf_locs = bf_locs[bf_locs["x"] < int(args.x2)]
if args.y1:
    bf_locs = bf_locs[bf_locs["y"] > int(args.y1)]
if args.y2:
    bf_locs = bf_locs[bf_locs["y"] < int(args.y2)]

wt_locs_xy = wt_locs[["x", "y"]].copy()
bf_locs_xy = bf_locs[["x", "y"]].copy()

wt_locs_xy = np.vstack([wt_locs_xy[item] for item in ["x", "y"]]).T.astype(
    np.int64
)  # Parse to numpy array:
bf_locs_xy = np.vstack([bf_locs_xy[item] for item in ["x", "y"]]).T.astype(np.int64)

if len(wt_locs_xy) != len(
    bf_locs_xy
):  # If number of points is different after filtering give an error an exit
    print(
        "Different number of spots after filtering (wt: "
        + str(len(wt_locs_xy))
        + " vs bf: "
        + str(len(bf_locs_xy))
        + "). Calculation can't continue"
    )
    print("Check your filtering settings")
else:  # If number of points is the same, calculate affine transform
    affine_transform = estimateAffinePartial2D(bf_locs_xy, wt_locs_xy)
    transform_mat = affine_transform[0]

    # Manually affine transform the points to output alignment plot

    transform_mat_for_points = np.vstack(
        [transform_mat, (0, 0, 1)]
    )  # have to add this row for affine transform

    warped_bf_locs = []

    for point in bf_locs_xy:
        new_point = (
            point[0],
            point[1],
            1,
        )  # need to add a 1 at the end of the point for affine transform
        # print(new_point)
        transformed_point = np.matmul(
            transform_mat_for_points, new_point
        )  # do the transformation
        new_point = (
            transformed_point[0],
            transformed_point[1],
        )  # transformed point
        warped_bf_locs.append(new_point)

    warped_bf_locs = np.array(warped_bf_locs)

    wt_locs_xy_sorted = wt_locs_xy[
        np.lexsort((wt_locs_xy[:, 1], wt_locs_xy[:, 0]))
    ]  # Sort the points in case they are in different orders
    warped_bf_locs_sorted = warped_bf_locs[
        np.lexsort((warped_bf_locs[:, 1], warped_bf_locs[:, 0]))
    ]
    rmsd = np.sqrt(
        ((((wt_locs_xy_sorted - warped_bf_locs_sorted) ** 2)) * 3).mean()
    )  # calculate RMSD

    numpyData = {
        "transform_matrix": transform_mat,
        "rmsd": rmsd,
        "wt_file": args.wt_file,
        "bf_file": args.bf_file,
    }  # Write transform matrix and rmsd to file
    with open(output_path + "transform_matrix.json", "w") as write_file:
        json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

    # Remove files created during localization
    if args.delete_temp_files:
        os.remove(wt_locs_path)
        os.remove(bf_locs_path)
        os.remove(output_path + Path(wt_locs_path).stem + ".yaml")
        os.remove(output_path + Path(bf_locs_path).stem + ".yaml")

    # %%
    # Plot aligned points
    plt.scatter(*zip(*wt_locs_xy), s=5)
    plt.scatter(*zip(*warped_bf_locs), s=5)
    plt.savefig(output_path + "aligned_points.png")


