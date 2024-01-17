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
parser.add_argument("irm_file", help="IRM tif file")
parser.add_argument(
    "-o",
    "--output-directory",
    default="output",
    help="Output directory. Default=output/",
)
parser.add_argument(
    "-m", "--transform-matrix", help="Previously calculated matrix in .json format"
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

args = parser.parse_args()

irm_path = args.irm_file
wt_path = args.wt_file
output_path = (
    args.output_directory + "/"
)  # The trailing slash is in case it wasn't added by the user


# %%

# Check if the directory already exists
os.makedirs(output_path, exist_ok=True)
# %%

if not args.transform_matrix:
    # Copy input files to output folder if calculating new matrix
    shutil.copy2(irm_path, output_path + os.path.basename(irm_path))
    shutil.copy2(wt_path, output_path + os.path.basename(wt_path))

    irm_path = output_path + os.path.basename(irm_path)
    wt_path = output_path + os.path.basename(wt_path)

    # rename tiff to tif files
    if irm_path.endswith(".tiff"):
        print(irm_path)
        os.rename(irm_path, irm_path[:-1])
        irm_path = irm_path[:-1]
    if wt_path.endswith(".tiff"):
        print(wt_path)
        os.rename(wt_path, wt_path[:-1])
        wt_path = wt_path[:-1]

# %%
irm = lk.ImageStack(irm_path)  # Loading a stack.
wt = lk.ImageStack(wt_path)  # Loading a stack.
wt.export_tiff(
    output_path + Path(wt_path).stem + "_aligned.tif"
)  # Save aligned wt stack


# %%

# Get channels
wt_g = wt.get_image(channel="green")
# wt_r = wt.get_image(channel='red')  #not really used
# wt_b = wt.get_image(channel='blue') #not really used
irm_g = irm.get_image()


# %%
# Get region of interest data

wt_metadata = wt._tiff_image_metadata()
wt_roi = wt_metadata["Alignment region of interest (x, y, width, height)"]

irm_metadata = irm._tiff_image_metadata()
irm_roi = irm_metadata[
    "Region of interest (x, y, width, height)"
]  # This is different because the wt was previously aligned I think. Can this cause issues?

# %%
# Padding is CANCELED. Once this is all working flawlessly I should fix the code to remove references to padding

padded_wt_filename = Path(wt_path).stem + "_padded.tif"
# wt_g_padded = np.pad(wt_g, [(int(wt_roi[0]), 0), (int(wt_roi[1]), 0)])
wt_g_padded = wt_g
tifffile.imwrite(output_path + padded_wt_filename, wt_g_padded)


padded_irm_filename = Path(irm_path).stem + "_padded.tif"
# irm_g_padded = np.pad(irm_g, [(int(irm_roi[0]), 0), (int(irm_roi[1]), 0)])
irm_g_padded = irm_g
tifffile.imwrite(output_path + padded_irm_filename, irm_g_padded)

# %%

transform_mat = []  # set to empty to check afterwards if I have a matrix

if args.transform_matrix:  # If I have provided a matrix, use that
    with open(args.transform_matrix, "r") as read_file:
        decodedArray = json.load(read_file)
        transform_mat = np.asarray(decodedArray["transform_matrix"])
        rmsd = decodedArray["rmsd"]

# %%
else:  # if matrix wasnt provided, calculate it
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
        + padded_irm_filename
        + " --fit-method "
        + args.fit_method
        + " -b "
        + str(args.box_size)
        + " --gradient "
        + str(args.min_gradient)
    )
    subprocess.run(run_string)

    # %%

    irm_locs_path = output_path + Path(padded_irm_filename).stem + "_locs.hdf5"
    irm_locs, irm_info = io.load_locs(irm_locs_path)
    wt_locs_path = output_path + Path(padded_wt_filename).stem + "_locs.hdf5"
    wt_locs, wt_info = io.load_locs(wt_locs_path)

    # %%
    wt_locs = wt_locs[wt_locs["sx"] < args.max_pos_error]
    wt_locs = wt_locs[wt_locs["sy"] < args.max_pos_error]
    irm_locs = irm_locs[irm_locs["sx"] < args.max_pos_error]
    irm_locs = irm_locs[irm_locs["sy"] < args.max_pos_error]
    if args.max_photons:
        wt_locs = wt_locs[wt_locs["photons"] < args.max_photons]
        irm_locs = irm_locs[irm_locs["photons"] < args.max_photons]

    wt_locs_xy = wt_locs[["x", "y"]].copy()
    irm_locs_xy = irm_locs[["x", "y"]].copy()

    wt_locs_xy = np.vstack([wt_locs_xy[item] for item in ["x", "y"]]).T.astype(
        np.int64
    )  # Parse to numpy array:
    irm_locs_xy = np.vstack([irm_locs_xy[item] for item in ["x", "y"]]).T.astype(
        np.int64
    )

    if len(wt_locs_xy) != len(
        irm_locs_xy
    ):  # If number of points is different after filtering give an error an exit
        print(
            "Different number of spots after filtering (wt: "
            + str(len(wt_locs_xy))
            + " vs irm: "
            + str(len(irm_locs_xy))
            + "). Calculation can't continue"
        )
        print("Check your filtering settings")
    else:  # If number of points is the same, calculate affine transform
        affine_transform = estimateAffinePartial2D(irm_locs_xy, wt_locs_xy)
        transform_mat = affine_transform[0]

        # Manually affine transform the points to output alignment plot

        transform_mat_for_points = np.vstack(
            [transform_mat, (0, 0, 1)]
        )  # have to add this row for affine transform

        warped_irm_locs = []

        for point in irm_locs_xy:
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
            warped_irm_locs.append(new_point)

        warped_irm_locs = np.array(warped_irm_locs)

        wt_locs_xy_sorted = wt_locs_xy[
            np.lexsort((wt_locs_xy[:, 1], wt_locs_xy[:, 0]))
        ]  # Sort the points in case they are in different orders
        warped_irm_locs_sorted = warped_irm_locs[
            np.lexsort((warped_irm_locs[:, 1], warped_irm_locs[:, 0]))
        ]
        rmsd = np.sqrt(
            ((((wt_locs_xy_sorted - warped_irm_locs_sorted) ** 2)) * 3).mean()
        )  # calculate RMSD

        numpyData = {
            "transform_matrix": transform_mat,
            "rmsd": rmsd,
            "wt_file": args.wt_file,
            "irm_file": args.irm_file,
        }  # Write transform matrix and rmsd to file
        with open(output_path + "transform_matrix.json", "w") as write_file:
            json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

        # Remove files created during localization
        if args.delete_temp_files:
            os.remove(wt_locs_path)
            os.remove(irm_locs_path)
            os.remove(output_path + Path(wt_locs_path).stem + ".yaml")
            os.remove(output_path + Path(irm_locs_path).stem + ".yaml")

        # %%
        # Plot aligned points
        plt.scatter(*zip(*wt_locs_xy), s=5)
        plt.scatter(*zip(*warped_irm_locs), s=5)
        plt.savefig(output_path + "aligned_points.png")


# %%
if len(transform_mat) != 0:  # If I have a matrix either from file or calculated
    irm_g_padded_warped = warpAffine(
        irm_g_padded, transform_mat, (wt_g_padded.shape[1], wt_g_padded.shape[0])
    )

    # This hack removes the 0s from the affine transform
    # Otherwise, the 0s make it hard to see because of contrast
    irm_g_padded_warped[irm_g_padded_warped <= np.amin(irm_g_padded)] = np.mean(
        irm_g_padded
    )

    # normalize
    irm_g_padded_warped = norm_image(irm_g_padded_warped, False)
    wt_g_padded = norm_image(wt_g_padded)

    irm_g_padded_warped_cropped = irm_g_padded_warped[
        0 : wt_g.shape[0], 0 : wt_g.shape[1]
    ]  # crop to size of wt

    tifffile.imwrite(
        output_path + Path(irm_path).stem + "_aligned.tif",
        irm_g_padded_warped_cropped,
        metadata=irm_metadata,
    )  # save irm image without the padding

    stacked_image = np.stack(
        [wt_g_padded, irm_g_padded_warped_cropped], axis=0
    )  # Save stacked g and irm image
    tifffile.imwrite(
        output_path + Path(wt_path).stem + "_multichannel_aligned.tif",
        np.float32(stacked_image),
        imagej=True,
        metadata={
            "Composite mode": "composite",  # This is what was needed for fiji to open it merged
        },
    )

    # %%
    if args.delete_temp_files:
        # delete leftover files
        os.remove(output_path + padded_irm_filename)
        os.remove(output_path + padded_wt_filename)
        if not args.transform_matrix:
            # If calculating a new matrix, delete the temp files
            irm._src.close()  # need to close the file before deleting
            wt._src.close()  # need to close the file before deleting
            os.remove(irm_path)
            os.remove(wt_path)
