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
from picasso import io, postprocess
import json
from json import JSONEncoder
import argparse
import itertools
import math
import subprocess


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
    default="./",
    help="Output directory (will be created if it doesn't exist)",
)
parser.add_argument(
    "-m", "--transform-matrix", help="Previously calculated matrix in .json format"
)
parser.add_argument("-f", "--fit_method", default="lq", help="Fit method for picasso")
parser.add_argument("-b", "--box_size", default=21, help="Box sized for picasso")
parser.add_argument(
    "-g", "--min_gradient", default=70000, help="Minimum gradient for picasso"
)
parser.add_argument(
    "-e",
    "--max_pos_error",
    default=3.5,
    help="Maximum standard dev accepted for x and y position of spots",
)
parser.add_argument("-p", "--max_photons", help="Maximum number of photons spots")


args = parser.parse_args()

irm_path = args.irm_file
wt_path = args.wt_file
output_path = (
    args.output_directory + "/"
)  # The trailing slash is in case it wasn't added by the user


# %%

# Check if the directory already exists
if not os.path.exists(output_path):
    # Create the directory
    os.makedirs(output_path)
    print("Output directory created successfully!")
else:
    print("Output directory already exists")

# %%

irm = lk.ImageStack(irm_path)  # Loading a stack.
wt = lk.ImageStack(wt_path)  # Loading a stack.
wt.export_tiff(
    output_path + os.path.basename(wt_path) + "_aligned.tif"
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
print(wt_roi)

irm_metadata = irm._tiff_image_metadata()
irm_roi = irm_metadata[
    "Region of interest (x, y, width, height)"
]  # This is different because the wt was previously aligned I think. Can this cause issues?
print(irm_roi)

# %%
# Pad both images to region of interest

wt_g_padded = np.pad(wt_g, [(int(wt_roi[0]), 0), (int(wt_roi[1]), 0)])
tifffile.imwrite(output_path + "wt_padded.tif", wt_g_padded)
# plt.imshow(wt_g_padded)

irm_g_padded = np.pad(irm_g, [(int(irm_roi[0]), 0), (int(irm_roi[1]), 0)])
tifffile.imwrite(output_path + "irm_padded.tif", irm_g_padded)
# plt.imshow(irm_g_padded)

# %%


transform_mat = []  # set to empty to check afterwards if I have a matrix

if args.transform_matrix:  # If I have provided a matrix, use that
    with open(args.transform_matrix, "r") as read_file:
        decodedArray = json.load(read_file)
        transform_mat = np.asarray(decodedArray["transform matrix"])
        rmsd = decodedArray["rmsd"]

# %%
else:  # if matrix wasnt provided, calculate it
    run_string = (
        "python -m picasso localize "
        + output_path
        + "wt_padded.tif --fit-method "
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
        + "irm_padded.tif --fit-method "
        + args.fit_method
        + " -b "
        + str(args.box_size)
        + " --gradient "
        + str(args.min_gradient)
    )
    subprocess.run(run_string)
    #!python -m picasso localize {output_path}wt_padded.tif --fit-method $args.fit_method -b  $args.box_size --gradient $args.min_gradient
    #!python -m picasso localize {output_path}irm_padded.tif --fit-method $args.fit_method -b $args.box_size --gradient $args.min_gradient

    # %%
    irm_locs, irm_info = io.load_locs(output_path + "irm_padded_locs.hdf5")
    wt_locs, wt_info = io.load_locs(output_path + "wt_padded_locs.hdf5")

    # %%
    wt_locs = wt_locs[wt_locs["sx"] < args.max_pos_error]
    wt_locs = wt_locs[wt_locs["sy"] < args.max_pos_error]
    irm_locs = irm_locs[irm_locs["sx"] < args.max_pos_error]
    irm_locs = irm_locs[irm_locs["sy"] < args.max_pos_error]
    if args.max_photons:
        wt_locs = wt_locs[wt_locs["photons"] < 1500000]
        irm_locs = irm_locs[irm_locs["photons"] < 1500000]

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
            + len(wt_locs_xy)
            + " vs irm: "
            + len(irm_locs_xy)
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
            "transform matrix": transform_mat,
            "rmsd": rmsd,
        }  # Write transform matrix and rmsd to file
        with open(output_path + "transform_matrix.json", "w") as write_file:
            json.dump(numpyData, write_file, cls=NumpyArrayEncoder)

        # %%
        # Plot aligned points
        plt.scatter(*zip(*wt_locs_xy), s=5)
        plt.scatter(*zip(*warped_irm_locs), s=5)
        plt.savefig(output_path + "aligned_points.png")
# %%

# %%
if transform_mat != []:  # If I have a matrix either from file or calculated
    irm_g_padded_warped = warpAffine(
        irm_g_padded, transform_mat, (wt_g_padded.shape[1], wt_g_padded.shape[0])
    )
    irm_g_padded_warped_cropped = irm_g_padded_warped[
        wt_roi[0] : wt_roi[0] + wt_g.shape[0], wt_roi[1] : wt_roi[1] + wt_g.shape[1]
    ]  # crop to size of wt
    # Save aligned padded images (not currently not used):
    # tifffile.imwrite("data/output/aligned_irm_padded.tif", irm_g_padded_warped)
    # tifffile.imwrite("data/output/aligned_wt_padded.tif", wt_g_padded)

    tifffile.imwrite(
        output_path + os.path.basename(irm_path) + "_aligned.tif",
        irm_g_padded_warped_cropped,
        metadata=irm_metadata,
    )  # save irm image without the padding

    stacked_image = np.stack(
        [wt_g, irm_g_padded_warped_cropped], axis=0
    )  # Save stacked g and irm image
    tifffile.imwrite(output_path + "multichannel_aligned.tif", stacked_image)

    # %%

    # %%
    # delete leftover files
    list_of_output_files = os.listdir(output_path)
    for file in list_of_output_files:
        if (
            file.endswith(".hdf5")
            or file.endswith(".yaml")
            or file.endswith("_padded.tif")
        ):
            os.remove(os.path.join(output_path, file))
