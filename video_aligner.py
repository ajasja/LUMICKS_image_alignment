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


def norm_image(image, inverse=False):
    amin = image.min()
    amax = image.max()

    if amax != amin:
        if inverse:
            return 1 - (image - amin) / (amax - amin)
        else:
            return (image - amin) / (amax - amin)
    else:
        return image


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


parser = argparse.ArgumentParser(
    description='"Scripts to align various channels based on reference beads."',
    epilog=""" """,
)
parser.add_argument("wt_file", help="WT tif file")
parser.add_argument("irm_file", help="IRM tif file")
parser.add_argument(
    "-bf",
    "--bright-field-file",
    help="Bright field file (optional)",
)
parser.add_argument(
    "-bfm",
    "--bf-transform-matrix",
    help="Previously calculated matrix for brighfield in .json format",
)
parser.add_argument(
    "-m", "--transform-matrix", help="Previously calculated matrix in .json format"
)
parser.add_argument(
    "-o",
    "--output-directory",
    default="output",
    help="Output directory. Default=output/",
)
args = parser.parse_args()

irm_path = args.irm_file
wt_path = args.wt_file
if args.bright_field_file is not None:
    bright_path = args.bright_field_file
    align_brightfield = True
    if args.bf_transform_matrix is not None:
        bf_transform_matrix_file = args.bf_transform_matrix
    else:
        print(
            "Error. Bright field file provided but not the bright field matrix. Terminating"
        )
        exit()
else:
    align_brightfield = False


output_path = (
    args.output_directory + "/"
)  # The trailing slash is in case it wasn't added by the user

transform_matrix_file = args.transform_matrix
use_existing_matrix = True  # always

# %%
# Check if the directory already exists
os.makedirs(output_path, exist_ok=True)

# %%
"""
if irm_path.endswith(".tiff"):
    print(f"renaming {irm_path} to .tif")
    os.rename(irm_path, irm_path[:-1])
    irm_path = irm_path[:-1]
if wt_path.endswith(".tiff"):
    print(f"renaming {wt_path} to .tif")
    os.rename(wt_path, wt_path[:-1])
    wt_path = wt_path[:-1]

if align_brightfield and bright_path.endswith(".tiff"):
    print(f"renaming {bright_path} to .tif")
    os.rename(bright_path, bright_path[:-1])
    bright_path = bright_path[:-1]
"""
# %%
irm = lk.ImageStack(irm_path)  # Loading a stack.
wt = lk.ImageStack(wt_path)  # Loading a stack.

if align_brightfield:
    bright_file = lk.ImageStack(bright_path)
    bright_g_video = bright_file.get_image(channel="green")
    bright_g = bright_g_video[0]
    bright_metadata = bright_file._tiff_image_metadata()

wt.export_tiff(
    output_path + Path(wt_path).stem + "_aligned.tif"
)  # Save aligned wt stack

# %%
# Get channels
wt_g_video = wt.get_image(channel="green")
irm_g_video = irm.get_image()
wt_r_video = wt.get_image(channel="red")
wt_b_video = wt.get_image(channel="blue")

wt_g = wt_g_video[0]
wt_r = wt_r_video[0]
wt_b = wt_b_video[0]
irm_g = irm_g_video[0]

# %%
# Get metadata

wt_metadata = wt._tiff_image_metadata()
wt_framerate = wt_metadata["Framerate (Hz)"]
wt_roi = wt_metadata["Region of interest (x, y, width, height)"]
wt_frame_averaging = wt_metadata["Frame averaging"]
# print(wt_framerate)

irm_metadata = irm._tiff_image_metadata()
irm_roi = irm_metadata[
    "Region of interest (x, y, width, height)"
]  # This is different because the wt was prexviously aligned I think. Can this cause issues?
irm_framerate = irm_metadata["Framerate (Hz)"]
irm_frame_averaging = irm_metadata["Frame averaging"]
# print(irm_framerate)

if align_brightfield:
    bright_roi = bright_metadata[
        "Region of interest (x, y, width, height)"
    ]  # This is different because the wt was prexviously aligned I think. Can this cause issues?
    bf_framerate = bright_metadata["Framerate (Hz)"]
    bf_frame_averaging = bright_metadata["Frame averaging"]
    # print(bright_roi)


# Pad both images to region of interest
padded_wt_filename = Path(wt_path).stem + "_padded.tif"
wt_g_padded = np.pad(wt_g, [(int(wt_roi[1]), 0), (int(wt_roi[0]), 0)])
# wt_g_padded = wt_g
# tifffile.imwrite(output_path + padded_wt_filename, wt_g_padded)

wt_r_padded = np.pad(wt_r, [(int(wt_roi[1]), 0), (int(wt_roi[0]), 0)])
wt_b_padded = np.pad(wt_b, [(int(wt_roi[1]), 0), (int(wt_roi[0]), 0)])


padded_irm_filename = Path(irm_path).stem + "_padded.tif"
irm_g_padded = np.pad(irm_g, [(int(irm_roi[1]), 0), (int(irm_roi[0]), 0)])
# irm_g_padded = irm_g
# tifffile.imwrite(output_path + padded_irm_filename, irm_g_padded)

if align_brightfield:
    padded_bright_filename = Path(bright_path).stem + "_padded.tif"
    bright_g_padded = np.pad(
        bright_g, [(int(bright_roi[1]), 0), (int(bright_roi[0]), 0)]
    )
    # irm_g_padded = irm_g
    # irm_g_padded = np.pad(irm_g, [(int(wt_roi[0]), 0), (int(wt_roi[1]), 0)])
    # irm_g_padded = np.pad(irm_g_padded, [(int(wt_roi[0]), 0), (int(wt_roi[1]), 0)])
    # tifffile.imwrite(output_path + padded_bright_filename, bright_g_padded)


# plt.imshow(bright_g_padded, alpha=0.5)


# ax1.imshow(irm_g_padded, alpha=0.5)
# ax1.set_xlim(irm_roi[0], irm_roi[0] + irm_roi[2])
# ax1.set_ylim(irm_roi[1], irm_roi[1] + irm_roi[3])


# ax1.imshow(bright_g_padded, alpha=0.5)
# ax1.set_xlim(bright_roi[0], bright_roi[0] + bright_roi[2])
# ax1.set_ylim(bright_roi[1], bright_roi[1] + bright_roi[3])

# %%
transform_mat = []  # set to empty to check afterwards if I have a matrix
bf_transform_mat = []


if use_existing_matrix:  # If I have provided a matrix, use that
    with open(transform_matrix_file, "r") as read_file:
        decodedArray = json.load(read_file)
        transform_mat = np.asarray(decodedArray["transform_matrix"])
        rmsd = decodedArray["rmsd"]
        # print(transform_mat)
    if align_brightfield:
        with open(bf_transform_matrix_file, "r") as read_file:
            decodedArray = json.load(read_file)
            bf_transform_mat = np.asarray(decodedArray["transform_matrix"])
            # print(bf_transform_mat)
            rmsd = decodedArray["rmsd"]


# %%
# print(irm_framerate)

real_irm_framerate = irm_framerate / irm_frame_averaging
if align_brightfield:
    real_bf_framerate = bf_framerate / bf_frame_averaging
real_wt_framerate = wt_framerate / wt_frame_averaging


# %%
irm_warped_video = []
bf_warped_video = []
wt_g_out_video = []
wt_r_out_video = []
wt_b_out_video = []

if len(transform_mat != 0):  # If I have a matrix either from file or calculated

    for frame_n, frame in enumerate(wt_g_video):

        irm_g_padded = np.pad(
            irm_g_video[round(frame_n * real_irm_framerate / real_wt_framerate)],
            [(int(irm_roi[1]), 0), (int(irm_roi[0]), 0)],
        )
        wt_g_padded = np.pad(frame, [(int(wt_roi[1]), 0), (int(wt_roi[0]), 0)])
        wt_r_padded = np.pad(
            wt_r_video[frame_n], [(int(wt_roi[1]), 0), (int(wt_roi[0]), 0)]
        )  # I think R and B shouldnt need padding and cropping, but let's just keep everything consistent
        wt_b_padded = np.pad(
            wt_b_video[frame_n], [(int(wt_roi[1]), 0), (int(wt_roi[0]), 0)]
        )

        irm_g_padded_warped = warpAffine(
            irm_g_padded, transform_mat, (wt_g_padded.shape[1], wt_g_padded.shape[0])
        )

        # This hack is done to reduce the total contrast in the resulting image
        # Otherwise, the 0s make it hard to see
        irm_g_padded_warped[irm_g_padded_warped <= np.amin(irm_g_padded)] = np.mean(
            irm_g_padded
        )

        irm_g_padded_warped = norm_image(irm_g_padded_warped)[
            int(wt_roi[1]) : int(wt_roi[1]) + int(wt_roi[3]),
            int(wt_roi[0]) : int(wt_roi[0]) + int(wt_roi[2]),
        ]
        wt_g_padded = norm_image(wt_g_padded)[
            int(wt_roi[1]) : int(wt_roi[1]) + int(wt_roi[3]),
            int(wt_roi[0]) : int(wt_roi[0]) + int(wt_roi[2]),
        ]
        wt_r_padded = norm_image(wt_r_padded)[
            int(wt_roi[1]) : int(wt_roi[1]) + int(wt_roi[3]),
            int(wt_roi[0]) : int(wt_roi[0]) + int(wt_roi[2]),
        ]
        wt_b_padded = norm_image(wt_b_padded)[
            int(wt_roi[1]) : int(wt_roi[1]) + int(wt_roi[3]),
            int(wt_roi[0]) : int(wt_roi[0]) + int(wt_roi[2]),
        ]

        wt_g_out_video.append(wt_g_padded)
        wt_r_out_video.append(wt_r_padded)
        wt_b_out_video.append(wt_b_padded)
        irm_warped_video.append(irm_g_padded_warped)

        if align_brightfield:
            if len(bf_transform_mat) != 0:
                bf_g_padded = np.pad(
                    bright_g_video[
                        round(frame_n * real_bf_framerate / real_wt_framerate)
                    ],
                    [(int(bright_roi[1]), 0), (int(bright_roi[0]), 0)],
                )

                bf_g_padded_warped = warpAffine(
                    bf_g_padded,
                    bf_transform_mat,
                    (wt_g_padded.shape[1], wt_g_padded.shape[0]),
                )

                # This hack is done to reduce the total contrast in the resulting image
                # Otherwise, the 0s make it hard to see

                bf_g_padded_warped[bf_g_padded_warped <= np.amin(bf_g_padded)] = (
                    np.mean(bf_g_padded)
                )

                bf_g_padded_warped = norm_image(bf_g_padded_warped, False)
                bf_warped_video.append(bf_g_padded_warped)

        # tifffile.imwrite(
        # output_path + Path(irm_path).stem + f"_aligned_{frame_n}.tif",
        # irm_g_padded_warped_cropped,

        # metadata=irm_metadata,

        # )  # save irm image without the padding
        # plt.imshow(irm_g_padded, alpha=0.5)
        # plt.imshow(bf_g_padded_warped, alpha=0.5, cmap="Blues")

    if align_brightfield:
        stacked_video = np.stack(
            [
                wt_r_out_video,
                wt_g_out_video,
                wt_b_out_video,
                (
                    [np.empty(wt_g_out_video[0].shape)] * len(wt_g_out_video)
                ),  # skip gray channel
                irm_warped_video,
                bf_warped_video,
            ],
            axis=1,
        )  # Save stacked g and irm image

    else:

        stacked_video = np.stack(
            [
                wt_r_out_video,
                wt_g_out_video,
                wt_b_out_video,
                (
                    [np.empty(wt_g_out_video[0].shape)] * len(wt_g_out_video)
                ),  # skip gray channel
                irm_warped_video,
            ],
            axis=1,
        )  # Save stacked g and irm image

    tifffile.imwrite(
        output_path + Path(wt_path).stem + "_multichannel_aligned.tif",
        np.float32(stacked_video),
        imagej=True,
        metadata={
            "Composite mode": "composite",  # This is what was needed for fiji to open it merged
        },
    )

    # plt.imshow(wt_g)

    # delete padded files

    # os.remove(output_path + padded_irm_filename)

    # os.remove(output_path + padded_wt_filename)
