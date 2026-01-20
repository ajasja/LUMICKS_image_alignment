# LUMICKS_image_alignment
Scripts to align various channels based on reference beads. 

## Installation

First create the conda environment
>conda env create --name <ImageAligner> -f image_aligner_env.yml

then activate it

>conda activate <ImageAligner>

and then install picasso

>pip install picassosr

The installation usually takes approximately half an hour.

---

## Running the script

Remember to first activate the environment

>conda activate <ImageAligner>

To align images easily simply run the script

>python image_aligner.py <wt_image> <irm_image> -m <transform_matrix>

or

>python video_aligner.py <wt_image> <irm_image> -m <transform_matrix>

for more help on commands run

>python image_aligner.py --help
>
>python video_aligner.py --help

The scripts can also be run on batch by using the files Align_folder.ipynb, Align_folder_with_brightfield.ipynb and Video_Align_folder_with_brightfield.ipynb

---

## Instructions for running the demo

1. Run the `Video_Align_folder.ipynb` notebook using the following inputs:
   - `folder`: path to the `example` folder (containing unaligned WT and IRM files)
   - `transform_matrix`: path to `transform_matrix.json`

### Expected output

The expected output is a `.tif` file with the following basename:  
`<timestamp>_WT_multichannel_aligned.tif`

The file is a stack of 4 channels:

1. Red channel (walker)
2. Blue channel (track)
3. Green channel (empty)
4. IRM channel (track)

The output file can be opened in ImageJ. The movie can be cropped so that it only contains one track (the example output HAS been cropped in order to fit Github size limitations). The contrast of the IRM channel has to be further autoadjusted for the undrifting in the next step to work properly.

### Expected runtime

1 minute.