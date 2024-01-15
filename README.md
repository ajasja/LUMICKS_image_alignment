# LUMICKS_image_alignment
Scripts to align various channels based on reference beads. 

## Installation

First create the conda environment
>conda env create --name <environment_name> -f image_aligner_env.yml

then activate it

>conda activate <environment_name>

and then install picasso

>pip install picassosr

## Running the script

Remember to first activate the environment

>conda activate <environment_name>

then just run the script

>python image_aligner.py <wt_image> <irm_image> -m <transform_matrix>

for more help on commands run

>python image_aligner.py --help
