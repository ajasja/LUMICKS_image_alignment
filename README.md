# LUMICKS Image Alignment

Scripts and tools to align multi-channel microscopy images from LUMICKS optical tweezers systems based on reference beads. This project provides both command-line tools and Jupyter notebooks for single-image, batch, and video alignment workflows.

## Overview

The LUMICKS image alignment system calibrates and applies affine transformations to align different imaging channels (WT, IRM, etc.) using reference beads as fiducial markers. The transformation matrix is computed once during calibration and can then be applied to align multiple datasets.

### Key Features
- Single image alignment via command-line interface
- Batch alignment of image folders via Jupyter notebooks
- Video alignment support for time-lapse datasets
- Brightfield reference channel option
- HDF5 and YAML coordinate format support
- Configurable transformation matrices

## Project Structure

```
├── image_aligner.py              # Single image alignment script
├── video_aligner.py              # Video alignment script
├── image_aligner.ipynb           # Notebook for image alignment
├── video_aligner.ipynb           # Notebook for video alignment
├── Align_folder.ipynb            # Batch alignment notebook
├── Align_folder_with_brightfield.ipynb
├── Video_Align_folder.ipynb
├── Video_Align_folder_with_brightfield.ipynb
├── Tests.ipynb                   # Testing and calibration notebook
├── transform_matrix.json         # Current transformation matrix
├── bf_transform_matrix.json      # Brightfield transformation matrix
├── image_aligner_env.yml         # Conda environment specification
├── ImageAlignerEnv.yml           # Alternative environment file
├── Data/                         # Test and calibration data
│   ├── 2025-07-30 data for new matrix/
│   ├── 2025-07-30_calibration_data/
│   ├── Arvind 20250603/          # Sample datasets
│   └── Previously_unaligned/
├── output/                       # Aligned output results
└── old_matrices/                 # Archive of previous transformation matrices
```

## Installation

### Step 1: Create Conda Environment

```bash
conda env create --name image_aligner -f image_aligner_env.yml
```

Or use the alternative environment file:

```bash
conda env create --name image_aligner -f ImageAlignerEnv.yml
```

### Step 2: Activate Environment

```bash
conda activate image_aligner
```

### Step 3: Install Additional Dependencies

```bash
pip install picassosr
```

## Usage

### Activating the Environment

Before running any scripts, always activate the environment:

```bash
conda activate image_aligner
```

### Single Image Alignment

#### Using Python Script

```bash
python image_aligner.py <wt_image> <irm_image> -m <transform_matrix>
```

**Example:**
```bash
python image_aligner.py Data/image_wt.tiff Data/image_irm.tiff -m transform_matrix.json
```

#### Using Video Alignment Script

```bash
python video_aligner.py <wt_video> <irm_video> -m <transform_matrix>
```

**Example:**
```bash
python video_aligner.py Data/video_wt.tiff Data/video_irm.tiff -m transform_matrix.json
```

#### Command-line Help

```bash
python image_aligner.py --help
python video_aligner.py --help
```

### Batch Processing

For aligning multiple images or videos, use the Jupyter notebooks:

- **Align_folder.ipynb** - Batch align static images
- **Align_folder_with_brightfield.ipynb** - Batch align with brightfield reference
- **Video_Align_folder.ipynb** - Batch align video sequences
- **Video_Align_folder_with_brightfield.ipynb** - Batch align videos with brightfield

### Calibration and Transformation Matrix

#### Computing a New Transformation Matrix

Calibration is performed using the `image_aligner.py` script **without passing a transformation matrix**:

```bash
python image_aligner.py <wt_image> <irm_image>
```

**Example:**
```bash
python image_aligner.py Data/calibration_wt.tiff Data/calibration_irm.tiff
```

This will:
1. Automatically detect reference beads in both images
2. Compute the affine transformation between channels
3. Save the transformation matrix to `transform_matrix.json`

The script needs to find the same amount of points in WT and IRM images, so adjust the parameters accordingly. The minimum and maximum coordinates can be used to crop one of the images to reduce the number of points.

After calibration, validate the transformation matrix by looking at the aligned_points.png output. The transform_matrix.json file also contains calculated RMSD

#### Optimizing Bead Localization

Before calibration determine optimal localization parameters using the **Picasso localize** program:

1. Open the Picasso Localize GUI
2. Load one of your calibration images
3. Use the localize tool to find optimal parameters for:
   - Detection sensitivity
   - Minimum photon count
   - All other parameters
4. Note the parameters that best detect your reference beads
5. Apply these parameters in the image alignment workflow

## Input/Output Formats

### Supported Input Formats
- TIFF images and video stacks

### Output Formats
- Aligned TIFF images or videos
- Transformed localization coordinates
- YAML metadata

## Transformation Matrices

- **transform_matrix.json** - Standard transformation matrix
- **bf_transform_matrix.json** - Brightfield-specific transformation matrix

## Troubleshooting

### Common Issues

If images are not being properly aligned anymore, the microscope optics might have moved. Perform calibration again. 

## Dependencies

Key dependencies (see `image_aligner_env.yml` for complete list):
- Python 3.x
- NumPy
- SciPy 
- OpenCV
- Picasso 
- Jupyter 
- PyYAML 
- h5py 

