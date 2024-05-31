# Vesuvius Segment2Voxel

## Overview

`vesuvius-segment2voxel` is a tool designed for the Vesuvius Challenge. It converts segment meshes into volumetric voxel labels in [Blosc2](https://github.com/Blosc/c-blosc2) NDarray format for compression and fast accessibility. The integer labels correspond to some arclength around the axis of rotation. The full process involves loading mesh segments, orienting UVs, voxelating the mesh, and assigning labels.

## Features

- Load mesh segments and their associated UV maps.
- Voxelate the mesh with specified `resolution` and `chunk sizes`, during this process a big `ply` file is created.
- Read data from the `ply` in batches and assign voxel labels based on UV coordinates and arclength around a `specified axis`.
- Utilize efficient compression and multithreading with Blosc2.

## Requirements

- Python 3.6+
- NumPy
- trimesh
- tqdm
- Pillow
- SciPy
- plyfile
- pandas
- blosc2
- subprocess

## Installation

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

## Note

This script uses `obj2voxel` from the [Eisenwave/obj2voxel](https://github.com/Eisenwave/obj2voxel) repository to convert OBJ files to voxel representations. Please follow these steps:

1. Download and compile the `obj2voxel` tool from the given repository.
2. Provide the path to the executable using the `--obj2voxel` argument.


## Usage

The main script for voxelating and labeling can be executed with the following command:

```sh
python segment2voxel.py --work_dir <WORKING_DIRECTORY> --segment_id <SEGMENT_ID> --chunk <CHUNK_SIZE> --axis <UV_AXIS> --workers <NUM_WORKERS> --batch_size <BATCH_SIZE> --obj2voxel <OBJ2VOXEL_EXECUTABLE>
```

### Arguments

- `--work_dir`: The working directory containing segment folders.
- `--segment_id`: The segment ID to process.
- `--chunk`: Chunk size for compressing (default: 256).
- `--axis`: UV axis to use for labeling (default: 0).
- `--workers`: Number of workers for Blosc2 (default: 16).
- `--batch_size`: Batch size for processing vertices (default: 8000000).
- `--obj2voxel`: Path to the `obj2voxel` executable.


## Example

```sh
python segment2voxel.py --work_dir ./data --segment_id segment_01 --chunk 256 --axis 0 --workers 16 --batch_size 8000000 --obj2voxel /path/to/obj2voxel
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This script is based on the work from the Vesuvius Challenge and uses `obj2voxel` from the [Eisenwave/obj2voxel](https://github.com/Eisenwave/obj2voxel) and a function taken from [ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor).
