#!/bin/bash

ml load system singularity
export IMG_FILE=$PFS_FOLDER/deep-learning-sca/singularity/deepscapy.sif
singularity exec -B $PFS_FOLDER/deep-learning-sca/ --nv $IMG_FILE poetry run pip install pyreadline
singularity exec -B $PFS_FOLDER/deep-learning-sca/ --nv $IMG_FILE poetry install -vvv
