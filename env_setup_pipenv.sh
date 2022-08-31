#!/bin/bash

ml load system singularity
rm -rf .venv/
export IMG_FILE=$PFS_FOLDER/deep-learning-sca/singularity2/deepscapy.sif
singularity exec -B $PFS_FOLDER/deep-learning-sca/ --nv $IMG_FILE pipenv install