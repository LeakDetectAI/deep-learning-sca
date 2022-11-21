#!/bin/bash

rsync -avz -P --rsh="ssh -o StrictHostKeyChecking=no -l anonymous" --exclude=".git" --exclude="results" --exclude="logs/" --exclude="nas_trials_directory*/" --exclude="trained_models/" --exclude="excel*" --exclude="wandb/" \
--exclude="build"  --exclude="dist" --exclude=".egg-info" --exclude=".~lock." --exclude=".idea" --exclude="*.pyc" --exclude="*.gitignore" --exclude="\*\sandbox" ~/git/deep-learning-sca n2login1.ab2021.pc2.uni-paderborn.de:/scratch/hpc-prf-obal/anonymous/