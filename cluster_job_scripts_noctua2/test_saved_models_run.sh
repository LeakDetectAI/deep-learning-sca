#!/bin/sh
#SBATCH -J "CheckSavedModels"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -A hpc-prf-obal
#SBATCH -t 10:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-obal/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-obal/prithag/clusterout/%x-%j

cd $PFS_FOLDER/deep-learning-sca/
module reset
module load system singularity
export IMG_FILE=$PFS_FOLDER/deep-learning-sca/singularity2/deepscapy.sif
export SCRIPT_FILE=$PFS_FOLDER/deep-learning-sca/exp_get_nas_models_finished.py

export DATASET=$1

module list
singularity exec -B $PFS_FOLDER/deep-learning-sca/ --nv $IMG_FILE pipenv run python $SCRIPT_FILE

exit 0
~