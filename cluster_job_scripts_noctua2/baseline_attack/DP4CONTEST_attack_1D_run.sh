#!/bin/sh
#SBATCH -J "DP4CONTEST_attack_1D_run"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -A hpc-prf-obal
#SBATCH -t 12:00:00
#SBATCH -p normal
#SBATCH -o /scratch/hpc-prf-obal/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-obal/prithag/clusterout/%x-%j

cd $PFS_FOLDER/deep-learning-sca/
module reset
module load system singularity
export IMG_FILE=$PFS_FOLDER/deep-learning-sca/singularity2/deepscapy.sif
export SCRIPT_FILE=$PFS_FOLDER/deep-learning-sca/exp_dataset_attack_run.py

export DATASET=$1
export LOSS_FN=$2
export MODEL_NAME=$3
export BYTE=$4
export USE_TUNER=$5
export WEIGHT_AVERAGING=$6

module list
singularity exec -B $PFS_FOLDER/deep-learning-sca/ --nv $IMG_FILE pipenv run python $SCRIPT_FILE --dataset=$DATASET --loss_function=$LOSS_FN --model_name=$MODEL_NAME --byte=$BYTE $USE_TUNER $WEIGHT_AVERAGING

exit 0
~