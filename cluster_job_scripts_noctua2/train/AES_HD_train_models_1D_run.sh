#!/bin/sh 
#SBATCH -J "AES_HD_1D_run"
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-obal
#SBATCH -t 1-00:00:00
#SBATCH -p gpu
#SBATCH -o /scratch/hpc-prf-obal/prithag/clusterout/%x-%j
#SBATCH -e /scratch/hpc-prf-obal/prithag/clusterout/%x-%j

cd $PFS_FOLDER/deep-learning-sca/
module reset
module load system singularity
export IMG_FILE=$PFS_FOLDER/deep-learning-sca/singularity2/deepscapy.sif
export SCRIPT_FILE=$PFS_FOLDER/deep-learning-sca/exp_baseline_models_run.py

export DATASET=$1
export LOSS_FN=$2
export MODEL_NAME=$3
export USE_TUNER=$4
export WEIGHT_AVERAGING=$5
export LEAKAGE_MODEL=$6

module list
singularity exec -B $PFS_FOLDER/deep-learning-sca/ --nv $IMG_FILE pipenv run python $SCRIPT_FILE --dataset=$DATASET --loss_function=$LOSS_FN --model_name=$MODEL_NAME $USE_TUNER $WEIGHT_AVERAGING --leakage_model=$LEAKAGE_MODEL

exit 0
~