declare -a dataset_array=("DP4CONTEST")
declare -a loss_array=("categorical_crossentropy" "ranking_loss" "cross_entropy_ratio" "sigmoid_focal_binary_crossentropy" "sigmoid_focal_categorical_crossentropy" "sigmoid_focal_categorical_crossentropy_ratio" "sigmoid_focal_binary_crossentropy_ratio")
declare -a model_array=("ascad_mlp_baseline" "ascad_cnn_baseline" "cnn_zaid_baseline")
declare -a byte_array=(0)
declare -a use_tuner_array=("--no-use_tuner")
declare -a use_weight_averaging_array=("--no-weight_averaging")

for dataset in "${dataset_array[@]}";
do
  for loss in "${loss_array[@]}";
  do
    for model in "${model_array[@]}";
    do
      for byte in "${byte_array[@]}";
      do
        for use_tuner in "${use_tuner_array[@]}";
        do
          for weight_averaging in "${use_weight_averaging_array[@]}";
          do
            sbatch DP4CONTEST_attack_1D_run.sh $dataset $loss $model $dataset_dimension $byte $use_tuner $weight_averaging
          done
        done
      done
    done
  done
done
