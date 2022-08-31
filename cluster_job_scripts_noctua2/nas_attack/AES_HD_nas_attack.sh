declare -a dataset_array=("AES_HD")
declare -a loss_array=("categorical_crossentropy" "ranking_loss" "cross_entropy_ratio" "sigmoid_focal_binary_crossentropy" "sigmoid_focal_categorical_crossentropy" "sigmoid_focal_categorical_crossentropy_ratio" "sigmoid_focal_binary_crossentropy_ratio")
declare -a model_array=("nas_basic5")
declare -a tuner_array=("hyperband" "greedy" "random" "bayesian")
declare -a byte_array=(0)
declare -a reshape_type_array=("2dCNNSqr" "2dCNNRect" "1dCNN")

for dataset in "${dataset_array[@]}";
do
  for loss in "${loss_array[@]}";
  do
    for model in "${model_array[@]}";
    do
      for tuner in "${tuner_array[@]}";
      do
        for byte in "${byte_array[@]}";
        do
          for reshape_type in "${reshape_type_array[@]}";
          do
            sbatch AES_HD_nas_attack_run.sh $dataset $loss $model $tuner $byte $reshape_type
          done
        done
      done
    done
  done
done

