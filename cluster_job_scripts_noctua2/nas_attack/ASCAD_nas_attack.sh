#declare -a dataset_array=("ASCAD_desync0_variable" "ASCAD_desync50_variable" "ASCAD_desync100_variable" "ASCAD_desync0" "ASCAD_desync50" "ASCAD_desync100")
declare -a dataset_array=("ASCAD_desync100_variable" "CHES_CTF")
declare -a loss_array=("categorical_crossentropy" "ranking_loss" "cross_entropy_ratio" "sigmoid_focal_binary_crossentropy" "sigmoid_focal_categorical_crossentropy" "sigmoid_focal_categorical_crossentropy_ratio" "sigmoid_focal_binary_crossentropy_ratio")
declare -a model_array=("nas_basic5")
declare -a tuner_array=("hyperband" "greedy" "random" "bayesian")
declare -a reshape_type_array=("2dCNNSqr" "2dCNNRect" "1dCNN")

for dataset in "${dataset_array[@]}";
do
  for loss in "${loss_array[@]}";
  do
    for model in "${model_array[@]}";
    do
      for tuner in "${tuner_array[@]}";
      do
        for reshape_type in "${reshape_type_array[@]}";
        do
          sbatch ASCAD_nas_attack_run.sh $dataset $loss $model $tuner $reshape_type
        done
      done
    done
  done
done
