declare -a dataset_array=("CHES_CTF")
declare -a loss_array=("categorical_crossentropy" "ranking_loss" "cross_entropy_ratio" "sigmoid_focal_binary_crossentropy" "sigmoid_focal_categorical_crossentropy" "sigmoid_focal_categorical_crossentropy_ratio" "sigmoid_focal_binary_crossentropy_ratio")
declare -a model_array=("nas_basic5")
declare -a tuner_array=("hyperband" "greedy" "random" "bayesian")
declare -a max_trials_array=(1000)
declare -a reshape_type_array=("2dCNNSqr" "2dCNNRect" "1dCNN")
declare -a leakage_models=("HW")
#declare -a leakage_models=("ID" "HW")

for dataset in "${dataset_array[@]}";
do
  for loss in "${loss_array[@]}";
  do
    for model in "${model_array[@]}";
    do
      for tuner in "${tuner_array[@]}";
      do
        for max_trials in "${max_trials_array[@]}";
        do
          for reshape_type in "${reshape_type_array[@]}";
          do
            for leakage_model in "${leakage_models[@]}";
            do
              sbatch CHES_CTF_nas_train_models_run.sh $dataset $loss $model $tuner $max_trials $reshape_type $leakage_model
            done
          done
        done
      done
    done
  done
done