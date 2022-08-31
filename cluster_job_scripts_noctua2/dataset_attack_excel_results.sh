declare -a dataset_array=("ASCAD_desync0_variable" "ASCAD_desync50_variable" "ASCAD_desync100_variable" "ASCAD_desync0" "ASCAD_desync50" "ASCAD_desync100" "AES_HD" "AES_RD" "DP4CONTEST" "CHES_CTF")

for dataset in "${dataset_array[@]}";
do
  sbatch dataset_attack_excel_results_run.sh $dataset
done