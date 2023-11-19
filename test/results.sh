#!/bin/bash

# Define the dataset IDs and their corresponding names
declare -A datasets
datasets[1044]="eye_movements"
datasets[4538]="gesture"
datasets[1477]="gas_concentration"

# Define the list of models
models=("lstm" "gru" "mlp" "bilstm" "bigru" "cnn")

# Generate the argument list for all models and datasets
args=()
for model in "${models[@]}"; do
    for id in "${!datasets[@]}"; do
        name=${datasets[$id]}
        args+=("$id $name --model=$model")
    done
done

# Export datasets array to be available in subshells
export -A datasets

# Run the jobs in parallel
# Adjust the -j parameter according to your system's capabilities
printf "%s\n" "${args[@]}" | parallel -j 2 --colsep ' ' python3 open_ml.py {1} {2} {3}

