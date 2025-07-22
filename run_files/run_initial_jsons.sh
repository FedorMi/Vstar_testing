#!/bin/bash

#SBATCH --account=es_tang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=9:00:00
#SBATCH --job-name="base_model_jsons"
#SBATCH --mem-per-cpu=16384
#SBATCH --mail-type=END

python initial_seal_testing_jsons.py

