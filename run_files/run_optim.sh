#!/bin/bash

#SBATCH --account=es_tang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=9:00:00
#SBATCH --job-name="optimization_true_testing"
#SBATCH --mem-per-cpu=16384
#SBATCH --mail-type=END

export PATH=/cluster/scratch/username/ollama/bin:$PATH

export OLLAMA_MODELS=/cluster/scratch/username/ollama/models

export OLPORT=11434
export OLLAMA_HOST=127.0.0.1:$OLPORT
export OLLAMA_BASE_URL="http://localhost:$OLPORT/v1"

ollama serve >ollama_$OLPORT.log 2>ollama_$OLPORT.err &

sleep 20

curl http://localhost:11434/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-r1:32b",
    "messages": [{"role": "user", "content": "Hello!"}]
}'

sleep 120

python model_testing_choices.py