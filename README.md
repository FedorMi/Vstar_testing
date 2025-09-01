# Overview

This repository contains the important experiments conducted during the Master Thesis: "Exploring Modular Guided Visual Search and Prompt Optimization on Visual Question Answering Tasks"

Most experiments in this repositor require the setup of the vstar repository with it
"https://github.com/penghao-wu/vstar" with it's V* benchmark "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain" saved as the folder "vbench". When setting up the vstar repository on the Euler cluster, the library "accelerate" has to be updated:

```console
pip install -U accelerate
```

The ollama library is necessary for the local hosting of the models, and the setup of ollama on the Euler cluster can be done by following the following tutorial "https://gist.github.com/ParthS007/e7db866b424fe112383f9c25c526d2bd", that Parth Shandilya and me have created together. 

For textgrad to work, specific port hosting can be done through:

```console
export OLPORT=11434
export OLLAMA_HOST=127.0.0.1:$OLPORT
export OLLAMA_BASE_URL="http://localhost:$OLPORT/v1"

ollama serve >ollama_$OLPORT.log 2>ollama_$OLPORT.err &
```

To test if the hosting works, and the models are available through the OpenAI api, the following test with the deepseek-r1 can be done, if it is installed:

```console
curl http://localhost:11434/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-r1:32b",
    "messages": [{"role": "user", "content": "Hello!"}]
}'

```

The test execution python files contain the option to run them, by giving them arguments, to run the specific experiment.

# Initial SEAL testing

The initial evaluation done on SEAL was in "initial_seal_testing_generate_jsons.py". The tests done on the additional benchmarks and tests with additional missing object labels are in the folder "specific_vbench_tests".

The additional small benchmarks for testing SEAL on new data are "vbench_additions" and "vbench_surgery".

The results and evalutation of those can be found in the folder "general_jsons" with the relevant plots being in the "plots/additional_plots" folder

# Hyperparameter Ablation

The hyperparameter ablation has been done through "hyperparameter_ablation.py" and the helper file "visual_search_parameter_ablation.py". 

The results and their evaluation can be found in the folder "hyperparameter_jsons", with the plots being in the folder "plots/hyperparameter_plots".

# Prompt Optimization

The final prompt optimization setup tested can be found "prompt_optimization_ollama.py", but experiments with textgrad, the freeform setup and the prompt optimization specifc setup, and specific setups, aiming to isolate the missing object label detection, can be found in the folder "prompt_optimization". The initial experiments with manual prompt midification and the ten initial prompts can be found in the folder "specific_vbench_tests".

If experiments to do not require textgrad or vstar files to run, it is recommended to have a separate python environment not containing the libraries required for these repositories, newer versions of libraries can make certain experiments significantly faster.

When setting up textgrad and SEAL together on the Euler cluster, the ollama, textgrad, and the updating of the accelerate library shouldn't be forgotten. If issues are encountered during setup, the sequence of installing the libraries should be changed.


# Visual Search Replacement Testing

The Comparison of V* and SEAL to other algorithms and models has been done in the folder "visual_search_replacements"

Experiments using Instruction Guided Visual Masking "https://github.com/2toinf/IVM/tree/master", or Grounding DINO "https://github.com/IDEA-Research/GroundingDINO", to generate the crops or bounding boxes require a separate environment setup with the repositories cloned for each and using the saved bounding boxes or crops the final evaluation can be made. If IVM does not propperly work, updating the libraries should resolve the issue.

For testing Gemini visual grounding, an api key has to be input first.

The attempted gamification of the zooming can be found in "visual_search_replacements/gamify_zooming.py"

# Common Issues

The setting up of the python environment with the github repositories together results in conflicting versions and things often not working. Grounding DINO was not able to be set up on the Euler cluster, and therefore was run on a personal device. If possible, environments combining the least amount of repository specific libraries should be used.

Running the base SEAL file on the V* benchmark using a V100 with 32GB VRAM or A100 40GB VRAM gpu will take on the Euler cluster around 4 hours and 30 minutes. Using an A100 80GB gpu on the Euler claster will take about 40 minutes to run. GPUs with less VRAM will not work. If on any experiment the experiment does not work, the issue can be often resolved by utilizing a bigger GPU. Most experiments on Prompt Optimization should be done using an A100 80GB VRAM GPU, to fit all the models on to it during runtime.

# Acknowledgement
Thank you to my Master Thesis supervisors Prof. Dr. Siyu Tang and Dr. Sergey Prokudin.