import json
import os
import matplotlib.pyplot as plt
import numpy as np

temp = "eval_result_manual_results"
json_files = [temp + str(idx) + ".json" for idx in range(1, 11)]

for i in json_files:
    path = os.path.join("initial_ten", i)
    path = os.path.join("prompt_optim_outputs", path)
    name = i.split(".")[0].replace("eval_results_", "")
    with open(path, "r") as f:
        hyperparameters = json.load(f)
    direct_attributes_correct = 0
    relative_position_correct = 0
    direct_attributes_count = 0
    relative_position_count = 0
    count = 0
    for j in hyperparameters:
        for k in hyperparameters[j]:
            if j == "direct_attributes":
                direct_attributes_correct += k["correct"]
                direct_attributes_count += 1
            if j == "relative_position":
                relative_position_correct += k["correct"]
                relative_position_count += 1
    overall_correct = direct_attributes_correct + relative_position_correct
    overall_count = direct_attributes_count + relative_position_count
    print("direct_attributes:", direct_attributes_correct / direct_attributes_count * 100, "%")
    print("relative_position:", relative_position_correct / relative_position_count * 100, "%")
    print("overall:", overall_correct / overall_count * 100, "%")
