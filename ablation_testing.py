import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_files = list(filter(lambda file: '.json' in file, os.listdir("hyperparameter_jsons")))


for i in json_files:
    path = os.path.join("hyperparameter_jsons", i)
    name = i.split(".")[0].replace("parameter_ablation", "") 
    with open(path, "r") as f:
        hyperparameters = json.load(f)
    overall_list = []
    relative_position_list = []
    direct_attributes_list = []
    x_axis_list = []
    for i in hyperparameters:
        direct_attributes_correct = 0
        relative_position_correct = 0
        direct_attributes_count = 0
        relative_position_count = 0
        count = 0
        for j in hyperparameters[i]:
            for k in hyperparameters[i][j]:
                if j == "direct_attributes":
                    direct_attributes_correct += k["correct"]
                    direct_attributes_count += 1
                if j == "relative_position":
                    relative_position_correct += k["correct"]
                    relative_position_count += 1
        overall_correct = direct_attributes_correct + relative_position_correct
        overall_count = direct_attributes_count + relative_position_count
        overall_list.append(overall_correct / overall_count * 100)
        relative_position_list.append(relative_position_correct / relative_position_count * 100)
        direct_attributes_list.append(direct_attributes_correct / direct_attributes_count * 100)
        if name != "confidence":
            x_axis_list.append(float(i.replace(name, "").replace("_", "")))
        # print percentage of correct
        print(i)
        print("direct_attributes:", direct_attributes_correct / direct_attributes_count * 100, "%")
        print("relative_position:", relative_position_correct / relative_position_count * 100, "%")
        print("overall:", overall_correct / overall_count * 100, "%")
    if name != "confidence":
        plt.plot(x_axis_list, overall_list, label="Overall")
        plt.plot(x_axis_list, relative_position_list, label="Relative Position")
        plt.plot(x_axis_list, direct_attributes_list, label="Direct Attributes")
        plt.xlabel("Ablation Step")
        plt.ylabel("Percentage Correct")
        plt.title("Hyperparameter Ablation Results on " + name.replace("_", " ").capitalize())
        plt.legend()
        save_path = os.path.join("hyperparameter_plots", f"{name}_ablation_results.png")
        plt.savefig(save_path)
        plt.clf()
    print("--------------------------------------------------")
