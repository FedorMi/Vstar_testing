import json
import os

json_files = list(filter(lambda file: '.json' in file, os.listdir("hyperparameter_jsons")))


for i in json_files:
    path = os.path.join("hyperparameter_jsons", i)
    with open(path, "r") as f:
        hyperparameters = json.load(f)
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
        # print percentage of correct
        print(i)
        print("direct_attributes:", direct_attributes_correct / direct_attributes_count * 100, "%")
        print("relative_position:", relative_position_correct / relative_position_count * 100, "%")
        print("overall:", overall_correct / overall_count * 100, "%")
    print("--------------------------------------------------")
