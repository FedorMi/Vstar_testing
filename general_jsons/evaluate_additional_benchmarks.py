import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_file = "eval_result_additional_benchmarks.json"
path = os.path.join("general_jsons", json_file)


with open(path, "r") as f:
    file = json.load(f)
for bench in file:
    bench_json = file[bench]
    direct_attributes_correct = 0
    relative_position_correct = 0
    object_choice_correct = 0
    object_existence_correct = 0
    count = 0
    for j in bench_json:
        for k in bench_json[j]:
            if j == "direct_attributes":
                direct_attributes_correct += k["correct"]
            if j == "relative_position":
                relative_position_correct += k["correct"]
            if j == "object_choice":
                object_choice_correct += k["correct"]
            if j == "object_existence":
                object_existence_correct += k["correct"]
    overall_correct = direct_attributes_correct + relative_position_correct + object_choice_correct + object_existence_correct
    print(f"Results for {bench}:")
    print("direct_attributes:", direct_attributes_correct / 10 * 100, "%")
    print("relative_position:", relative_position_correct / 10 * 100, "%")
    print("object_choice:", object_choice_correct / 10 * 100, "%")
    print("object_existence:", object_existence_correct / 10 * 100, "%")
    print("overall:", overall_correct / 40 * 100, "%")
