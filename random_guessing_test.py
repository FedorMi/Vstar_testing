import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_file = os.path.join("general_jsons", "eval_results_initial_seal_testing.json")


with open(json_file, "r") as f:
    hyperparameters = json.load(f)
result = 0
count = 0
for j in hyperparameters:
    count_specific = 0
    result_specific = 0
    for k in hyperparameters[j]:
        temp = 1.0/len(k["options"])
        count_specific += 1
        result_specific += temp
    count += count_specific
    result += result_specific
    print("Average result: ", result_specific / count_specific * 100, "% for ", j)
print("Overall average result: ", result / count * 100, "%")
