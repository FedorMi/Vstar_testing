import os
import json
# open json file results_divided_testing.json
path = os.path.join("prompt_optim_outputs", "results_divided_testing.json")
with open(path, "r") as f:
    data = json.load(f)
deltas = []
for d in data:
    avg_delta = 0
    for g in data[d]:
        delta = data[d][g]["test_acc"][-1] - data[d][g]["test_acc"][0]
        avg_delta += delta
    avg_delta /= len(data[d])
    deltas.append(avg_delta)
print(deltas)