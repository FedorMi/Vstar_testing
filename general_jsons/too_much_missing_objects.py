import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_files = ["eval_result_additional_missing_objects.json", "eval_result_additional_choice_missing_objects.json"]
json_files = [os.path.join("general_jsons", file) for file in json_files if os.path.isfile(os.path.join("general_jsons", file))]

i1 = json_files[0]
i2 = json_files[1]

with open(i1, "r") as f:
    add1 = json.load(f)
with open(i2, "r") as f:
    add2 = json.load(f)
overall_list1 = [0] * 7
overall_list2 = [0] * 7
x_axis_list = range(7)
overall_count = 191
for i in ["direct_attributes", "relative_position"]:
    for fil in range(2):
        curr_list = overall_list1 if fil == 0 else overall_list2
        curr_file = add1 if fil == 0 else add2
        temp_list = []
        for idexl in range(7):
            correct = 0
            count = 0
            for j in curr_file["run_"+str(idexl)][i]:
                correct += j["correct"]
                count += 1
            temp_list.append(correct / count * 100)
            curr_list[idexl] += correct
        name = "additional choice" if fil == 1 else "additional"
        plt.plot(x_axis_list, temp_list, label=i.replace("_", " ").capitalize()+ " " + name.replace("_", " ").capitalize())

    plt.xlabel("Additional Missing Objects")
    plt.ylabel("Percentage Correct")
    plt.title("Additional Missing Objects Results on " + i.replace("_", " ").capitalize())
    plt.legend()
    save_path = os.path.join("additional_plots", f"{i}.png")
    plt.savefig(save_path)
    plt.clf()
overall_list1 = [x / overall_count * 100 for x in overall_list1]
overall_list2 = [x / overall_count * 100 for x in overall_list2]
plt.plot(x_axis_list, overall_list1, label="Overall Additional Missing Objects")
plt.plot(x_axis_list, overall_list2, label="Overall Additional Choice Missing Objects")
plt.xlabel("Additional Missing Objects")
plt.ylabel("Percentage Correct")
plt.title("Additional Missing Objects Results Overall")
plt.legend()
save_path = os.path.join("additional_plots", f"overall.png")
plt.savefig(save_path)
plt.clf()
print("--------------------------------------------------")
