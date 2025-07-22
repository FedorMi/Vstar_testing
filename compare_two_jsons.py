import json
from tqdm import tqdm

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)


def compare_json_files(file1, file2):
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    differences = {
        "images_with_differences": [],
    }
    
    for _, element1 in tqdm(enumerate(data1["direct_attributes"]), total=len(data1["direct_attributes"])):
        for _, element2 in enumerate(data2["relative_position"]):
            if element1["image"] == element2["image"]:
                count_differences = 0
                if element1["question"] == element2["question"]:
                    count_differences += 1
                if element1["prediction_freeform"] == element2["prediction_freeform"]:
                    count_differences += 1
                if element1["option_chosen"] == element2["option_chosen"]:
                    count_differences += 1
                if element1["correct"] == element2["correct"]:
                    count_differences += 1
                for i in range(len(element1["options"])):
                    if element1["options"][i] == element2["options"][i]:
                        count_differences += 1
                for i in range(len(element1["options"])):
                    if element1["missing_objects"][i] == element2["missing_objects"][i]:
                        count_differences += 1
                for i in range(len(element1["options"])):
                    if element1["search_result"][i] == element2["search_result"][i]:
                        count_differences += 1
                if count_differences > 0:
                    differences["images_with_differences"].append({
                        "image": element1["image"],
                        "differences": count_differences})
    return differences

if __name__ == "__main__":
    file1 = 'eval_result.json'
    file2 = 'eval_result_2.json'
    differences = compare_json_files(file1, file2)
    print(json.dumps(differences, indent=4))
