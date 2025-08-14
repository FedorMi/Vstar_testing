import os
import json
import numpy as np
import textgrad as tg


from ollama import chat, ResponseError, pull
from typing import Callable, List, Any



def handle_llava_response(response: str):
        temp = response.strip().split(", ")
        out = []
        for i in temp:
            out.append(i.strip())
        return out

def call_ollama_model_image(model_name: str, prompt: str, input: str, image_name: str, test_type: str):
    #image_path = "sa_17.jpg"
    path = os.path.join("vbench", test_type, image_name)
    # Read the image content
    with open(path, 'rb') as image_file:
        image_data = image_file.read()
    if "<QUESTION>" in prompt:
        text = prompt.replace("<QUESTION>", input)
    else:
        text = f"{prompt} {input}"

    # Prepare the request payload
    try:
        response = chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': text, 'images':[image_data]}
            ]
        )
    except ResponseError as e:
        if e.status_code == 404:
            print('Model not found, pulling the model...')
            pull(model_name)
            response = chat(
                model=model_name,
                messages=[
                    {'role': 'user', 'content': text, 'images':[image_data]}
                ]
            )
        else:
            raise e
    temp_result = response['message']['content']
    temp_result = temp_result.strip().split("</think>")[-1].lower().strip().split("#objects: ")[-1]
    result = []
    if model_name == "llava:34b":
        result = handle_llava_response(temp_result)
    return result
def get_missing_object_labels_correct(annotation):
    return annotation['target_object']

def missing_objects_comparison():
    prompt = "You are a helpful assistant that provides the objects present in the question to the user. The relevant question is: <QUESTION>. Please extract and list the essential entities, concepts, and key objects mentioned in the question, separated by commas, in lowercase, without explanations or full sentences. Focus on identifying the most crucial elements, ignoring irrelevant details, and prioritize clarity over completeness while maintaining a balance between brevity and accuracy."
    overall_count, overall_recall_total, overall_precision_total = 0, 0, 0
    for i in ["direct_attributes", "relative_position"]:
        folder = os.path.join("vbench", i)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        specific_count, specific_recall_total, specific_precision_total = 0, 0, 0
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            annotation = json.load(open(annotation_path))
            question = annotation['question']
            true_positive, false_positive, false_negative = 0, 0, 0
            new_missing = call_ollama_model_image("llava:34b", prompt, question, image_file, i)
            correct_missing = annotation['target_object']
            for missing in new_missing:
                if missing in correct_missing:
                    true_positive += 1
                if missing not in correct_missing:
                    false_positive += 1
            for missing in correct_missing:
                if missing not in new_missing:
                    false_negative += 1
            # compute individual recall and precision
            individual_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            individual_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specific_precision_total += individual_precision
            specific_recall_total += individual_recall
            specific_count += 1
        overall_count += specific_count
        print(f"Missing objects for {i}:")
        print(f"Precision percentage: {specific_precision_total / specific_count * 100 if specific_count > 0 else 0}%")
        print(f"Recall percentage: {specific_recall_total / specific_count * 100 if specific_count > 0 else 0}%")
        overall_precision_total += specific_precision_total
        overall_recall_total += specific_recall_total
    print(f"Missing objects overall:")
    print(f"Precision percentage: {overall_precision_total / overall_count * 100 if overall_count > 0 else 0}%")
    print(f"Recall percentage: {overall_recall_total / overall_count * 100 if overall_count > 0 else 0}%")

if __name__ == "__main__":
    missing_objects_comparison()