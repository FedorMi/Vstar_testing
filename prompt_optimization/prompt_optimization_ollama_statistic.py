import os
import json
import torch
import random
import argparse
import concurrent
import numpy as np


from ollama import chat, ResponseError, pull
from typing import Callable, List, Any
from copy import deepcopy
from PIL import Image
from tqdm import tqdm

#from textgrad.tasks.base import DataLoader
from torch.utils.data import DataLoader, Dataset
from statistics import mean 

from vstar_bench_eval import VQA_LLM, expand2square, normalize_bbox
from visual_search_optim import parse_args, VSM, visual_search
from torch.utils.data import random_split


# this is a modification of the textrad prompt optimization notebook

vqa_llm = None
vsm = None
vsm_model_path = "craigwu/seal_vsm_7b"
minimum_size_scale = 4.0
minimum_size = 224

class MyVariable:
    def __init__(self, value, requires_grad=False, role_description=None):
        self.value = value
        self.requires_grad = requires_grad
        self.role_description = role_description
    def set_value(self, value):
        self.value = value
    def get_value(self):
        return self.value

class MyDataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __getitem__(self, index):
        return self.files[index]
    
    def __len__(self):
        return len(self.files)

def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
    y2 = min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    return inter_area/(bbox1[2]*bbox1[3]+bbox2[2]*bbox2[3]-inter_area)

def handle_deepseek_response(response: str):
        temp = response.strip().split(", ")
        out = []
        for i in temp:
            out.append(i.strip())
        return out
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
    elif model_name == "deepseek-r1:32b":
        result = handle_deepseek_response(temp_result)
    return result
def call_ollama_model(model_name: str, prompt: str, input:str):
    # Initialize the Ollama client
    if "<QUESTION>" in prompt:
        text = prompt.replace("<QUESTION>", input)
    else:
        text = f"{prompt} {input}"
    try:
        response = chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': text}
            ]
        )
    except ResponseError as e:
        if e.status_code == 404:
            print('Model not found, pulling the model...')
            pull(model_name)
            response = chat(
                model=model_name,
                messages=[
                    {'role': 'user', 'content': text}
                ]
            )
        else:
            raise e
    temp_result = response['message']['content']
    temp_result = temp_result.strip().split("</think>")[-1].lower().strip().split("#objects: ")[-1]
    result = []
    if model_name == "llava:34b":
        result = handle_llava_response(temp_result)
    elif model_name == "deepseek-r1:32b":
        result = handle_deepseek_response(temp_result)
    return result

def call_vqa_model(model_name: str, prompt: str, question: str, image_path: str) -> str:
    missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
    image = Image.open(image_path).convert('RGB')
    image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
    # generate free-form response to check whether visual search needs to be activated
    prediction = vqa_llm.free_form_inference(image, question)
    missing_objects = []
    if missing_objects_msg in prediction:
        missing_objects = prediction.split(missing_objects_msg)[-1]
        if missing_objects.endswith('.'):
            missing_objects = missing_objects[:-1]
        missing_objects = missing_objects.split(',')
        missing_objects = [missing_object.strip() for missing_object in missing_objects]
    return missing_objects
def get_missing_object_labels_correct(annotation):
    return annotation['target_object']
def get_bounding_boxes_seal(missing_objects, image_path, annotation, prompt_template):
    global vsm
    if vsm is None:
        vsm_args = parse_args({})
        vsm_args.version = vsm_model_path
        vsm = VSM(vsm_args)
    if "<LABEL>" in prompt_template:
        prompt_template = prompt_template.replace("<LABEL>", "{}")
    else:
        prompt_template = prompt_template + " {}"
    search_result = []
    if len(missing_objects) > 0:
        # visual search
        for object_name in missing_objects:
            #print("Searching for object:", object_name)
            image = Image.open(image_path).convert('RGB')
            smallest_size = max(int(np.ceil(min(image.width, image.height)/minimum_size_scale)), minimum_size)
            final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name, target_bbox=None, smallest_size=smallest_size, prompt=prompt_template)
            if all_valid_boxes is not None:
                # might exist multiple target instances
                for search_bbox in all_valid_boxes:
                    search_final_patch = final_step['bbox']
                    search_bbox[0] += search_final_patch[0]
                    search_bbox[1] += search_final_patch[1]
                    search_result.append({'bbox':search_bbox.tolist(),'name':object_name})
            else:
                search_bbox = final_step['detection_result']
                search_final_patch = final_step['bbox']
                search_bbox[0] += search_final_patch[0]
                search_bbox[1] += search_final_patch[1]
                search_result.append({'bbox':search_bbox.tolist(),'name':object_name})
    return search_result
def get_bounding_boxes_correct(missing_objects, image_path, annotation):
    search_result = []
    for idx, object_name in enumerate(missing_objects):
        search_result.append({'bbox': annotation['bbox'][idx], 'name': object_name})
    return search_result
def get_multiple_choice_seal(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template):
    global vqa_llm
    question_prompt = ""
    object_prompt = prompt_template
    object_prompt = object_prompt.replace("<LABEL>", "{}").replace("<BOUNDING_BOX>", "[{:.3f},{:.3f},{:.3f},{:.3f}]")
    # predict the multiple-choice option
    options = annotation['options']
    image = Image.open(image_path).convert('RGB')
    if len(missing_objects) > 0:
        object_names = [_['name'] for _ in search_result]
        bboxs = deepcopy([_['bbox'] for _ in search_result])
        if len(object_names) <= 2:
            images_long = [False]
            objects_long = [True]*len(object_names)
        else:
            images_long = [False]
            objects_long = [False]*len(object_names)
        object_crops = []
        for bbox in bboxs:
            object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=1.2)
            object_crops.append(object_crop)
        object_crops = torch.stack(object_crops, 0)
        image, left, top = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
        bbox_list = []
        for bbox in bboxs:
            bbox[0] += left
            bbox[1] += top
            bbox_list.append(bbox)
        bbox_list = [normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
        cur_focus_msg = focus_msg
        for i, (object_name, bbox) in enumerate(zip(object_names, bbox_list)):
            try:
                cur_focus_msg = cur_focus_msg + object_prompt.format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            except Exception as e:
                cur_focus_msg = cur_focus_msg + object_prompt + "{} [{:.3f},{:.3f},{:.3f},{:.3f}]".format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            if i != len(bbox_list)-1:
                cur_focus_msg = cur_focus_msg+"; "
            else:
                cur_focus_msg = cur_focus_msg +'.'
        question_with_focus = cur_focus_msg+"\n"+ question_prompt + question
        option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops, images_long=images_long, objects_long=objects_long)
    else:
        option_chosen = vqa_llm.multiple_choices_inference(image, question_prompt + question, options)
    correct = 1 if option_chosen==0 else 0
    return correct, options, options[option_chosen]

def get_multiple_choice_seal_standard(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template):
    global vqa_llm
    #focus_msg = prompt_template
    question_prompt = ""
    object_prompt = "<LABEL> <object> at location <BOUNDING_BOX>"
    object_prompt = object_prompt.replace("<LABEL>", "{}").replace("<BOUNDING_BOX>", "[{:.3f},{:.3f},{:.3f},{:.3f}]")
    # predict the multiple-choice option
    options = annotation['options']
    image = Image.open(image_path).convert('RGB')
    if len(missing_objects) > 0:
        object_names = [_['name'] for _ in search_result]
        bboxs = deepcopy([_['bbox'] for _ in search_result])
        if len(object_names) <= 2:
            images_long = [False]
            objects_long = [True]*len(object_names)
        else:
            images_long = [False]
            objects_long = [False]*len(object_names)
        object_crops = []
        for bbox in bboxs:
            object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=1.2)
            object_crops.append(object_crop)
        object_crops = torch.stack(object_crops, 0)
        image, left, top = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
        bbox_list = []
        for bbox in bboxs:
            bbox[0] += left
            bbox[1] += top
            bbox_list.append(bbox)
        bbox_list = [normalize_bbox(bbox, image.width, image.height) for bbox in bbox_list]
        cur_focus_msg = focus_msg
        for i, (object_name, bbox) in enumerate(zip(object_names, bbox_list)):
            try:
                cur_focus_msg = cur_focus_msg + object_prompt.format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            except Exception as e:
                cur_focus_msg = cur_focus_msg + object_prompt + "{} [{:.3f},{:.3f},{:.3f},{:.3f}]".format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            if i != len(bbox_list)-1:
                cur_focus_msg = cur_focus_msg+"; "
            else:
                cur_focus_msg = cur_focus_msg +'.'
        question_with_focus = cur_focus_msg+"\n"+ question_prompt + question
        option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops, images_long=images_long, objects_long=objects_long)
    else:
        option_chosen = vqa_llm.multiple_choices_inference(image, question_prompt + question, options)
    correct = 1 if option_chosen==0 else 0
    return correct, options, options[option_chosen]

def test_missing_objects(prompt_template, evaluation_set,with_image=True, model_name="llava:34b"):
    true_positive = 0
    count = 0
    for item in evaluation_set:
        image_file = item.split("$")[1]
        test_type = item.split("$")[0]
        folder = os.path.join("vbench", test_type)
        image_path = os.path.join(folder, image_file)
        annotation_path = image_path.split('.')[0] + '.json'
        annotation = json.load(open(annotation_path))
        question = annotation['question']
        real_missing_objects = annotation['target_object']
        result = []
        if with_image:
            result = call_ollama_model_image(model_name, prompt_template, question, image_file, test_type)
        else:
            result = call_ollama_model(model_name, prompt_template, question)
        #print("--------------------------")
        #print("result: ", result)
        #print("real_missing_objects: ", real_missing_objects)
        #print("--------------------------")
        not_found = False
        for i in real_missing_objects:
            if i not in result:
                not_found = True
                break
        if not not_found:
            true_positive += 1
        count += 1
    recall = true_positive / count if count > 0 else 0
    return recall
def test_bounding_boxes_iou(prompt_template, evaluation_set):
    global vsm
    global vqa_llm
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", type=str, default="final_call")
        parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
        parser.add_argument("--vqa-model-base", type=str, default=None)
        parser.add_argument("--conv_type", default="v1", type=str,)
        parser.add_argument("--benchmark-folder", type=str, default="vbench")
        parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
        parser.add_argument("--output-path", type=str, default="eval_result.json")
        parser.add_argument("--minimum_size_scale", default=4.0, type=float, help="minimum sub-image scale for the termination of search")
        parser.add_argument("--minimum_size", default=224, type=int, help="minimum sub-image size for the termination of search")
        args = parser.parse_args()
    if vqa_llm is None:
        vqa_llm = VQA_LLM(args)
    if vsm is None:
        vsm_args = parse_args({})
        vsm_args.version = vsm_model_path
        vsm = VSM(vsm_args)
    iou_total = 0
    count = 0
    for item in evaluation_set:
        image_file = item.split("$")[1]
        test_type = item.split("$")[0]
        folder = os.path.join("vbench", test_type)
        correct_data = json.load(open("eval_results_correct_bounding_boxes.json", 'r'))
        image_path = os.path.join(folder, image_file)
        annotation_path = image_path.split('.')[0] + '.json'
        image = Image.open(image_path).convert('RGB')
        annotation = json.load(open(annotation_path))
        image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
        question = annotation['question']
        missing_objects = get_missing_object_labels_correct(annotation)
        try:
            search_result = get_bounding_boxes_seal(missing_objects, image_path, annotation, prompt_template)
            temp = []
            for i in correct_data[test_type]:
                if i["image"] == image_file:
                    temp = i
                    break
            correct_data = temp
            correct_search_result = correct_data["search_result"]
            for i in search_result:
                missing_label = i["name"]
                missing_bbox = i["bbox"]
                for j in range(len(correct_search_result)):
                    if correct_search_result[j]["name"] == missing_label:
                        correct_bbox = correct_search_result[j]["bbox"]
                        iou_total += iou(missing_bbox, correct_bbox)
                        break
            count += len(search_result)
        except Exception as e:
            count += len(missing_objects)

    return iou_total / count if count > 0 else 0
def test_bounding_boxes_final_result(prompt_template, evaluation_set):
    global vsm
    global vqa_llm
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", type=str, default="final_call")
        parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
        parser.add_argument("--vqa-model-base", type=str, default=None)
        parser.add_argument("--conv_type", default="v1", type=str,)
        parser.add_argument("--benchmark-folder", type=str, default="vbench")
        parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
        parser.add_argument("--output-path", type=str, default="eval_result.json")
        parser.add_argument("--minimum_size_scale", default=4.0, type=float, help="minimum sub-image scale for the termination of search")
        parser.add_argument("--minimum_size", default=224, type=int, help="minimum sub-image size for the termination of search")
        args = parser.parse_args()
    if vqa_llm is None:
        vqa_llm = VQA_LLM(args)
    if vsm is None:
        vsm_args = parse_args({})
        vsm_args.version = vsm_model_path
        vsm = VSM(vsm_args)
    count= 0
    correct_total = 0
    focus_msg = "Additional visual information to focus on: "
    for item in evaluation_set:
        image_file = item.split("$")[1]
        test_type = item.split("$")[0]
        folder = os.path.join("vbench", test_type)
        image_path = os.path.join(folder, image_file)
        annotation_path = image_path.split('.')[0] + '.json'
        image = Image.open(image_path).convert('RGB')
        annotation = json.load(open(annotation_path))
        image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
        question = annotation['question']
        missing_objects = get_missing_object_labels_correct(annotation)
        try:
            search_result = get_bounding_boxes_seal(missing_objects, image_path, annotation, prompt_template)
            correct, _, _ = get_multiple_choice_seal_standard(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template)
            correct_total += correct
        except Exception as e:
            pass
        count += 1
    return correct_total / count if count > 0 else 0
def test_final_call(prompt_template, evaluation_set):
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", type=str, default="final_call")
        parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
        parser.add_argument("--vqa-model-base", type=str, default=None)
        parser.add_argument("--conv_type", default="v1", type=str,)
        parser.add_argument("--benchmark-folder", type=str, default="vbench")
        parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
        parser.add_argument("--output-path", type=str, default="eval_result.json")
        parser.add_argument("--minimum_size_scale", default=4.0, type=float, help="minimum sub-image scale for the termination of search")
        parser.add_argument("--minimum_size", default=224, type=int, help="minimum sub-image size for the termination of search")
        args = parser.parse_args()
    global vqa_llm
    global vsm
    # init VQA LLM
    if vqa_llm is None:
        vqa_llm = VQA_LLM(args)
    focus_msg = "Additional visual information to focus on: "
    initial_json = json.load(open('eval_results_initial_seal_testing.json', 'r'))
    count= 0
    correct_total = 0
    for item in evaluation_set:
        image_file = item.split("$")[1]
        test_type = item.split("$")[0]
        folder = os.path.join(args.benchmark_folder, test_type)
        initial_type_data = initial_json[test_type]
        initial_image_data = {}
        for i in initial_type_data:
            if i['image'] == image_file:
                initial_image_data = i
                break
        image_path = os.path.join(folder, image_file)
        annotation_path = image_path.split('.')[0] + '.json'
        annotation = json.load(open(annotation_path))
        question = annotation['question']
        missing_objects = initial_image_data['missing_objects']
        #print("missing objects: ", missing_objects)
        search_result = initial_image_data['search_result']
        correct, _, _ = get_multiple_choice_seal(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template)
        correct_total += correct
        count += 1
    return correct_total / count if count > 0 else 0

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def run_validation_revert(system_prompt: MyVariable, results, val_set, eval_func):
    try:
        val_performance = eval_func(system_prompt.get_value(), val_set)
    except Exception as e:
        print(f"Error during validation: {e}")
        val_performance = 0.0
    previous_performance = results["validation_acc"][-1]
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.get_value()}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance
        return False
    else:
        results["validation_acc"].append(val_performance)
        return True

def prompt_generator(model_name: str, prompt: str):
    # Initialize the Ollama client
    response = chat(
        model=model_name,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    result = response['message']['content']
    return result

def make_new_prompt_focus_message(prompt_template, loss, results, model="llama3:70b"):
    starting_text = "Create a new prompt based on the following previous prompt templates and their evaluation results:\n"
    starting_text = "The current prompt template is: " + prompt_template + "\n The results for that prompt were: " + str(loss) + "\n"
    for i in range(len(results["prompt"])):
        starting_text += f"Prompt {i}: {results['prompt'][i]}, with validation accuracy: {results['validation_acc'][i]}, and test accuracy: {results['test_acc'][i]}\n"
    starting_text += "Do not explain the prompt, do not explain the chain of thought. Only return the new prompt template, do not return any other text. The new prompt should be better than the previous one, take into account the previous prompts and their evaluation results.\n"
    starting_text += "Do not announce the prompt, do not present the prompt, only answer with the prompt, and do not invent any accuracy metric\n"
    new_prompt = prompt_generator(model, starting_text)
    return new_prompt

def make_new_prompt_object(prompt_template, loss, results, model="llama3:70b"):
    starting_text = "Create a new prompt based on the following previous prompt templates and their evaluation results:\n"
    starting_text = "The current prompt template is: " + prompt_template + "\n The results for that prompt were: " + str(loss) + "\n"
    for i in range(len(results["prompt"])):
        starting_text += f"Prompt {i}: {results['prompt'][i]}, with validation accuracy: {results['validation_acc'][i]}, and test accuracy: {results['test_acc'][i]}\n"
    starting_text += "Do not explain the prompt, do not explain the chain of thought. Only return the new prompt template, do not return any other text. The new prompt should be better than the previous one, take into account the previous prompts and their evaluation results.\n"
    starting_text += "The new prompt has to still contain the <LABEL>, <BOUNDING_BOX> and the <object> placeholders, but you can change the rest of the prompt.\n"
    starting_text += "Do not announce the prompt, do not present the prompt, only answer with the prompt, and do not invent any accuracy metric\n"
    new_prompt = prompt_generator(model, starting_text)
    return new_prompt

def make_new_prompt_question(prompt_template, loss, results, model = "llama3:70b"):
    starting_text = "Create a new prompt based on the following previous prompt templates and their evaluation results:\n"
    starting_text = "The current prompt template is: " + prompt_template + "\n The results for that prompt were: " + str(loss) + "\n"
    for i in range(len(results["prompt"])):
        starting_text += f"Prompt {i}: {results['prompt'][i]}, with validation accuracy: {results['validation_acc'][i]}, and test accuracy: {results['test_acc'][i]}\n"
    starting_text += "Do not explain the prompt, do not explain the chain of thought. Only return the new prompt template, do not return any other text. The new prompt should be better than the previous one, take into account the previous prompts and their evaluation results.\n"
    starting_text += "Do not announce the prompt, do not present the prompt, only answer with the prompt, and do not invent any accuracy metric\n"
    new_prompt = prompt_generator(model, starting_text)
    return new_prompt

def make_new_prompt_missing_object(prompt_template, loss, results, model = "llama3:70b"):
    starting_text = "Create a new prompt based on the following previous prompt templates and their evaluation results:\n"
    starting_text = "The current prompt template is: " + prompt_template + "\n The results for that prompt were: " + str(loss) + "\n"
    for i in range(len(results["prompt"])):
        starting_text += f"Prompt {i}: {results['prompt'][i]}, with validation accuracy: {results['validation_acc'][i]}, and test accuracy: {results['test_acc'][i]}\n"
    starting_text += "Do not explain the prompt, do not explain the chain of thought. Only return the new prompt template, do not return any other text. The new prompt should be better than the previous one, take into account the previous prompts and their evaluation results.\n"
    starting_text += "The new prompt has to still contain the <QUESTION> placeholder, but you can change the rest of the prompt.\n"
    starting_text += "Do not announce the prompt, do not present the prompt, only answer with the prompt, and do not invent any accuracy metric\n"
    new_prompt = prompt_generator(model, starting_text)
    return new_prompt

def make_new_prompt_bounding_box(prompt_template, loss, results, model = "llama3:70b"):
    starting_text = "Create a new prompt based on the following previous prompt templates and their evaluation results:\n"
    starting_text = "The current prompt template is: " + prompt_template + "\n The results for that prompt were: " + str(loss) + "\n"
    for i in range(len(results["prompt"])):
        starting_text += f"Prompt {i}: {results['prompt'][i]}, with validation accuracy: {results['validation_acc'][i]}, and test accuracy: {results['test_acc'][i]}\n"
    starting_text += "Do not explain the prompt, do not explain the chain of thought. Only return the new prompt template, do not return any other text. The new prompt should be better than the previous one, take into account the previous prompts and their evaluation results.\n"
    starting_text += "The new prompt has to still contain the <LABEL> placeholder, but you can change the rest of the prompt.\n"
    starting_text += "Do not announce the prompt, do not present the prompt, only answer with the prompt, and do not invent any accuracy metric\n"
    new_prompt = prompt_generator(model, starting_text)
    return new_prompt

def make_new_prompt_complete(prompt_template, loss, results, model = "llama3:70b"):
    starting_text = "Create a new prompt based on the following previous prompt templates and their evaluation results:\n"
    starting_text = "The current prompt template is: " + prompt_template + "\n The results for that prompt were: " + str(loss) + "\n"
    for i in range(len(results["prompt"])):
        starting_text += f"Prompt {i}: {results['prompt'][i]}, with validation accuracy: {results['validation_acc'][i]}, and test accuracy: {results['test_acc'][i]}\n"
    starting_text += "Do not explain the prompt, do not explain the chain of thought. Only return the new prompt template, do not return any other text. The new prompt should be better than the previous one, take into account the previous prompts and their evaluation results.\n"
    starting_text += "The new prompt has to still contain the <LABEL>, <BOUNDING_BOX> and the <object> placeholders, but you can change the rest of the prompt.\n"
    starting_text += "The new prompt should still be divided into three parts using the newline \"\\n\" symbol.\n"
    starting_text += "Do not announce the prompt, do not present the prompt, only answer with the prompt, and do not invent any accuracy metric\n"
    new_prompt = prompt_generator(model, starting_text)
    return new_prompt


def ollama_prompt_optimization(eval_func, data_set, starting_prompt: str, opti_model: str = "llama3:70b", prompt_gen_func: Callable = make_new_prompt_object, args = None):
    # Load the data and the evaluation function
    first_frac = 0.25
    second_frac = 0.25
    third_frac = 0.25
    first_len = int(len(data_set)*first_frac)      
    second_len = int(len(data_set)*second_frac)
    third_len = int(len(data_set)*third_frac)
    fourth_len = len(data_set) - first_len - second_len - third_len
    first_set, second_set, third_set, fourth_set = random_split(data_set, [first_len, second_len, third_len, fourth_len])
    set_of_sets = [first_set, second_set, third_set, fourth_set]
    init_prompt = starting_prompt
    out = {}
    for index_sets in range(len(set_of_sets)):
        data_set_frac = set_of_sets[index_sets]
        temp_out = {}
        for index in range(1,6):
            starting_prompt = init_prompt
            train_fraction = 0.33
            val_fraction = 0.34
            test_fraction = 1.0 - train_fraction - val_fraction
            train_len = int(len(data_set_frac)*train_fraction)      
            val_len = int(len(data_set_frac)*val_fraction)
            test_len = len(data_set_frac) - train_len - val_len
            train_set, val_set, test_set = random_split(data_set_frac, [train_len, val_len, test_len])
            #train_set = data_set
            #val_set = data_set
            #test_set = data_set
            #train_set, val_set, test_set, eval_fn = load_task("BBH_object_counting", evaluation_api=llm_api_eval)
            data_set_train = MyDataset(train_set)
            train_loader = DataLoader(data_set_train, batch_size=12, shuffle=True)
            results = {"test_acc": [], "prompt": [], "validation_acc": []}
            print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
            if isinstance(starting_prompt, str):
                # Testing the 0-shot performance of the evaluation engine
                system_prompt = MyVariable(starting_prompt, 
                                            requires_grad=True,
                                            role_description="prompt to the model to answer the VQA task")
                

                #results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
                #results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))

                results["test_acc"].append(eval_func(system_prompt.get_value(), test_set))
                results["validation_acc"].append(eval_func(system_prompt.get_value(), val_set))
                results["prompt"].append(system_prompt.get_value())
                
                print("Initial test accuracy: ", results["test_acc"][-1])
                print("Initial validation accuracy: ", results["validation_acc"][-1])
                print("Initial prompt: ", results["prompt"][-1])
            else:
                for i in range(len(starting_prompt)):
                    system_prompt = MyVariable(starting_prompt, 
                                            requires_grad=True,
                                            role_description="prompt to the model to answer the VQA task")
                

                    #results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
                    #results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))

                    results["test_acc"].append(eval_func(system_prompt.get_value(), test_set))
                    results["validation_acc"].append(eval_func(system_prompt.get_value(), val_set))
                    results["prompt"].append(system_prompt.get_value())
                highest_validation = 0
                highest_idx = -1
                for i in range(len(results["prompt"])):
                    if results["validation_acc"][i] > highest_validation:
                        highest_validation = results["validation_acc"][i]
                        highest_idx = i
                print("Initial test accuracy: ", results["test_acc"][highest_idx])
                print("Initial validation accuracy: ", results["validation_acc"][highest_idx])
                print("Initial prompt: ", results["prompt"][highest_idx])
                system_prompt = MyVariable(results["prompt"][highest_idx], 
                                            requires_grad=True,
                                            role_description="prompt to the model to answer the VQA task")

            # Training loop
            for epoch in range(30):
                for steps, batch_x in enumerate((pbar := tqdm(train_loader, position=0))):
                    #print(batch_x)
                    pbar.set_description(f"Training step {steps}. Epoch {epoch}")
                    
                    eval_output_variable = eval_func(system_prompt.get_value(), batch_x)
                    total_loss = eval_output_variable

                    system_prompt = prompt_gen_func(system_prompt.get_value(), total_loss, results, model=opti_model)
                    system_prompt = MyVariable(system_prompt, 
                                        requires_grad=True,
                                        role_description="prompt to the model to answer the VQA task")

                    better = run_validation_revert(system_prompt, results, val_set, eval_func)
                    
                    print("sys prompt: ", system_prompt.get_value())
                    if better:
                        test_acc = eval_func(system_prompt.get_value(), test_set)
                        results["test_acc"].append(test_acc)
                        print("test_acc: ", test_acc)
                        results["prompt"].append(system_prompt.get_value())
                    if steps == 100:
                        break
            temp_out[index] = results
        out[index_sets] = temp_out
    # save results
    with open("results_divided_testing.json", "w") as f:
        json.dump(out, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="final_call")
    args = parser.parse_args()
    data_set = []
    for test_type in ['direct_attributes', 'relative_position']:
        folder = os.path.join("vbench", test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for image_file in image_files:
            image_path = test_type + "$" + image_file
            data_set.append(image_path)
    prompt = """You are a helpful assistant that provides the objects present in the question to the user. 
                You do not give explanations, you don't respond in full sentences, you only respond with objects. The relevant question is: <QUESTION>. 
                Do not answer the question, only provide the objects that are relevant to the question. 
                The objects should be separated by commas, and the objects should be in lowercase."""
    #prompt = "You are a helpful assistant that provides the objects present in the question to the user. The relevant question is: <QUESTION>. Please identify and list the main entities, concepts, and key objects mentioned in the question, separated by commas, in lowercase, without explanations or full sentences. Focus on extracting the most important elements, ignoring irrelevant details, and prioritize clarity over completeness."
    #prompt = "You are a helpful assistant that provides the objects present in the question to the user. The relevant question is: <QUESTION>. Please extract and list the essential entities, concepts, and key objects mentioned in the question, separated by commas, in lowercase, without explanations or full sentences. Focus on identifying the most crucial elements, ignoring irrelevant details, and prioritize clarity over completeness while maintaining a balance between brevity and accuracy."
    func_to_give = test_missing_objects
    optim_model_name = "llama3:70b"
    #optim_model_name = "llama3:8b"
    prompt_gen_func = make_new_prompt_missing_object
    if args.experiment == "bbox_iou":
        prompt = "Please locate the <LABEL> in this image."
        func_to_give = test_bounding_boxes_iou
        optim_model_name = "llama3:70b_box_iou"
        #optim_model_name = "llama3:8b_box_iou"
        prompt_gen_func = make_new_prompt_bounding_box
    elif args.experiment == "bbox_final":
        prompt = "Please locate the <LABEL> in this image."
        func_to_give = test_bounding_boxes_final_result
        optim_model_name = "llama3:70b_box_final"
        #optim_model_name = "llama3:8b_box_final"
        prompt_gen_func = make_new_prompt_bounding_box
    elif args.experiment == "final_call":
        prompt = "<LABEL> <object> at location <BOUNDING_BOX>"
        #prompt = []
        #with open("prompts.json") as f:
        #    data = json.load(f)
        #    for item in data:
        #        data[item]["objects"]

        #prompt = "The model should consider this: <object> refers to <LABEL>, located at <BOUNDING_BOX>"
        #prompt = "<object> is located at <BOUNDING_BOX> and represents a <LABEL>."
        #prompt = "In the image, a <object> is shown as a <LABEL> at location <BOUNDING_BOX>"
        #prompt = "The <LABEL> depicted is a <object> situated at <BOUNDING_BOX>."

        #prompt = "The Question: "
        #prompt = "Please clarify or rephrase the question:"

        #prompt = "Please answer the following Question: "
        #prompt = "Please provide a response to the following question:"

        #prompt = "Additional visual information to focus on: "
        #prompt = "Specific regions of interest to highlight in images"
        #prompt = "Identifying key objects or features within images to facilitate targeted analysis or processing."

        #prompt = "Additional visual information to focus on: \n"
        #prompt += "<LABEL> <object> at location <BOUNDING_BOX>\n"
        #prompt += "The Question: "

        optim_model_name = "llama3:70b_final_call"
        #optim_model_name = "llama3:8b_final_call"
        func_to_give = test_final_call
        prompt_gen_func = make_new_prompt_object

    ollama_prompt_optimization(func_to_give, data_set, prompt, optim_model_name, prompt_gen_func, args)