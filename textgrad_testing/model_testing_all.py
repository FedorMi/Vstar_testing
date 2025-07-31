import ollama
from object_recall_grad_descent_ollama import optimize_prompt_ollama, optimize_prompt_ollama_two, optimize_prompt_ollama_last_call, optimize_prompt_ollama_image_first
import json
import os
from ollama import chat, ResponseError, pull 
#from vbench_testing import test_model_multiple_choice

import argparse
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_object_token

from visual_search import parse_args, VSM, visual_search
from vstar_bench_eval import VQA_LLM, expand2square, normalize_bbox


vqa_llm = None
vsm = None

def get_missing_object_labels_correct(image, question, missing_objects_msg, annotation):
    return annotation['target_object'], "no prediction needed, the object labels are correct"

def get_bounding_boxes_seal(missing_objects, image_path, args, annotation, prompt_template):
    search_result = []
    if len(missing_objects) > 0:
        # visual search
        for object_name in missing_objects:
            image = Image.open(image_path).convert('RGB')
            smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
            final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name, target_bbox=None, smallest_size=smallest_size)
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

def get_bounding_boxes_correct(missing_objects, image_path, args, annotation):
    search_result = []
    for idx, object_name in enumerate(missing_objects):
        search_result.append({'bbox': annotation['bbox'][idx], 'name': object_name})
    return search_result

def get_multiple_choice_seal(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template):
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
            cur_focus_msg = cur_focus_msg + "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]".format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            if i != len(bbox_list)-1:
                cur_focus_msg = cur_focus_msg+"; "
            else:
                cur_focus_msg = cur_focus_msg +'.'
        question_with_focus = cur_focus_msg+"\n"+question
        option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops, images_long=images_long, objects_long=objects_long)
    else:
        option_chosen = vqa_llm.multiple_choices_inference(image, question, options)
    correct = 1 if option_chosen==0 else 0
    return correct, options, options[option_chosen]

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
def call_ollama_model_image(model_name: str, prompt: str, input: str, image_name: str):
    #image_path = "sa_17.jpg"
    path = os.path.join("vbench", "direct_attributes", image_name)
    # Read the image content
    with open(path, 'rb') as image_file:
        image_data = image_file.read()

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

def test_missing_objects(prompt_template,with_image=True, model_name="llava:34b"):
    true_positive = 0
    count = 0
    for test_type in ['direct_attributes', 'relative_position']:
        folder = os.path.join("vbench", test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for image_file in tqdm(image_files):
            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            annotation = json.load(open(annotation_path))
            question = annotation['question']
            real_missing_objects = annotation['target_object']
            result = []
            if with_image:
                result = call_ollama_model_image(model_name, prompt_template, question, image_file)
            else:
                result = call_ollama_model(model_name, prompt_template, question)
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
            

def test_bounding_boxes_iou():
    # init VQA LLM
    vqa_llm = VQA_LLM(args)
    # init VSM
    vsm_args = parse_args({})
    vsm_args.version = args.vsm_model_path
    vsm = VSM(vsm_args)

    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []

    missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
    focus_msg = "Additional visual information to focus on: "
    for test_type in ['direct_attributes', 'relative_position']:
        results[test_type] = []
        folder = os.path.join(args.benchmark_folder, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for image_file in tqdm(image_files):
            result_single_sample = {}
            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            image = Image.open(image_path).convert('RGB')
            annotation = json.load(open(annotation_path))
            image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
            
            question = annotation['question']
def test_bounding_boxes_final_result():
    pass
def test_final_call(prompt_template):
    # init VQA LLM
    if vqa_llm is None:
        vqa_llm = VQA_LLM(args)
    results = {}
    focus_msg = "Additional visual information to focus on: "
    initial_json = json.load(open('eval_results_initial_seal_testing.json', 'r'))
    for test_type in ['direct_attributes', 'relative_position']:
        results[test_type] = []
        folder = os.path.join(args.benchmark_folder, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        initial_type_data = initial_json[test_type]
        for image_file in tqdm(image_files):
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
            search_result = initial_image_data['search_result']
            get_multiple_choice_seal(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template)

def main_test_objects(initial_prompt, with_image):
    # open the eval_result_correct_objects.json file
    with open('eval_result_correct_objects.json', 'r') as f:
        data = json.load(f)
    question_list = []
    image_list = []
    missing_object_list = []
    iteri = 0
    for item in tqdm(data["direct_attributes"]):
        if iteri > 19:
            break
        question = item["question"]
        image = item["image"]
        missing_object = item["missing_objects"]
        iteri += 1
        #question_list.append("Return your answer with '#OBJECTS: ' at the beginning. Do not return logical reasoning or other comments. The question is: " + question)
        question_list.append(question)
        image_list.append(image)
        missing_object_list.append(missing_object)

    #initial_prompt = "You are a helpful assistant. To answer the question you need focus on objects in the question. Return only the objects to focus on."
    # use optimise_prompt to optimize the prompt
    optimized_prompt = ""
    if with_image:
        optimized_prompt = optimize_prompt_ollama_two(
            model_fn=call_ollama_model_image,
            model_name="llava:34b",
            initial_prompt=initial_prompt,
            input_set=question_list,
            image_input_set=image_list,
            expected_output_set=missing_object_list,
            steps=8
        )
    else:
        optimized_prompt = optimize_prompt_ollama(
            model_fn=call_ollama_model,
            model_name="llava:34b",
            initial_prompt=initial_prompt,
            input_set=question_list,
            expected_output_set=missing_object_list,
            steps=8
        )
    if False:
        #save the optimized prompt to a file
        with open('optimized_prompt_text.txt', 'w') as f:
            f.write(optimized_prompt)
    return optimized_prompt
def main_test_objects_optimise(with_image=True):
    # initial setup
    # open the eval_result_correct_objects.json file
    with open('eval_result_correct_objects.json', 'r') as f:
        data = json.load(f)
    typer = "direct_attributes"
    image_list_test = []
    expected_output_test = []
    question_list_test = []
    iteri = 0
    for item in tqdm(data[typer]):
        image = item["image"]
        missing_object_list = item["missing_objects"]
        iteri += 1
        image_list_test.append(image)
        question_list_test.append(item["question"])
        expected_output_test.append(missing_object_list)
    #initial prompt
    prompt = "You are a helpful assistant. Do not answer the question, do not write any lengthy explanations, only provide the label for the objects relevant to the question, the full object with adjectives, as mentioned in the question."
    while True:
        # use optimise_prompt to optimize the prompt
        optimized_prompt = main_test_objects(
            initial_prompt=prompt,
            with_image=with_image
        )
        #test the optimized prompt
        model_name = "llava:34b"
        expected_output_set = expected_output_test
        #compute recall for the optimized prompt
        true_positive = 0
        false_negative = 0
        for i in range(len(image_list_test)):
            image_test = image_list_test[i]
            missing_objects = expected_output_set[i]
            if with_image:
                result = call_ollama_model_image(model_name, optimized_prompt, question_list_test[i], image_test)
            else:
                result = call_ollama_model(model_name, optimized_prompt, question_list_test[i])
            found = True
            for i in missing_objects:
                if i not in result:
                    found = False
                    break
            if found:
                true_positive += 1
            else:
                false_negative += 1
            
        #compute recall
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        print(f"Optimized prompt: {optimized_prompt}")
        print(f"Recall: {recall:.2f}")
        if recall > 0.8:
            # save the optimized prompt to a file
            with open('optimized_prompt_text.txt', 'w') as f:
                f.write(optimized_prompt)
            break
        prompt = optimized_prompt

def main_test_seal_choice(initial_prompt):
    # open the eval_result_correct_objects.json file
    with open('eval_result_correct_objects.json', 'r') as f:
        data = json.load(f)
    typer = "direct_attributes"
    type_list = []
    image_list = []
    expected_output = []
    iteri = 0
    for item in tqdm(data[typer]):
        if iteri > 10:
            break
        image = item["image"]
        iteri += 1
        image_list.append(image)
        type_list.append(typer)
        expected_output.append("correct answer")
    # use optimise_prompt to optimize the prompt
    optimized_prompt = optimize_prompt_ollama_two(
        model_fn=test_model_multiple_choice,
        model_name="llava:34b",
        initial_prompt=initial_prompt,
        input_set=type_list,
        image_input_set=image_list,
        expected_output_set=expected_output,
        steps=1
    )
    #save the optimized prompt to a file
    if False:
        with open('optimized3.txt', 'w') as f:
            f.write(optimized_prompt)
    return optimized_prompt
def main_test_seal_choice_optimise():
    # initial setup
    # open the eval_result_correct_objects.json file
    with open('eval_result_correct_objects.json', 'r') as f:
        data = json.load(f)
    typer = "direct_attributes"
    type_list_test = []
    image_list_test = []
    expected_output_test = []
    iteri = 0
    for item in tqdm(data[typer]):
        image = item["image"]
        iteri += 1
        image_list_test.append(image)
        type_list_test.append(typer)
        expected_output_test.append("correct answer")

    
    #initial prompt
    #prompt = "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]"
    prompt = "<object_name> <object> at location <bounding_box>"
    while True:
        # use optimise_prompt to optimize the prompt
        optimized_prompt = main_test_seal_choice(
            initial_prompt=prompt
        )
        #test the optimized prompt
        model_name = "llava:34b"
        expected_output_set = expected_output_test
        #compute recall for the optimized prompt
        total_correct = 0
        total_tests = 0
        for i in range(len(image_list_test)):
            test_type = type_list_test[i]
            image_file = image_list_test[i]
            missing_objects = expected_output_set[i]
            result = test_model_multiple_choice(model_name, optimized_prompt, test_type, image_file)
            total_tests += 1
            total_correct += result == missing_objects
        #compute percentage correct
        percentage_correct = total_correct / total_tests
        print(f"Optimized prompt: {optimized_prompt}")
        print(f"Percentage correct: {percentage_correct:.2f}")
        if percentage_correct > 0.8:
            # save the optimized prompt to a file
            with open('optimized_prompt_text.txt', 'w') as f:
                f.write(optimized_prompt)
            break
        prompt = optimized_prompt

def main_test_seal_bbox(initial_prompt):
    # open the eval_result_correct_objects.json file
    with open('eval_result_correct_objects.json', 'r') as f:
        data = json.load(f)
    typer = "direct_attributes"
    type_list = []
    image_list = []
    expected_output = []
    iteri = 0
    for item in tqdm(data[typer]):
        if iteri > 9:
            break
        image = item["image"]
        image_json_path = os.path.join("vbench", typer, image.split("/")[-1].split(".")[0]+".json")
        #open the json file and get the question
        with open(image_json_path, 'r') as json_file:
            json_data = json.load(json_file)
        curr_focus_msg = ""
        prompt_use = {"pre_object": "The model should consider this: <object> refers to ", "mid_object": ", located at ", "post_object": ".", "question": "Please answer: "}
        for iteri, objs in enumerate(json_data["target_object"]):
            bbox = json_data["bbox"][iteri]
            curr_focus_msg = curr_focus_msg + prompt_use["pre_object"] + "{}".format(objs) + prompt_use["mid_object"] + "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(bbox[0], bbox[1], bbox[2], bbox[3]) + prompt_use["post_object"]
            if iteri != len(json_data["target_object"])-1:
                curr_focus_msg = curr_focus_msg+"; "
            else:
                curr_focus_msg = curr_focus_msg +'.'
        if curr_focus_msg == "":
            curr_focus_msg = "no missing objects"
        iteri += 1
        image_list.append(image)
        expected_output.append(curr_focus_msg)
        type_list.append(typer)

    
    # use optimise_prompt to optimize the prompt
    optimized_prompt = optimize_prompt_ollama_two(
        model_fn=test_model_multiple_bbox,
        model_name="llava:34b",
        initial_prompt=initial_prompt,
        input_set=type_list,
        image_input_set=image_list,
        expected_output_set=expected_output,
        steps=3
    )
    #save the optimized prompt to a file
    if False:
        with open('optimized_prompt_text.txt', 'w') as f:
            f.write(optimized_prompt)
    return optimized_prompt
def main_test_seal_bbox_optimise():
    # initial setup
    # open the eval_result_correct_objects.json file
    with open('eval_result_correct_objects.json', 'r') as f:
        data = json.load(f)
    typer = "direct_attributes"
    type_list_test = []
    image_list_test = []
    expected_output_test = []
    iteri = 0
    for item in tqdm(data[typer]):
        image = item["image"]
        image_json_path = os.path.join("vbench", typer, image.split("/")[-1].split(".")[0]+".json")
        #open the json file and get the question
        with open(image_json_path, 'r') as json_file:
            json_data = json.load(json_file)
        result = {}
        for iteri, objs in enumerate(json_data["missing_objects"]):
            bbox = json_data["bbox"][iteri]
            result.append({'bbox':bbox,'name':objs})
        iteri += 1
        image_list_test.append(image)
        expected_output_test.append(result)
        type_list_test.append(typer)

    
    #initial prompt
    #prompt = "{} <object> at location [{:.3f},{:.3f},{:.3f},{:.3f}]"
    prompt = "Please locate the {} in this image."
    while True:
        # use optimise_prompt to optimize the prompt
        optimized_prompt = main_test_seal_bbox(
            initial_prompt=prompt
        )
        #test the optimized prompt
        model_name = "llava:34b"
        expected_output_set = expected_output_test
        #compute recall for the optimized prompt
        total_correct = 0
        total_tests = 0
        for i in range(len(image_list_test)):
            test_type = type_list_test[i]
            image_file = image_list_test[i]
            bounding_boxes = expected_output_set[i]
            result = test_model_multiple_choice(model_name, optimized_prompt, test_type, image_file, retrieve_bounding_boxes=True)
            total_tests += 1
            total_correct += iou(result, bounding_boxes) > 0.5
        #compute percentage correct
        percentage_correct = total_correct / total_tests
        print(f"Optimized prompt: {optimized_prompt}")
        print(f"Percentage correct: {percentage_correct:.2f}")
        if percentage_correct > 0.8:
            # save the optimized prompt to a file
            with open('optimized_prompt_text.txt', 'w') as f:
                f.write(optimized_prompt)
            break
        prompt = optimized_prompt

def test_model_multiple_choice(model_name, prompt_part, test_type, image_file, vqa_do=False, missing_objects=[]):
    #image_file = image_test.split("$")[0]
    #test_type = image_test.split("$")[1]

    parser = argparse.ArgumentParser()
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
    # init VSM
    vsm_args = parse_args({})
    vsm_args.version = args.vsm_model_path
    if vsm is None:
        vsm = VSM(vsm_args)
    

    prompt_part = prompt_part.replace("<object_name>", "{}").replace("<bounding_box>", "[{:.3f},{:.3f},{:.3f},{:.3f}]")

    #prompt_use = {"pre_object": "The model should consider this: <object> refers to ", "mid_object": ", located at ", "post_object": ".", "question": "Please answer: "}
    #prompt_use = {"pre_object": "The model should consider this: <object> refers to ", "mid_object": ", located at ", "post_object": ".", "question": prompt_part}



    results = {}
    per_type_acc = defaultdict(list)
    missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
    focus_msg = "Additional visual information to focus on: "
    results[test_type] = []
    folder = os.path.join(args.benchmark_folder, test_type)
    image_path = os.path.join(folder, image_file)
    annotation_path = image_path.split('.')[0] + '.json'
    image = Image.open(image_path).convert('RGB')
    annotation = json.load(open(annotation_path))
    missing_objects = annotation['target_object']
    image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
    
    question = annotation['question']
    # generate free-form response to check whether visual search needs to be activated
    if vqa_do:
        prediction = vqa_llm.free_form_inference(image, question)
        if missing_objects_msg in prediction:
            missing_objects = prediction.split(missing_objects_msg)[-1]
            if missing_objects.endswith('.'):
                missing_objects = missing_objects[:-1]
            missing_objects = missing_objects.split(',')
            missing_objects = [missing_object.strip() for missing_object in missing_objects]

    search_result = []
    if len(missing_objects) > 0:
        # visual search
        for object_name in missing_objects:
            image = Image.open(image_path).convert('RGB')
            smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
            final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm, image, object_name, target_bbox=None, smallest_size=smallest_size)
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
            cur_focus_msg = cur_focus_msg + prompt_part.format(object_name, bbox[0], bbox[1], bbox[2], bbox[3])
            if i != len(bbox_list)-1:
                cur_focus_msg = cur_focus_msg+"; "
            else:
                cur_focus_msg = cur_focus_msg +'.'
        question_with_focus = cur_focus_msg+"\n"+question

        option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops, images_long=images_long, objects_long=objects_long)
    else:
        option_chosen = vqa_llm.multiple_choices_inference(image, question, options)

    correct = 1 if option_chosen==0 else 0
    per_type_acc[test_type].append(correct)

    #print("correct: ", correct, "option_chosen: ", option_chosen, "question: ", question, "options: ", options)
    
    return "correct answer" if correct else "incorrect answer"

def test_model_multiple_bbox(model_name, prompt_part, test_type, image_file, retrieve_bounding_boxes = False, vqa_do=False, missing_objects=[]):

    parser = argparse.ArgumentParser()
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
    # init VSM
    vsm_args = parse_args({})
    vsm_args.version = args.vsm_model_path
    if vsm is None:
        vsm = VSM(vsm_args)


    prompt_use = {"pre_object": "The model should consider this: <object> refers to ", "mid_object": ", located at ", "post_object": ".", "question": "Please answer: "}
    #prompt_use = {"pre_object": "The model should consider this: <object> refers to ", "mid_object": ", located at ", "post_object": prompt_part, "question": "Please answer: "}

    results = {}
    per_type_acc = defaultdict(list)
    missing_objects_msg = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear:"
    focus_msg = "Additional visual information to focus on: "
    results[test_type] = []
    folder = os.path.join(args.benchmark_folder, test_type)
    image_path = os.path.join(folder, image_file)
    annotation_path = image_path.split('.')[0] + '.json'
    image = Image.open(image_path).convert('RGB')
    annotation = json.load(open(annotation_path))
    missing_objects = annotation['target_object']
    image, _, _ = expand2square(image, tuple(int(x*255) for x in vqa_llm.image_processor.image_mean))
    
    question = annotation['question']
    # generate free-form response to check whether visual search needs to be activated
    if vqa_do:
        prediction = vqa_llm.free_form_inference(image, question)
        if missing_objects_msg in prediction:
            missing_objects = prediction.split(missing_objects_msg)[-1]
            if missing_objects.endswith('.'):
                missing_objects = missing_objects[:-1]
            missing_objects = missing_objects.split(',')
            missing_objects = [missing_object.strip() for missing_object in missing_objects]

    search_result = []
    if len(missing_objects) > 0:
        # visual search
        for object_name in missing_objects:
            image = Image.open(image_path).convert('RGB')
            smallest_size = max(int(np.ceil(min(image.width, image.height)/args.minimum_size_scale)), args.minimum_size)
            final_step, path_length, search_successful, all_valid_boxes = visual_search(prompt_part, vsm, image, object_name, target_bbox=None, smallest_size=smallest_size)
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
    # predict the multiple-choice option
    image = Image.open(image_path).convert('RGB')
    if len(missing_objects) > 0:
        object_names = [_['name'] for _ in search_result]
        bboxs = deepcopy([_['bbox'] for _ in search_result])
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
            cur_focus_msg = cur_focus_msg + prompt_use["pre_object"] + "{}".format(object_name) + prompt_use["mid_object"] + "[{:.3f},{:.3f},{:.3f},{:.3f}]".format(bbox[0], bbox[1], bbox[2], bbox[3]) + prompt_use["post_object"]
            if i != len(bbox_list)-1:
                cur_focus_msg = cur_focus_msg+"; "
            else:
                cur_focus_msg = cur_focus_msg +'.'
        if retrieve_bounding_boxes:
            return search_result
        return cur_focus_msg
    else:
        if retrieve_bounding_boxes:
            return {}
        return "no missing objects"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, default="missing_objects", help="Name of the test to run: 'missing_objects', 'bounding_box', 'final_choice'")
    args = parser.parse_args()
    test_name = args.test_name

    if test_name == "missing_objects":
        main_test_objects_optimise(with_image=True)
    elif test_name == "bounding_box":
        main_test_seal_bbox()
    elif test_name == "choice":
        main_test_seal_choice_optimise()
