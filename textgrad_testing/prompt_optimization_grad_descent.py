import os
import json
import torch
import random
import argparse
import concurrent
import numpy as np
import textgrad as tg


from ollama import chat, ResponseError, pull
from typing import Callable, List, Any
from copy import deepcopy
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from textgrad.engine.local_model_openai_api import ChatExternalClient  # (not needed anymore)
from textgrad.autograd import MultimodalLLMCall
from textgrad.loss import ImageQALoss
#from textgrad.tasks.base import DataLoader
from torch.utils.data import DataLoader, Dataset

from vstar_bench_eval import VQA_LLM, expand2square, normalize_bbox
from visual_search import parse_args, VSM, visual_search
from torch.utils.data import random_split


# this is a modification of the textrad prompt optimization notebook

vqa_llm = None
vsm = None
vsm_model_path = "craigwu/seal_vsm_7b"
minimum_size_scale = 4.0
minimum_size = 224

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
    return annotation['target_object'], "no prediction needed, the object labels are correct"
def get_bounding_boxes_seal(missing_objects, image_path, annotation, prompt_template):
    search_result = []
    if len(missing_objects) > 0:
        # visual search
        for object_name in missing_objects:
            image = Image.open(image_path).convert('RGB')
            smallest_size = max(int(np.ceil(min(image.width, image.height)/minimum_size_scale)), minimum_size)
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
def get_bounding_boxes_correct(missing_objects, image_path, annotation):
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
        search_result = get_bounding_boxes_seal(missing_objects, image_path, annotation, prompt_template)
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
    return iou_total / count if count > 0 else 0
def test_bounding_boxes_final_result(prompt_template, evaluation_set):
    global vsm
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
        search_result = get_bounding_boxes_seal(missing_objects, image_path, annotation, prompt_template)
        correct, _, _ = get_multiple_choice_seal(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template)
        correct_total += correct
        count += 1
    return correct_total / count if count > 0 else 0
def test_final_call(prompt_template, evaluation_set):
    if True:
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
        search_result = initial_image_data['search_result']
        correct, _, _ = get_multiple_choice_seal(image_path, question, search_result, annotation, missing_objects, focus_msg, prompt_template)
        correct_total += correct
        count += 1
    return correct_total / count if count > 0 else 0

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def run_validation_revert(system_prompt: tg.Variable, results, val_set, eval_func):
    val_performance = eval_func(system_prompt.value, val_set)
    previous_performance = results["validation_acc"][-1]
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance
    results["validation_acc"].append(val_performance)

def textgrad_prompt_optimization(eval_func, data_set):
    set_seed(42)

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    engine = ChatExternalClient(client=client, model_string='llama3:70b')
    tg.set_backward_engine(engine, override=True)

    # Load the data and the evaluation function
    train_fraction = 0.7
    val_fraction = 0.15
    test_fraction = 1.0 - train_fraction - val_fraction
    train_len = int(len(data_set)*train_fraction)      
    val_len = int(len(data_set)*val_fraction)
    test_len = len(data_set) - train_len - val_len
    train_set, val_set, test_set = random_split(data_set, [train_len, val_len, test_len])
    #train_set, val_set, test_set, eval_fn = load_task("BBH_object_counting", evaluation_api=llm_api_eval)
    data_set_train = MyDataset(train_set)
    train_loader = DataLoader(data_set_train, batch_size=12, shuffle=True)

    print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
    STARTING_SYSTEM_PROMPT = "here is prompt template"
    # Testing the 0-shot performance of the evaluation engine
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                                requires_grad=True,
                                role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")

    optimizer = tg.TGD(engine=engine, parameters=[system_prompt])

    results = {"test_acc": [], "prompt": [], "validation_acc": []}

    #results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
    #results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))

    results["test_acc"].append(eval_func(system_prompt.value, test_set))
    results["validation_acc"].append(eval_func(system_prompt.value, val_set))
    results["prompt"].append(system_prompt.get_value())
    for epoch in range(3):
        for steps, batch_x in enumerate((pbar := tqdm(train_loader, position=0))):
            print(batch_x)
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            optimizer.zero_grad()
            losses = []
            training_divisions = 2
            mini_batch_len = len(batch_x) // training_divisions
            
            for div_num  in range(training_divisions):
                if len(batch_x) % training_divisions != 0 and div_num == training_divisions - 1:
                    mini_batch_len = len(batch_x) - mini_batch_len * (training_divisions - 1)
                curr_batch_x = batch_x[div_num * mini_batch_len:(div_num + 1) * mini_batch_len]
                #curr_batch_y = batch_y[div_num * mini_batch_len:(div_num + 1) * mini_batch_len]
                eval_output_variable = tg.Variable(str(eval_func(system_prompt.value, curr_batch_x)), requires_grad=False, role_description="evaluation of mini-batch results")
                losses.append(eval_output_variable)
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()
            
            run_validation_revert(system_prompt, results, val_set, eval_func)
            
            print("sys prompt: ", system_prompt)
            test_acc = eval_func(system_prompt.value, test_set)
            results["test_acc"].append(test_acc)
            results["prompt"].append(system_prompt.get_value())
            if steps == 3:
                break

if __name__ == "__main__":
    data_set = []
    for test_type in ['direct_attributes', 'relative_position']:
        folder = os.path.join("vbench", test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for image_file in image_files:
            image_path = test_type + "$" + image_file
            data_set.append(image_path)
    textgrad_prompt_optimization(test_missing_objects, data_set)