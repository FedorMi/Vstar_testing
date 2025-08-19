import os
import json
import random
import argparse
import numpy as np


from ollama import chat, ResponseError, pull
from typing import Callable, List, Any
from tqdm import tqdm

#from textgrad.tasks.base import DataLoader
from torch.utils.data import DataLoader, Dataset
from statistics import mean 

from torch.utils.data import random_split


# this is a modification of the textrad prompt optimization notebook


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
    if model_name == "llava:34b_custom":
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
    if model_name == "llava:34b_custom":
        result = handle_llava_response(temp_result)
    elif model_name == "deepseek-r1:32b":
        result = handle_deepseek_response(temp_result)
    return result
def get_missing_object_labels_correct(annotation):
    return annotation['target_object']
def test_missing_objects(prompt_template, evaluation_set,with_image=True, model_name="llava:34b_custom"):
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
    
    if val_performance <= previous_performance:
        print(f"rejected prompt: {system_prompt.get_value()}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance
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

def ollama_prompt_optimization(eval_func, data_set, starting_prompt: str, opti_model: str = "llama3:70b", prompt_gen_func: Callable = make_new_prompt_missing_object, args = None):
    # Load the data and the evaluation function
    train_fraction = 0.5
    val_fraction = 0.25
    test_fraction = 1.0 - train_fraction - val_fraction
    train_len = int(len(data_set)*train_fraction)      
    val_len = int(len(data_set)*val_fraction)
    test_len = len(data_set) - train_len - val_len
    train_set, val_set, test_set = random_split(data_set, [train_len, val_len, test_len])
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
    for epoch in range(100):
        for steps, batch_x in enumerate((pbar := tqdm(train_loader, position=0))):
            #print(batch_x)
            pbar.set_description(f"Training step {steps}. Epoch {epoch}")
            losses = []
            training_divisions = 2
            mini_batch_len = len(batch_x) // training_divisions
            
            for div_num  in range(training_divisions):
                if len(batch_x) % training_divisions != 0 and div_num == training_divisions - 1:
                    mini_batch_len = len(batch_x) - mini_batch_len * (training_divisions - 1)
                curr_batch_x = batch_x[div_num * mini_batch_len:(div_num + 1) * mini_batch_len]
                #curr_batch_y = batch_y[div_num * mini_batch_len:(div_num + 1) * mini_batch_len]
                eval_output_variable = eval_func(system_prompt.get_value(), curr_batch_x)
                losses.append(eval_output_variable)
            total_loss = mean(losses)

            system_prompt = prompt_gen_func(system_prompt.get_value(), total_loss, results, model=opti_model)
            system_prompt = MyVariable(system_prompt, 
                                requires_grad=True,
                                role_description="prompt to the model to answer the VQA task")

            run_validation_revert(system_prompt, results, val_set, eval_func)
            
            print("sys prompt: ", system_prompt.get_value())
            test_acc = eval_func(system_prompt.get_value(), test_set)
            results["test_acc"].append(test_acc)
            print("test_acc: ", test_acc)
            results["prompt"].append(system_prompt.get_value())
            if steps == 100:
                break
        # save results
        with open("results_" + args.experiment + ".json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="missing_object")
    args = parser.parse_args()
    data_set = []
    for test_type in ['direct_attributes', 'relative_position']:
        folder = os.path.join("vbench", test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for image_file in image_files:
            image_path = test_type + "$" + image_file
            data_set.append(image_path)
    prompt = """You are a helpful assistant that provides the objects present in the question to the user. 
                You do not give explanations, you don't respond in full sentences, you only respond with objects. The relevant question is: <QUESTION>.\n 
                Do not answer the question, only provide the objects that are relevant to the question. 
                The objects should be separated by commas, and the objects should be in lowercase."""
    #prompt = "You are a helpful assistant that provides the objects present in the question to the user. The relevant question is: <QUESTION>. Please identify and list the main entities, concepts, and key objects mentioned in the question, separated by commas, in lowercase, without explanations or full sentences. Focus on extracting the most important elements, ignoring irrelevant details, and prioritize clarity over completeness."
    #prompt = "You are a helpful assistant that provides the objects present in the question to the user. The relevant question is: <QUESTION>. Please extract and list the essential entities, concepts, and key objects mentioned in the question, separated by commas, in lowercase, without explanations or full sentences. Focus on identifying the most crucial elements, ignoring irrelevant details, and prioritize clarity over completeness while maintaining a balance between brevity and accuracy."
    func_to_give = test_missing_objects
    optim_model_name = "llama3:70b_custom"
    #optim_model_name = "llama3:8b"
    prompt_gen_func = make_new_prompt_missing_object

    ollama_prompt_optimization(func_to_give, data_set, prompt, optim_model_name, prompt_gen_func, args)