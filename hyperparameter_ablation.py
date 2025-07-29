import argparse
import os
import json
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

from visual_search_parameter_ablation import parse_args, VSM, visual_search

def normalize_bbox(bbox, image_width, image_height):
    normalized_bbox = [bbox[0]/image_width, bbox[1]/image_height, (bbox[0]+bbox[2])/image_width, (bbox[1]+bbox[3])/image_height]
    normalized_bbox = [np.clip(_, 0, 1) for _ in normalized_bbox]
    return normalized_bbox
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img, 0, 0
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result, 0, (width - height) // 2
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result, (height - width) // 2, 0

class VQA_LLM:
    def __init__(self, args):
        disable_torch_init()
        model_path = args.vqa_model_path
        model_name = get_model_name_from_path(model_path)
        model_name += 'llava'
        model_base = None
        device_map = "auto"
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name)
        self.conv_type = args.conv_type

    def get_patch(self, bbox, image_width, image_height, patch_size=224, patch_scale=None):
        object_width = int(np.ceil(bbox[2]))
        object_height = int(np.ceil(bbox[3]))

        object_center_x = int(bbox[0] + bbox[2]/2)
        object_center_y = int(bbox[1] + bbox[3]/2)

        if patch_scale is None:
            patch_width = max(object_width, patch_size)
            patch_height = max(object_height, patch_size)
        else:
            patch_width = int(object_width*patch_scale)
            patch_height = int(object_height*patch_scale)

        left = max(0, object_center_x-patch_width//2)
        right = min(left+patch_width, image_width)

        top = max(0, object_center_y-patch_height//2)
        bottom = min(top+patch_height, image_height)

        return [left, top, right, bottom]
    
    def get_object_crop(self, image, bbox, patch_scale):
        resized_bbox = self.get_patch(bbox, image.width, image.height, patch_scale=patch_scale)
        object_crop = image.crop((resized_bbox[0], resized_bbox[1], resized_bbox[2], resized_bbox[3]))
        object_crop = object_crop.resize((self.image_processor.crop_size['width'],self.image_processor.crop_size['height']))
        object_crop = self.image_processor.preprocess(object_crop, return_tensors='pt')['pixel_values'][0]
        return object_crop

    @torch.inference_mode()
    def free_form_inference(self, image, question, temperature=0, top_p=None, num_beams=1, max_new_tokens=200, object_crops=None, images_long=None, objects_long=None):
        conv = conv_templates[self.conv_type].copy()
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question    
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        input_ids = tokenizer_image_object_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            object_features=object_crops.half().cuda() if object_crops is not None else None,
            images_long = images_long,
            objects_long = objects_long,
            do_sample= True if temperature > 0 else False,
            num_beams=num_beams,
            temperature=temperature,
            top_p = top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    @torch.inference_mode()
    def multiple_choices_inference(self, image, question, options, object_crops=None, images_long=None, objects_long=None):
        conv = conv_templates[self.conv_type].copy()
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question    
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        question_input_ids = tokenizer_image_object_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        output_question = self.model(
            question_input_ids,
            use_cache=True,
            images=image_tensor.unsqueeze(0).half().cuda(),
            object_features=object_crops.half().cuda() if object_crops is not None else None,
            images_long = images_long,
            objects_long = objects_long)

        question_logits = output_question.logits
        question_past_key_values = output_question.past_key_values

        loss_list = []

        for option in options:
            conv = conv_templates[self.conv_type].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], option)
            full_prompt = conv.get_prompt()

            full_input_ids = tokenizer_image_object_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            option_answer_input_ids = full_input_ids[:, question_input_ids.shape[1]:]

            output_option = self.model(input_ids=option_answer_input_ids,
                                use_cache=True,
                                attention_mask=torch.ones(1, question_logits.shape[1]+option_answer_input_ids.shape[1], device=full_input_ids.device),
                                past_key_values=question_past_key_values)
            
            logits = torch.cat([question_logits[:, -1:], output_option.logits[:, :-1]], 1)

            loss_fct = CrossEntropyLoss()
            logits = logits.view(-1, self.model.config.vocab_size)
            labels = option_answer_input_ids.view(-1)
            loss = loss_fct(logits, labels)

            loss_list.append(loss)

        option_chosen = torch.stack(loss_list).argmin()

        return option_chosen.cpu().item()


def eval_model(args, parameter_name, parameter_value, parameter_value_optional=None):
    # init VQA LLM
    vqa_llm = VQA_LLM(args)
    # init VSM
    vsm_args = parse_args({}, parameter_name, parameter_value, parameter_value_optional)
    vsm_args.version = args.vsm_model_path
    vsm = VSM(vsm_args)




    patch_scale = 1.2
    minimum_size_scale = 4.0
    minimum_size = 224
    if parameter_name == "confidence":
        if parameter_value >= parameter_value_optional:
            return "misinput"
    elif parameter_name == "patch_scale":
        patch_scale = parameter_value
    elif parameter_name == "minimum_size_scale":
        minimum_size_scale = parameter_value
    elif parameter_name == "minimum_size":
        minimum_size = parameter_value

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
            # generate free-form response to check whether visual search needs to be activated
            prediction = vqa_llm.free_form_inference(image, question)
            missing_objects = []
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
                    smallest_size = max(int(np.ceil(min(image.width, image.height)/minimum_size_scale)), minimum_size)
                    final_step, path_length, search_successful, all_valid_boxes = visual_search(vsm_args, vsm, image, object_name, target_bbox=None, smallest_size=smallest_size)
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
                object_crops = []
                for bbox in bboxs:
                    object_crop = vqa_llm.get_object_crop(image, bbox, patch_scale=patch_scale)
                    object_crops.append(object_crop)
                if len(object_crops) > 0:
                    object_crops = torch.stack(object_crops, 0)
                    option_chosen = vqa_llm.multiple_choices_inference(image, question_with_focus, options, object_crops, images_long=images_long, objects_long=objects_long)
                else:
                    option_chosen = vqa_llm.multiple_choices_inference(image, question, options)
            else:
                option_chosen = vqa_llm.multiple_choices_inference(image, question, options)

            correct = 1 if option_chosen==0 else 0
            per_type_acc[test_type].append(correct)
            all_acc.append(correct)

            result_single_sample['question'] = question
            result_single_sample['options'] = options
            result_single_sample['image'] = image_file
            result_single_sample['prediction_freeform'] = prediction
            result_single_sample['missing_objects'] = missing_objects
            result_single_sample['search_result'] = search_result    
            result_single_sample['option_chosen'] = option_chosen
            result_single_sample['correct'] = correct
            results[test_type].append(result_single_sample)

        print(test_type, np.mean(per_type_acc[test_type]))

    print(np.mean(all_acc))

    return results

if __name__ == "__main__":
    # To initialize the test, put the test type here
    #test_type = "patch_scale"
    # chose from ["patch_scale", "minimum_size_scale", "minimum_size", "target_cue_threshold", 
    #               "target_cue_threshold_decay", "target_cue_threshold_minimum", "model_max_length", "confidence"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-type", type=str, default="patch_scale", help="Type of hyperparameter to test")
    parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
    parser.add_argument("--vqa-model-base", type=str, default=None)
    parser.add_argument("--conv_type", default="v1", type=str,)
    parser.add_argument("--benchmark-folder", type=str, default="vbench")
    parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
    parser.add_argument("--output-path", type=str, default="eval_result.json")
    parser.add_argument("--minimum_size_scale", default=4.0, type=float, help="minimum sub-image scale for the termination of search")
    parser.add_argument("--minimum_size", default=224, type=int, help="minimum sub-image size for the termination of search")
    args = parser.parse_args()

    test_type = args.test_type


    output_path = "parameter_ablation" + test_type + ".json"
    

    # Hyperparameters for ablation only SEAL
    value_spread_patch_scale = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6]

    # Hyperparameters for ablation SEAL + VSM
    value_spread_minimum_size_scale = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 20.0, 100.0]
    value_spread_minimum_size = [16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 600, 1200]
    
    # Hyperparameters for ablation VSM
    value_spread_target_cue_threshold = [4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 6.4, 6.8, 7.2, 7.6, 8.0]
    value_spread_target_cue_threshold_decay = [0.5, 0.54, 0.58, 0.62, 0.66, 0.7, 0.74, 0.78, 0.82, 0.86, 0.9]
    value_spread_target_cue_threshold_minimum = [1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0]
    value_spread_model_max_length = [64, 128, 256, 512, 1024, 2048, 4096]

    value_spread_confidence_low = [0.1, 0.2, 0.3, 0.4, 0.5]
    value_spread_confidence_high = [0.3, 0.4, 0.5, 0.6, 0.7]

    
    total_results = {}
    value_spread = []

    # case for confidence ablation
    if test_type == "confidence":
        for value in value_spread_confidence_low:
            for value_high in value_spread_confidence_high:
                results = eval_model(args, test_type, value, value_high)
                if results == "misinput":
                    continue
                total_results[f"{test_type}_{value}_{value_high}"] = results

    else:
        if test_type == "patch_scale":
            value_spread = value_spread_patch_scale
        elif test_type == "minimum_size_scale":
            value_spread = value_spread_minimum_size_scale
        elif test_type == "minimum_size":
            value_spread = value_spread_minimum_size
        elif test_type == "target_cue_threshold":
            value_spread = value_spread_target_cue_threshold
        elif test_type == "target_cue_threshold_decay":
            value_spread = value_spread_target_cue_threshold_decay
        elif test_type == "target_cue_threshold_minimum":
            value_spread = value_spread_target_cue_threshold_minimum
        elif test_type == "model_max_length":
            value_spread = value_spread_model_max_length
        for value in value_spread:
            results = eval_model(args, test_type, value)
            total_results[f"{test_type}_{value}"] = results

    
    with open(output_path, 'w') as f:
        json.dump(total_results, f, indent=4)