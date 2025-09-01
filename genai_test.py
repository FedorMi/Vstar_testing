
from google import genai
from google.genai import types


import json
import random
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor

import io
import requests
from io import BytesIO



import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria, tokenizer_image_object_token

from visual_search import parse_args, VSM, visual_search
GOOGLE_API_KEY = "here_comes_your_api_key"


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

def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def plot_bounding_boxes_origin(img_path, im, bounding_boxes):
    additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)
    
    # Iterate over the bounding 
    out = []
    try:
        out = json.loads(bounding_boxes)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(bounding_boxes)
        return []


    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(out):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
      abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
      abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
      abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

    # save the image with bounding boxes
    img.save(img_path.replace('.jpg', '_bounding_boxes.jpg').replace('vbench', 'genoutput'))



def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
        and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    # Iterate over the bounding 
    out = []
    try:
        out = json.loads(bounding_boxes)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(bounding_boxes)
        return []

    for i in range(len(out)):
        bounding_box = out[i]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        b_height = abs_y2 - abs_y1
        b_width = abs_x2 - abs_x1

        bounding_box["box_2d"] = [abs_x1, abs_y1, b_width, b_height]

        # Draw the bounding box
        out[i] = bounding_box
    
    return out


def genai_visual_search(image_path, object_name, client):

    model_name = "gemini-2.5-flash-preview-05-20"
    bounding_box_system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        """
    safety_settings = [
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ]

    
    image = image_path
    im = Image.open(image)
    im.thumbnail([620,620], Image.Resampling.LANCZOS)

    prompt = "Detect the 2d bounding boxes of " + object_name   # @param {type:"string"}

    # Load and resize image
    im = Image.open(BytesIO(open(image, "rb").read()))
    im.thumbnail([1024,1024], Image.Resampling.LANCZOS)

    # Run model to find bounding boxes
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, im],
        config = types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
        )
    )

    # Check output
    plot_bounding_boxes_origin(image_path, im, response.text)

    return plot_bounding_boxes(im, response.text)

def eval_model(args):
    

    client = genai.Client(api_key=GOOGLE_API_KEY)
    vsm_args = parse_args({})
    vsm_args.version = args.vsm_model_path

    output_dict = {}
    direct_files = ["sa_19272.jpg", "sa_9957.jpg", "sa_80352.jpg", "sa_24546.jpg", "sa_29744.jpg"]
    relative_files = ["sa_24031.jpg", "sa_31038.jpg", "sa_58331.jpg", "sa_64257.jpg"]
    for test_type in ['direct_attributes', 'relative_position']:
        folder = os.path.join(args.benchmark_folder, test_type)
        image_files = []
        if test_type == 'direct_attributes':
            image_files = direct_files
        elif test_type == 'relative_position':
            image_files = relative_files
        #image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        temp = []
        for image_file in tqdm(image_files):
            image_temp = {}
            image_temp['image_file'] = image_file
            image_temp['objects'] = []
            image_path = os.path.join(folder, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            annotation = json.load(open(annotation_path))
            missing_objects = annotation["target_object"]
            if len(missing_objects) > 0:
                # visual search
                for object_name in missing_objects:
                    bboxes_with_objects = genai_visual_search(image_path, object_name, client)
                    image_temp["objects"].extend(bboxes_with_objects)
            temp.append(image_temp)
        output_dict[test_type] = temp
    with open(args.output_path, 'w') as f:
        json.dump(output_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
    parser.add_argument("--vqa-model-base", type=str, default=None)
    parser.add_argument("--conv_type", default="v1", type=str,)
    parser.add_argument("--benchmark-folder", type=str, default="vbench")
    parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
    parser.add_argument("--output-path", type=str, default="eval_result_genai.json")
    parser.add_argument("--minimum_size_scale", default=4.0, type=float, help="minimum sub-image scale for the termination of search")
    parser.add_argument("--minimum_size", default=224, type=int, help="minimum sub-image size for the termination of search")

    args = parser.parse_args()
    eval_model(args)