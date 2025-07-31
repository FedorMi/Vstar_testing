from IVM import load, forward_batch, auto_postprocess
from typing import Type, Tuple, List
from torch.nn import functional as F

import numpy as np
import json
import cv2
import os
import torch
from PIL import Image

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def initialize_and_check_model(ckpt_path, low_gpu_memory=False):
    model = load(ckpt_path, low_gpu_memory)
    print("Model initialized")
    
    total_size = 0
    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        total_size += param_size
        print(f"Component: {name}, Size: {param_size / (1024 ** 2):.2f} MB ({param_size / (1024 ** 3):.2f} GB)")
    
    print(f"Total Model Size: {total_size / (1024 ** 2):.2f} MB ({total_size / (1024 ** 3):.2f} GB)")
    return model
@torch.no_grad()
def forward_batch_intermediate(
    model, 
    image, # list of PIL.Image
    instruction: List[str], # list of instruction
    threshold: float = 0.1, # threshold for pixel reserve/drop
    do_crop = False,
    overlay_color = (255,255,255)
):
    ori_sizes = [img.size for img in image]
    ori_images = [np.asarray(img).astype(np.float32) for img in image]
    masks = model.generate_batch([img.resize((1024, 1024)) for img in image], instruction)

    result = []
    for mask, ori_image, ori_size in zip(masks, ori_images, ori_sizes):
        mask = torch.sigmoid(F.interpolate(
            mask.unsqueeze(0),
            (ori_size[1], ori_size[0]),
            mode="bilinear",
            align_corners=False,
        )[0, 0, :, :]).detach().cpu().numpy().astype(np.float32)[:,:,np.newaxis]
        if threshold > mask.max(): mask += threshold # fail to find the target, reserve the full image
        mask = auto_postprocess((mask > threshold).astype(np.float32))

        if len(ori_image.shape) < 3: ori_image = ori_image[:,:,np.newaxis].repeat(3,-1)
        
        processed_image = ori_image * mask + torch.tensor(overlay_color, dtype=torch.uint8).repeat(ori_size[1], ori_size[0], 1).numpy() * (1 - mask)
        temp = torch.tensor(overlay_color, dtype=torch.uint8).repeat(ori_size[1], ori_size[0], 1).numpy()
        try:
            y_indices, x_indices = np.where(mask[:,:,0] > 0)
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            processed_image = processed_image[y_min:y_max+1, x_min:x_max+1]
        except:
            print("Warning, unable to crop a sample, reserve whole image")
        result.append(processed_image)
        return processed_image, ori_image, mask, temp, x_min, x_max, y_min, y_max
    return result

def main_normal():
    ckpt_path = os.path.join("IVM", "IVM-V1.0.bin") # your model path here
    model = load(ckpt_path, low_gpu_memory = False)

    test_types = ["relative_position", "direct_attributes"]
    for test_type in test_types:
        folder = os.path.join("vbench", test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for i in image_files:
            image_path = os.path.join(test_type, i)
            json_path = os.path.join(test_type, i.split(".")[0] + ".json")
            full_json_path = os.path.join("vbench", json_path)
            full_image_path = os.path.join("vbench", image_path)
            data1 = load_json(full_json_path)
            instruction = data1["question"]        
            image = Image.open(full_image_path) # your image path
            result = forward_batch(model, [image], [instruction], threshold = 0.8, do_crop = True)

            # Convert the result to an image and save it
            output_image = Image.fromarray((result[0]).astype(np.uint8))
            output_path = os.path.join("ivm_image_results", test_type, i)
            output_image.save(output_path) # specify your output directory and file name
            print("image",i,"done")

def main_test():
    ckpt_path = os.path.join("IVM", "IVM-V1.0.bin") # your model path here
    model = initialize_and_check_model(ckpt_path, low_gpu_memory=False)
    #model = load(ckpt_path, low_gpu_memory = False)
    test_types = ["relative_position", "direct_attributes"]
    for test_type in test_types:
        folder = os.path.join("vbench", test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for i in image_files:
            image_path = os.path.join(test_type, i)
            json_path = os.path.join(test_type, i.split(".")[0] + ".json")
            full_json_path = os.path.join("vbench", json_path)
            full_image_path = os.path.join("vbench", image_path)
            data1 = load_json(full_json_path)
            images_related = data1["target_object"]
            instruction = data1["question"]        
            image = Image.open(full_image_path) # your image path
            _, temp_image, temp_mask, temp_temp, o1, o2, o3, o4 = forward_batch_intermediate(model, [image], [instruction], threshold = 0.8)
            temp_masks = [temp_mask]
            for j in images_related:
                print(j)
                _, _, temp_mask, _, o1, o2, o3, o4 = forward_batch_intermediate(model, [image], ["Please locate the"+j+"in this image."], threshold = 0.8)
                temp_masks.append(temp_mask)
            mask = np.zeros_like(temp_masks[0])
            for temp_mask in temp_masks:
                mask = np.maximum(mask,temp_mask)
            result = temp_image * mask + temp_temp * (1 - mask)
            try:
                y_indices, x_indices = np.where(mask[:,:,0] > 0)
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                result = result[y_min:y_max+1, x_min:x_max+1]
            except:
                print("Warning, unable to crop a sample, reserve whole image")
            # Convert the result to an image and save it
            output_image = Image.fromarray((result).astype(np.uint8))
            output_path = os.path.join("ivm_image_results_related", test_type, i)
            output_image.save(output_path) # specify your output directory and file name
            print("image",i,"done")
if __name__ == "__main__":
    main_normal()

