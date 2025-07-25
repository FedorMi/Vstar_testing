from IVM import load, forward_batch

import numpy as np
import json
import cv2
import os
import torch

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

def main_normal():
    ckpt_path = os.path.join("pytorch_model", "IVM-V1.0.bin") # your model path here
    model = load(ckpt_path, low_gpu_memory = False)
    from PIL import Image
    data_origin = load_json("eval_result.json")
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
    ckpt_path = os.path.join("pytorch_model", "IVM-V1.0.bin") # your model path here
    model = initialize_and_check_model(ckpt_path, low_gpu_memory=False)
    #model = load(ckpt_path, low_gpu_memory = False)
    from PIL import Image
    data_origin = load_json("eval_result.json")
    images_direct = []
    images_direct_related = []
    images_relative = []
    images_relative_related = []
    for i in data_origin["direct_attributes"]:
        images_direct.append(i["image"])
        images_direct_related.append(i["missing_objects"])
    for i in data_origin["relative_position"]:
        images_relative.append(i["image"])
        images_relative_related.append(i["missing_objects"])
    print("images loaded")
    print(len(images_direct), "direct images")
    print(len(images_relative), "relative images")
    #images_direct = ["sa_1244.jpg","sa_9877.jpg","sa_9957.jpg","sa_10033.jpg","sa_19272.jpg","sa_24546.jpg","sa_27986.jpg","sa_29744.jpg","sa_38610.jpg","sa_59664.jpg","sa_70112.jpg","sa_80352.jpg","sa_86882.jpg"]
    #images_relative = ["sa_24031.jpg","sa_31038.jpg","sa_58331.jpg","sa_25747.jpg","sa_64257.jpg","sa_80084.jpg","sa_87051.jpg","sa_40555.jpg","sa_65804.jpg","sa_78272.jpg"]
    print(len(images_direct)+len(images_relative))
    if True:
        for idx,i in enumerate(images_direct):
            image_path = os.path.join("direct_attributes", i)
            json_path = os.path.join("direct_attributes", i.split(".")[0] + ".json")
            data1 = load_json(json_path)
            instruction = data1["question"]        
            image = Image.open(image_path) # your image path
            _, temp_image, temp_mask, temp_temp = forward_batch(model, [image], [instruction], threshold = 0.8)
            temp_masks = [temp_mask]
            for j in images_direct_related[idx]:
                print(j)
                _, _, temp_mask, _, o1, o2, o3, o4 = forward_batch(model, [image], ["Please locate the"+j+"in this image."], threshold = 0.8)
                temp_masks.append(temp_mask)
            mask = np.zeros_like(temp_masks[0])
            for temp_mask in temp_masks:
                mask = np.maximum(mask,temp_mask)
            result = temp_image * mask + temp_temp * (1 - mask)
            # Convert the result to an image and save it
            output_image = Image.fromarray((result).astype(np.uint8))
            output_path = os.path.join("conversion_related", "direct_attributes", i)
            output_image.save(output_path) # specify your output directory and file name
            print("image",i,"done")
    for idx,i in enumerate(images_relative):
        image_path = os.path.join("relative_position", i)
        json_path = os.path.join("relative_position", i.split(".")[0] + ".json")
        data1 = load_json(json_path)
        instruction = data1["question"]   
        image = Image.open(image_path) # your image path
        _, temp_image, temp_mask, temp_temp, o1, o2, o3, o4 = forward_batch(model, [image], [instruction], threshold = 0.8)
        temp_masks = [temp_mask]
        for j in images_relative_related[idx]:
            print(j)
            _, _, temp_mask, _ = forward_batch(model, [image], ["Please locate the"+j+"in this image."], threshold = 0.8)
            temp_masks.append(temp_mask)
        mask = np.zeros_like(temp_masks[0])
        for temp_mask in temp_masks:
            mask = np.maximum(mask,temp_mask)
        result = temp_image * mask + temp_temp * (1 - mask)
        # Convert the result to an image and save it
        output_image = Image.fromarray((result).astype(np.uint8))
        output_path = os.path.join("conversion_related","relative_position", i)
        output_image.save(output_path) # specify your output directory and file name
        print("image",i,"done")
if __name__ == "__main__":
    main_normal()

