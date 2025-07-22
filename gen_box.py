import json
import cv2
import os

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def draw_bounding_boxes(image_path, json_path, output_path, color=(0, 0, 255)):
    data = load_json(json_path)
    image = cv2.imread(image_path)
    
    for bbox in data["bbox"]:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
    
    cv2.imwrite(output_path, image)

def draw_eval_bounding_boxes(image_path, eval_json_path, output_path, image_name, image_type, color=(255, 0, 0)):
    data = load_json(eval_json_path)
    image = cv2.imread(image_path)
    
    for element in data[image_type]:
        if element["image"] == image_name:
            for result in element["search_result"]:
                bbox = result["bbox"]
                x, y, w, h = bbox
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
    
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    image_name = "sa_80352.jpg"
    #image_type = "relative_position"
    image_type = "direct_attributes"
    image_path = os.path.join("vbench", image_type, image_name)
    json_path = os.path.join("vbench", image_type, image_name.replace(".jpg", ".json"))
    eval_json_path = "eval_result.json"
    output_path = os.path.join("output_images",image_name.replace(".jpg", ".json"), image_name)
    eval_output_path = os.path.join("output_images",image_name.replace(".jpg", ".json"), "eval_" + image_name)
    new_directory = os.path.join("output_images",image_name.replace(".jpg", ".json"))
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    
    draw_bounding_boxes(image_path, json_path, output_path)
    draw_eval_bounding_boxes(image_path, eval_json_path, eval_output_path, image_name, image_type)
