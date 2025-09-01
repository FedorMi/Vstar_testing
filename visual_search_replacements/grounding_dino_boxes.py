from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
import json
import tqdm
import numpy 
import torch



def save_json(output_path, boxes, phrases, image_shape):
    """Save bounding boxes and corresponding objects to a JSON file."""
    h, w, _ = image_shape  # Get image dimensions
    # Convert boxes to normal image coordinates in [x, y, width, height] format
    boxes_scaled = (boxes * torch.tensor([w, h, w, h])).tolist()
    boxes_xywh = [
        [box[0], box[1], box[2], box[3]]  # Convert [x1, y1, x2, y2] to [x, y, width, height]
        for box in boxes_scaled
    ]
    json_data = {
        "boxes": boxes_xywh,  # Scaled bounding boxes in [x, y, width, height] format
        "missing_objects": phrases
    }
    json_file_path = output_path.replace(".jpg", ".json")
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)
    print(f"JSON file saved as '{json_file_path}'")

def process_image_with_quadrants(image, model, text_prompt, box_threshold, text_threshold):
    """Process an image by dividing it into 4 quadrants, running GroundingDINO, and returning adjusted annotations."""
    _, h, w = image.shape

    # Define quadrants
    quadrants = {
        "top_left": (0, 0, h // 2, w // 2),
        "top_right": (0, w // 2, h // 2, w),
        "bottom_left": (h // 2, 0, h, w // 2),
        "bottom_right": (h // 2, w // 2, h, w)
    }

    all_boxes = []
    all_phrases = []
    all_logits = []

    for quadrant_name, (y1, x1, y2, x2) in quadrants.items():
        # Extract quadrant image
        quadrant_image = image[:, y1:y2, x1:x2]

        #print(quadrant_image.shape)

        # Run prediction on the quadrant
        boxes, logits, phrases = predict(
            model=model,
            image=quadrant_image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        print(boxes)
        # Adjust box coordinates to the full image
        for box in boxes:
            # Adjust x coordinates
            box[0] = (box[0]*(w//2) +x1)/w
            #box[2] = (box[2]*(w//2) +x1)/w
            box[2] = box[2]/2
            # Adjust y coordinates
            box[1] = (box[1]*(h//2) +y1)/h
            #box[3] = (box[3]*(h//2) +y1)/h
            box[3] = box[3]/2
            print(box)
            all_boxes.append(box)

        all_phrases.extend(phrases)
        all_logits.extend(logits)
    return all_boxes, all_phrases, all_logits

def main(with_quadrants=False):

    # Load the GroundingDINO model
    model = load_model(
        "groundingdino/config/GroundingDINO_SwinT_OGC.py", 
        "weights/groundingdino_swint_ogc.pth"
    )

    # Modify the loop for direct_attributes_list
    test_types = ["direct_attributes", "relative_position"]
    for test_type in test_types:
        folder = os.path.join("vbench", test_type)
        attribute_list = list(filter(lambda file: '.json' not in file, os.listdir(folder)))
        for i in attribute_list:
            # Define input image and parameters
            location = os.path.join("vbench", test_type)
            image_path = os.path.join(location, i)
            json_path = os.path.join(location, i.split(".")[0] + ".json")

            #open json file
            with open(json_path, "r") as f:
                data = json.load(f)
            #extract missing objects
            target_objects = data["target_object"]
            #extract target object names
            text_prompt = ""
            for j in range(len(target_objects)-1):
                #append to text prompt
                text_prompt += target_objects[j] + " . "
            text_prompt += target_objects[-1]  # Add the last object without a trailing period
            box_threshold = 0.35
            text_threshold = 0.25

            # Load the image
            image_source, image = load_image(image_path)

            # Run prediction
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            print(boxes.shape)
            print(logits.shape)
            if with_quadrants:
                quadrant_boxes, quadrant_phrases, quadrant_logits = process_image_with_quadrants(
                    image=image,
                    model=model,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                    )
            
            total_boxes = boxes
            total_logits = logits
            if with_quadrants:
                for box in quadrant_boxes:
                    #make box shape instead of torch.Size([4]) be torch.Size([1, 4])
                    total_boxes = torch.cat([total_boxes, box.unsqueeze(0)], dim=0)
                for logit in quadrant_logits:
                    print(logit)
                    total_logits = torch.cat([total_logits, logit.unsqueeze(0)], dim=0)
            total_phrases = []
            total_phrases.extend(phrases)
            if with_quadrants:
                total_phrases.extend(quadrant_phrases)

            # Annotate the image with predictions
            annotated_frame = annotate(
                image_source=image_source,
                boxes=total_boxes,
                logits=total_logits,
                phrases=total_phrases
            )

            # Save the annotated image
            output_path = os.path.join("dino_quadrant_output", test_type, i)
            cv2.imwrite(output_path, annotated_frame)
            print(f"Annotated image saved as '{i}'")

            # Save the JSON file with scaled boxes
            save_json(output_path, total_boxes, total_phrases, image_source.shape)

if __name__ == "__main__":
    main(with_quadrants=True)