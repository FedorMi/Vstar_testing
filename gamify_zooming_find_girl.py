import json
import cv2
import torch
import os
import av
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as TF

from ollama import chat, ResponseError, pull
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from typing import Type, Tuple, List
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

### 1 Image as History
factor = 0.05

def process_image(rect_image_path, temp_image_path, question_text):
    # Process image with the specified model
    try:
        response = chat(
            model='minicpm-v',
            messages=[
                {'role': 'user', 'content': question_text, 'images':[rect_image_path, temp_image_path]}
#                {'role': 'user', 'content': question_text, 'images':[temp_image_path]}
            ]
        )
    except ResponseError as e:
        if e.status_code == 404:
            print('Model not found, pulling the model...')
            pull('minicpm-v')
            response = chat(
                model='minicpm-v',
                messages=[
                    {'role': 'user', 'content': question_text}
                ],
                images=[rect_image_path, temp_image_path]
            )
        else:
            raise e
    result = response['message']['content']
    return result

def explore_image(image, action, x1, y1, x2, y2, ori_image, rectangle_image_path, temp_image_path, ox1, oy1, ox2, oy2 ):
    oheight, owidth = ori_image.shape[:2]
    
    
    #height, width = image.shape[:2]
    crop_width = x2 - x1
    crop_height = y2 - y1

    if action == "zoom in":
        x1 += int(crop_width * factor)
        y1 += int(crop_height * factor)
        x2 -= int(crop_width * factor)
        y2 -= int(crop_height * factor)
    elif action == "zoom out":
        x1 = max(0, x1 - int(crop_width * factor))
        y1 = max(0, y1 - int(crop_height * factor))
        x2 = min(ox2, x2 + int(crop_width * factor))
        y2 = min(oy2, y2 + int(crop_height * factor))
    elif action == "left":
        x1 = max(0, x1 - int(crop_width * factor))
        x2 = x2 - int(crop_width * factor)
    elif action == "right":
        x1 = x1 + int(crop_width * factor)
        x2 = min(ox2, x2 + int(crop_width * factor))
    elif action == "up":
        y1 = max(0, y1 - int(crop_height * factor))
        y2 = y2 - int(crop_height * factor)
    elif action == "down":
        y1 = y1 + int(crop_height * factor)
        y2 = min(oy2, y2 + int(crop_height * factor))

    # Draw a red rectangle on the original image to show the current view
    original_image = ori_image.copy()
    original_image2 = ori_image.copy()
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 0, 255), 8)
    cv2.imwrite(rectangle_image_path, original_image)
    cv2.imwrite(temp_image_path, original_image2[y1:y2, x1:x2])
    return image[y1:y2, x1:x2], x1, y1, x2, y2

def possible_actions(x1, y1, x2, y2, ox1, oy1, ox2, oy2):
    actions = []
    crop_width = x2 - x1
    crop_height = y2 - y1

    if x1 - int(crop_width * factor) >= ox1 and x2 + int(crop_width * factor) <= ox2 and y1 - int(crop_height * factor) >= oy1 and y2 + int(crop_height * factor) <= oy2:
        actions.append("zoom out")
    if x1 - int(crop_width * factor) >= ox1:
        actions.append("left")
    if x2 + int(crop_width * factor) <= ox2:
        actions.append("right")
    if y1 - int(crop_height * factor) >= oy1:
        actions.append("up")
    if y2 + int(crop_height * factor) <= oy2:
        actions.append("down")

    return actions

def create_instruction(question_text, actions, x):
    first = (
                "You are a Vision-Language assistant.\n"
#                "You are shown the image : <image_placeholder>\n"
                 "You are shown the first image : <image_placeholder> and the second image : <image_placeholder>.\n"
                "Your goal is to answer the following question about the image:\n"
                "Your goal is to follow the following instruction.\n"
                f"{question_text}\n\n"
                "You see the entire image at normal resolution, but you can request a high-resolution crop of specific areas.\n\n" "Use #MOVE to request a new crop:\n"
            )
    left = (
        "If you want the frame to move to the left, respond with:\n"
        "#MOVE: direction=left\n")
    right = (
        "If you want the frame to move to the right, respond with:\n"
        "#MOVE: direction=right\n")
    up = (
        "If you want the frame to move up, respond with:\n"
        "#MOVE: direction=up\n")
    down = (
        "If you want the frame to move down, respond with:\n"
        "#MOVE: direction=down\n")
    zoom_in = (
        "If you want to zoom in, respond with:\n"
        "#MOVE: direction=zoom in\n")
    zoom_out = (
        "If you want to zoom out, respond with:\n"
        "#MOVE: direction=zoom out\n")
    move_instruction = (
                "\nYou can only chose one action at a time.\n"
#                "The image is the current view of the image.\n\n"
                "The first image is the full image.\n"
                "The second image is your current view after the previous moves.\n"
                "The location of the current zoom is marked on the first image with rectangle.\n\n"
                
            )
    answer_instruction = (
                "Only when you are sure of your answer respond to the question.\n"
                "If there is any doubt make a move to explore the image.\n"
                "When you are sure of your answer, respond with: \n"
                "#ANSWER: <your answer>\n\n"
            )
    move = ("Do NOT reveal internal chain-of-thought. Only output a #MOVE line.")
    move_answer = ("Do NOT reveal internal chain-of-thought. Only output a #MOVE or #ANSWER line.")
    answer = ("Do NOT reveal internal chain-of-thought. Only output a #ANSWER line.")
    
    system_prompt = first
    if "up" in actions:
        system_prompt += up
    if "down" in actions:
        system_prompt += down
    if "left" in actions:
        system_prompt += left
    if "right" in actions:
        system_prompt += right
    system_prompt += zoom_in
    if "zoom out" in actions:
        system_prompt += zoom_out
    system_prompt += move_instruction
    if x > 200:
        system_prompt += answer_instruction + move_answer
    else:
        system_prompt += move
    return system_prompt

def interactive_process_image(question, question_text, image_name):
    #question_text = "Focus the frame on the little girl in the image."
    question_text = "Focus on the left frame edge of the image."
    folder = "left_minicpmv"
    idxt = 0
    image_jpgless = image_name.split(".")[0]
    image_path = os.path.join(question['category'], image_name)
    patht = os.path.join(folder, question['category'],image_jpgless)
    temp_image_path = os.path.join(patht, str(idxt)+"t_"+image_name)
    rectangle_image_path = os.path.join(patht, str(idxt)+"r_"+image_name)
    if not os.path.exists(patht):
        Path(patht).mkdir(parents=True, exist_ok=True)
    ori_image = cv2.imread(image_path)
    ox1, oy1, ox2, oy2 = 0, 0, ori_image.shape[1], ori_image.shape[0]
    x1, y1, x2, y2 = ox1, oy1, ox2, oy2
    image = ori_image.copy()
    cv2.imwrite(temp_image_path, image)
    rectangle_image = ori_image.copy()
    cv2.rectangle(rectangle_image, (x1, y1), (x2, y2), (0, 0, 255), 8)
    cv2.imwrite(rectangle_image_path, rectangle_image)
    print("Images written")
    done = None
    x=0
    possible_action = []
    while True:
        system_prompt = create_instruction(question_text, possible_action, x)
        result = process_image(rectangle_image_path, temp_image_path, system_prompt)
        print(f"Model response: {result}")
        action = "temprorary"
        if "#ANSWER:"  in result:
            done = result.replace("#ANSWER:", " ").strip().split(" ")[0].strip()
            break
        elif "#MOVE: direction=" in result:
            action = result.split("=")[1].strip()
            idxt += 1
        if action == "temprorary":
            print("we have an issue")
        temp_image_path = os.path.join(patht, str(idxt)+ "t_"+image_name)
        rectangle_image_path = os.path.join(patht, str(idxt)+"r_"+image_name)
        image, x1, y1, x2, y2 = explore_image(image, action, x1, y1, x2, y2, ori_image, rectangle_image_path, temp_image_path, ox1, oy1, ox2, oy2)
        x+=1
        possible_action = possible_actions(x1, y1, x2, y2, ox1, oy1, ox2, oy2)
    
    print(f"Final answer: {done}")
    return done

def compare_results(label, result):
    # Compare the result with the solution label
    return result.strip().upper() == label.strip().upper()

def main():
    questions_file = 'test_questions.jsonl'
    
    direct_attributes_correct = 0
    direct_attributes_total = 0
    relative_position_correct = 0
    relative_position_total = 0
    
    with open(questions_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    for question in tqdm(questions, desc="Processing questions"):
        image_name = question['image'].split('/')[-1]
        if image_name != 'sa_17.jpg':
            continue
        impath = os.path.join(question['category'], image_name)
        if os.path.exists(impath):
            question_text = question['text']
            #path = os.path.join("vbench", "direct_attributes")
            result = interactive_process_image(question, question_text, image_name)
            print("correct answer", question['label'])
            is_correct = compare_results(question['label'], result.replace("(", " ").replace(")", " ").strip())
            if question['category'] == 'direct_attributes':
                direct_attributes_total += 1
                if is_correct:
                    direct_attributes_correct += 1
            if question['category'] == 'relative_position':
                relative_position_total += 1
                if is_correct:
                    relative_position_correct += 1
            print(f'Image: {image_name}, Correct: {is_correct}')
        
    overall_correct = direct_attributes_correct + relative_position_correct
    overall_total = direct_attributes_total + relative_position_total
    
    print(f'Direct Attributes: {direct_attributes_correct}/{direct_attributes_total} ({direct_attributes_correct/direct_attributes_total*100:.2f}%)')
    print(f'Relative Position: {relative_position_correct}/{relative_position_total} ({relative_position_correct/relative_position_total*100:.2f}%)')
    print(f'Overall: {overall_correct}/{overall_total} ({overall_correct/overall_total*100:.2f}%)')

if __name__ == "__main__":
    main()


### Video as History


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def video_llava():
    # Load the model in half-precision
    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf", torch_dtype=torch.float16, device_map="auto")
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

    # Load the video as an np.arrau, sampling uniformly 8 frames
    #video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
    video_path = "sample_video.mp4"
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    # For better results, we recommend to prompt the model in the following format
    prompt = "USER: <video>\nWhy is this funny? ASSISTANT:"
    inputs = processor(text=prompt, videos=video, return_tensors="pt")

    out = model.generate(**inputs, max_new_tokens=60)
    processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

