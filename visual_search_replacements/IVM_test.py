import os
import json
from ollama import chat, ResponseError, pull
from PIL import Image
from tqdm import tqdm
from IVM_test_boxes import main_normal

def process_image(image_path, question_text):
    # Process image with the specified model
    try:
        response = chat(
            model='llava:34b',
            messages=[
                {'role': 'user', 'content': question_text, 'images':[image_path]}
            ]
        )
    except ResponseError as e:
        if e.status_code == 404:
            print('Model not found, pulling the model...')
            pull('llava:34b')
            response = chat(
                model='llava:34b',
                messages=[
                    {'role': 'user', 'content': question_text}
                ],
                images=[image_path]
            )
        else:
            raise e
    result = response['message']['content']
    return result

def compare_results(label, result):
    # Compare the result with the solution label
    return result.strip().upper() == label.strip().upper()

def main():
    main_normal()
    questions_file = 'test_questions.jsonl'
    outputter_folder = 'ivm_image_results'
    
    direct_attributes_correct = 0
    direct_attributes_total = 0
    relative_position_correct = 0
    relative_position_total = 0
    
    with open(questions_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    
    for question in tqdm(questions, desc="Processing questions"):
        image_name = question['image']
        image_path = os.path.join(outputter_folder, image_name)
        if os.path.exists(image_path):
            result = process_image(image_path, question['text'])
            is_correct = compare_results(question['label'], result)
            if question['category'] == 'direct_attributes':
                direct_attributes_total += 1
                if is_correct:
                    direct_attributes_correct += 1
            elif question['category'] == 'relative_position':
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
