import json
import os

def iou(bbox1, bbox2):
	x1 = max(bbox1[0], bbox2[0])
	y1 = max(bbox1[1], bbox2[1])
	x2 = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
	y2 = min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
	inter_area = max(0, x2 - x1) * max(0, y2 - y1)
	return inter_area/(bbox1[2]*bbox1[3]+bbox2[2]*bbox2[3]-inter_area)

def open_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def normal_run_test():
    pass
def normal_run_missing_test():
    pass
def normal_run_missing_bbox_test():
    pass

def missing_objects_comparison(json_file_correct, json_file_new):
    json_correct = open_json_file(json_file_correct)
    json_new = open_json_file(json_file_new)
def bounding_boxes_comparison(json_file_correct, json_file_new):
    json_correct = open_json_file(json_file_correct)
    json_new = open_json_file(json_file_new)
def bounding_boxes_different_missing_comparison(json_file_correct, json_file_new):
    json_correct = open_json_file(json_file_correct)
    json_new = open_json_file(json_file_new)
def final_results_correct_comparison(json_file_new):
    json_new = open_json_file(json_file_new)
    overall_count = 0
    overall_correct = 0
    for i in json_new:
        specific_count = 0
        specific_correct = 0
        for j in json_new[i]:
            if j['correct'] == 1:
                specific_correct += 1
                overall_correct += 1
            specific_count += 1
            overall_count += specific_count
        print(f"Correct for {i}: {specific_correct}")
    print(f"Overall correct: {overall_correct}")


if __name__ == "__main__":
    test_type = "normal_full_run"
    if test_type == "normal_run_full":
        normal_run_test()
    elif test_type == "normal_run_missing":
        normal_run_missing_test()
    elif test_type == "normal_run_missing_bbox":
        normal_run_missing_bbox_test()