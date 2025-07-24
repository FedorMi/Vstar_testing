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

def normal_run_test(how = "full"):
    if how == "full":
        final_results_correct_comparison("general_jsons/eval_results_initial_seal_testing.json")
    if how == "missing":
        final_results_correct_comparison("general_jsons/eval_results_correct_missing_object_labels.json")
    if how == "missing_bbox":
        final_results_correct_comparison("general_jsons/eval_results_correct_bounding_boxes.json")
    if how == "from_paper":
        final_results_correct_comparison("general_jsons/eval_result_from_paper.json")

def missing_objects_comparison(json_file_correct, json_file_new):
    json_correct = open_json_file(json_file_correct)
    json_new = open_json_file(json_file_new)
    overall_count, overall_recall_total, overall_precision_total = 0, 0, 0
    for i in json_new:
        specific_count, specific_recall_total, specific_precision_total = 0, 0, 0
        for idx, new_eval in enumerate(json_new[i]):
            true_positive, false_positive, false_negative = 0, 0, 0
            correct_eval = json_correct[i][idx]
            new_missing = new_eval['missing_objects']
            correct_missing = correct_eval['missing_objects']
            for i in new_missing:
                if i in correct_missing:
                    true_positive += 1
                if i not in correct_missing:
                    false_positive += 1
            for i in correct_missing:
                if i not in new_missing:
                    false_negative += 1
            # compute individual recall and precision
            individual_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            individual_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specific_precision_total += individual_precision
            specific_recall_total += individual_recall
            specific_count += 1
        overall_count += specific_count
        print(f"Missing objects for {i}:")
        print(f"Precision percentage: {specific_precision_total / specific_count * 100 if specific_count > 0 else 0}%")
        print(f"Recall percentage: {specific_recall_total / specific_count * 100 if specific_count > 0 else 0}%")
        overall_precision_total += specific_precision_total
        overall_recall_total += specific_recall_total
    print(f"Missing objects overall:")
    print(f"Precision percentage: {overall_precision_total / overall_count * 100 if overall_count > 0 else 0}%")
    print(f"Recall percentage: {overall_recall_total / overall_count * 100 if overall_count > 0 else 0}%")

def bounding_boxes_comparison(json_file_correct, json_file_new, what_kind='iou', raw_or_recall='raw', threshold=0.5):
    json_correct = open_json_file(json_file_correct)
    json_new = open_json_file(json_file_new)
    overall_count, overall_box_count, overall_value, overall_recall_total, overall_precision_total = 0, 0, 0, 0, 0
    for i in json_new:
        specific_count, specific_box_count, specific_value, specific_recall_total, specific_precision_total = 0, 0, 0, 0, 0
        for idx, new_eval in enumerate(json_new[i]):
            individual_total_value, true_positive, false_positive, false_negative = 0, 0, 0, 0
            correct_eval = json_correct[i][idx]
            new_boxes = new_eval['bounding_boxes']
            correct_boxes = correct_eval['bounding_boxes']
            for i in range(len(new_boxes)):
                new_box = new_boxes[i]
                correct_box = correct_boxes[i]
                judge_value = 0
                if what_kind == 'iou':
                    judge_value = iou(new_box, correct_box)
                else:
                    judge_value = iou(new_box, correct_box)
                
                if judge_value < threshold:
                    false_positive += 1
                else:
                    true_positive += 1
                individual_total_value += judge_value
            specific_box_count += len(new_boxes)
            false_negative += len(correct_boxes) - true_positive
            individual_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            individual_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specific_value += individual_total_value / len(new_boxes) if len(new_boxes) > 0 else 0
            specific_precision_total += individual_precision
            specific_recall_total += individual_recall
            specific_count += 1
        overall_count += specific_count
        overall_box_count += specific_box_count
        overall_value += specific_value
        print(f"Bounding boxes for {i}:")
        if raw_or_recall == 'recall':
            print(f"Precision percentage: {specific_precision_total / specific_count * 100 if specific_count > 0 else 0}%")
            print(f"Recall percentage: {specific_recall_total / specific_count * 100 if specific_count > 0 else 0}%")
        else:
            # if raw_or_recall == 'raw':
            print(f"Average value percentage: {specific_value / specific_box_count if specific_box_count > 0 else 0}%")
        overall_precision_total += specific_precision_total
        overall_recall_total += specific_recall_total
    print(f"Overall bounding boxes:")
    if raw_or_recall == 'recall':
        print(f"Precision percentage: {overall_precision_total / overall_count * 100 if overall_count > 0 else 0}%")
        print(f"Recall percentage: {overall_recall_total / overall_count * 100 if overall_count > 0 else 0}%")
    else:
        # if raw_or_recall == 'raw':
        print(f"Average value percentage: {overall_value / overall_box_count if overall_box_count > 0 else 0}%")

def bounding_boxes_different_missing_comparison(json_file_correct, json_file_new, raw_or_recall='raw', what_kind='iou', threshold=0.5):
    json_correct = open_json_file(json_file_correct)
    json_new = open_json_file(json_file_new)
    overall_count, overall_box_count, overall_value, overall_recall_total, overall_precision_total = 0, 0, 0, 0, 0
    for i in json_new:
        specific_count, specific_box_count, specific_value, specific_recall_total, specific_precision_total = 0, 0, 0, 0, 0
        for idx, new_eval in enumerate(json_new[i]):
            individual_total_value, true_positive, false_positive, false_negative = 0, 0, 0, 0
            correct_eval = json_correct[i][idx]
            new_search_results = new_eval['search_result']
            correct_search_results = correct_eval['search_result']
            new_boxes = [j["bbox"] for j in new_search_results]  # create a list of bounding boxes from search results
            correct_boxes = [j["bbox"] for j in correct_search_results]  # create a list of bounding boxes from search results
            tracking_list = [0] * len(new_boxes)  # list of zeroes for every slot in the new_boxes
            for k in range(len(correct_boxes)):
                correct_box = correct_boxes[k]
                judge_value = -1
                judge_index = -1
                for j in range(len(new_boxes)):
                    new_box = new_boxes[j]
                    temp_judge_value = 0
                    if what_kind == 'iou':
                        temp_judge_value = iou(new_box, correct_box)
                    else:
                        temp_judge_value = iou(new_box, correct_box)
                    if temp_judge_value > judge_value:
                        judge_value = temp_judge_value
                        judge_index = j
                tracking_list[judge_index] = 1

                if judge_value < threshold:
                    false_negative += 1
                else:
                    true_positive += 1
                individual_total_value += judge_value
            empty_count = tracking_list.count(0)
            specific_box_count += len(new_boxes)
            false_positive += empty_count - true_positive
            individual_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            individual_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            specific_value += individual_total_value / len(new_boxes) if len(new_boxes) > 0 else 0
            specific_precision_total += individual_precision
            specific_recall_total += individual_recall
            specific_count += 1
        overall_count += specific_count
        overall_box_count += specific_box_count
        overall_value += specific_value
        print(f"Bounding boxes for {i}:")
        if raw_or_recall == 'recall':
            print(f"Precision percentage: {specific_precision_total / specific_count * 100 if specific_count > 0 else 0}%")
            print(f"Recall percentage: {specific_recall_total / specific_count * 100 if specific_count > 0 else 0}%")
        else:
            # if raw_or_recall == 'raw':
            print(f"Average value percentage: {specific_value / specific_box_count if specific_box_count > 0 else 0}%")
        overall_precision_total += specific_precision_total
        overall_recall_total += specific_recall_total
    print(f"Overall bounding boxes:")
    if raw_or_recall == 'recall':
        print(f"Precision percentage: {overall_precision_total / overall_count * 100 if overall_count > 0 else 0}%")
        print(f"Recall percentage: {overall_recall_total / overall_count * 100 if overall_count > 0 else 0}%")
    else:
        # if raw_or_recall == 'raw':
        print(f"Average value percentage: {overall_value / overall_box_count if overall_box_count > 0 else 0}%")
    
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
        print(f"Correct percentage for {i}: {specific_correct / specific_count * 100 if specific_count > 0 else 0}%")
    print(f"Overall correct percentage: {overall_correct / overall_count * 100 if overall_count > 0 else 0}%")


if __name__ == "__main__":
    test_type = "normal_run_missing"
    if test_type == "normal_run_full":
        normal_run_test("full")
    elif test_type == "normal_run_missing":
        normal_run_test("missing")
    elif test_type == "normal_run_missing_bbox":
        normal_run_test("missing_bbox")
    elif test_type == "from_paper":
        normal_run_test("from_paper")
    elif test_type == "bbox_unequal_missing":
        bounding_boxes_different_missing_comparison(
            "general_jsons/eval_results_correct_bounding_boxes.json",
            "general_jsons/eval_results_correct_bounding_boxes.json",
            raw_or_recall='raw', what_kind='iou', threshold=0.5
        )