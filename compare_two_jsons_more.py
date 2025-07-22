import json
from tqdm import tqdm

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def iou(bbox1, bbox2):
	x1 = max(bbox1[0], bbox2[0])
	y1 = max(bbox1[1], bbox2[1])
	x2 = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
	y2 = min(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])
	inter_area = max(0, x2 - x1) * max(0, y2 - y1)
	return inter_area/(bbox1[2]*bbox1[3]+bbox2[2]*bbox2[3]-inter_area)


def compare_json_file_to_origin_json(file1):
    data1 = load_json(file1)
    
    
    acc_list_direct = []
    location_direct_attributes = "vbench\\direct_attributes\\"
    for _, element1 in tqdm(enumerate(data1["direct_attributes"]), total=len(data1["direct_attributes"])):
        name_file = element1["image"].replace(".jpg", "")
        gt_bbox = load_json(location_direct_attributes+name_file+".json")["bbox"][0]
        if len(element1["search_result"]) == 0:
            acc_list_direct.append(0.0)
            continue
        search_bbox = element1["search_result"][0]["bbox"]
        iou_i = iou(search_bbox, gt_bbox)
        det_acc = 1.0 if iou_i > 0.5 else 0.0
        acc_list_direct.append(det_acc)
    

    acc_list_relative = []
    location_relative_position = "vbench\\relative_position\\"
    for _, element1 in tqdm(enumerate(data1["relative_position"]), total=len(data1["relative_position"])):
        name_file = element1["image"].replace(".jpg", "")
        relative_file = load_json(location_relative_position+name_file+".json")
        if len(element1["search_result"]) == 0:
            acc_list_relative.append(0.0)
            continue
        for d in range(len(relative_file["bbox"])):
            for j in element1["search_result"]:
                if j["name"] == relative_file["target_object"][d]:
                    search_bbox = j["bbox"]
                    gt_bbox = relative_file["bbox"][d]
                    iou_i = iou(search_bbox, gt_bbox)
                    det_acc = 1.0 if iou_i > 0.5 else 0.0
                    acc_list_relative.append(det_acc)
                    break
                else:
                    acc_list_relative.append(0.0)
                    continue 
    return acc_list_direct, acc_list_relative

def compare_json_to_origin(file1):
    data1 = load_json(file1)
    
    
    acc_list_direct = []
    unmatched_direct_files = []
    location_direct_attributes = "vbench\\direct_attributes\\"
    for _, element1 in tqdm(enumerate(data1["direct_attributes"]), total=len(data1["direct_attributes"])):
        name_file = element1["image"].replace(".jpg", "")
        gt_bbox = load_json(location_direct_attributes+name_file+".json")["bbox"][0]
        if len(element1["search_result"]) == 0:
            acc_list_direct.append(0.0)
            unmatched_direct_files.append(name_file)
            continue
        search_bbox = element1["search_result"][0]["bbox"]
        iou_i = iou(search_bbox, gt_bbox)
        det_acc = 1.0 if iou_i > 0.5 else 0.0
        acc_list_direct.append(det_acc)
        if det_acc == 0.0:
            unmatched_direct_files.append(name_file)
    

    acc_list_relative = []
    unmatched_relative_files = []
    location_relative_position = "vbench\\relative_position\\"
    for _, element1 in tqdm(enumerate(data1["relative_position"]), total=len(data1["relative_position"])):
        name_file = element1["image"].replace(".jpg", "")
        relative_file = load_json(location_relative_position+name_file+".json")
        if len(element1["search_result"]) == 0:
            acc_list_relative.append(0.0)
            unmatched_relative_files.append(name_file)
            continue
        matched = False
        for d in range(len(relative_file["bbox"])):
            for j in element1["search_result"]:
                if j["name"] == relative_file["target_object"][d]:
                    search_bbox = j["bbox"]
                    gt_bbox = relative_file["bbox"][d]
                    iou_i = iou(search_bbox, gt_bbox)
                    det_acc = 1.0 if iou_i > 0.5 else 0.0
                    acc_list_relative.append(det_acc)
                    if det_acc == 0.0:
                        unmatched_relative_files.append(name_file)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            acc_list_relative.append(0.0)
            unmatched_relative_files.append(name_file)
    
    accumul_dir = sum(acc_list_direct) / len(acc_list_direct)
    accumul_relat = sum(acc_list_relative) / len(acc_list_relative)
    
    return accumul_dir, accumul_relat, unmatched_direct_files, unmatched_relative_files

def compare_json_correct(file1):
    data1 = load_json(file1)
    
    incorrect_direct_files = []
    for element1 in data1["direct_attributes"]:
        if element1["correct"] == 0:
            incorrect_direct_files.append(element1["image"].replace(".jpg", ""))
    
    incorrect_relative_files = []
    for element1 in data1["relative_position"]:
        if element1["correct"] == 0:
            incorrect_relative_files.append(element1["image"].replace(".jpg", ""))
    
    return incorrect_direct_files, incorrect_relative_files

if __name__ == "__main__":
    file = "eval_result_2.json"
    accumul_dir, accumul_relat, unmatched_direct_files, unmatched_relative_files = compare_json_to_origin(file)
    print("Direct Attributes Accuracy:", accumul_dir)
    print("Relative Position Accuracy:", accumul_relat)
    print("Unmatched Direct Files:", unmatched_direct_files)
    print("Unmatched Relative Files:", unmatched_relative_files)
    
    incorrect_direct_files, incorrect_relative_files = compare_json_correct(file)
    print("Incorrect Direct Files:", incorrect_direct_files)
    print("Incorrect Relative Files:", incorrect_relative_files)

    if False:
        file = "eval_result_2.json"
        acc_list_direct, acc_list_relative = compare_json_file_to_origin_json(file)
        accumul_dir = 0.0
        for i in acc_list_direct:
            accumul_dir += i
        accumul_dir /= len(acc_list_direct)
        accumul_relat = 0.0
        for i in acc_list_relative:
            accumul_relat += i
        accumul_relat /= len(acc_list_relative)
        print(accumul_dir, accumul_relat)
    #print(json.dumps(differences, indent=4))
