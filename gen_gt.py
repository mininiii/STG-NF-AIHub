
import json
import numpy as np
import os

import json
import numpy as np

def process_json_file(directory_path, json_file):
    file_name = json_file.replace("_alphapose_tracked_person", '')
    json_file_path = os.path.join(directory_path, file_name)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    target = json_file.split('_')[1]
    # block_detail이 target("A20", ...)인 것만 찾기
    frames_count = int(data['file'][0]['videos']['block_information'][-1]['end_frame_index']) + 1

    ground_truth = np.zeros(frames_count)

    for block in data['file'][0]['videos']['block_information']:
        if block['block_detail'] == target:
            start_frame = int(block['start_frame_index'])
            end_frame = int(block['end_frame_index'])
            ground_truth[start_frame:end_frame + 1] = 1

    # 텍스트 파일로 저장
    output_file_path = os.path.join('/home/myyang/projects/STG-NF/data/AIHub/gt/', file_name[:-5] + '.txt')
    np.save(output_file_path, ground_truth)
    print(f"Processed: {json_file_path}")

def find_files(type):
# 디렉토리 내의 모든 .json 파일 찾기
    directory_path = f'/home/myyang/projects/STG-NF/data/AIHub/pose/{type}'
    json_files = [f for f in os.listdir(directory_path) if f.endswith('alphapose_tracked_person.json')]
    filepath = '/home/myyang/projects/dataset/AIHub/zipfiles/training/label/invation/look_inside' if type == 'train' \
        else '/home/myyang/projects/dataset/AIHub/zipfiles/validation/unzip/look_inside/label'
    for json_file in json_files:
        process_json_file(filepath, json_file)

find_files('train')
find_files('test')
