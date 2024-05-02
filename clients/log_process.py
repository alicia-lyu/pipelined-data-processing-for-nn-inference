import os
import re

def read_files_in_directory(directory:str,system_type:str):
    files = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and subdir.startswith(system_type):
            for filename in sorted(os.listdir(subdir_path)):
                file_path = os.path.join(subdir_path, filename)
                if os.path.isfile(file_path):
                    files.append(file_path)
    return files

def extract_info_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        process_info = {}
        process_info['file_path'] = file_path
        for line in lines:
            if line.strip():
                if ' process created' in line:
                    process_info['process_created'] = float(line.split()[0])
                elif ' preprocessing started' in line:
                    process_info['preprocessing_started'] = float(line.split()[0])
                elif ' preprocessing ended' in line:
                    process_info['preprocessing_ended'] = float(line.split()[0])
                elif ' detection inference started' in line:
                    process_info['detection_inference_started'] = float(line.split()[0])
                elif ' detection inference ended' in line:
                    process_info['detection_inference_ended'] = float(line.split()[0])
                elif ' cropping started' in line:
                    process_info['cropping_started'] = float(line.split()[0])
                elif ' cropping ended' in line:
                    process_info['cropping_ended'] = float(line.split()[0])
                elif ' recognition inference started' in line:
                    process_info['recognition_inference_started'] = float(line.split()[0])
                elif ' recognition inference ended' in line:
                    process_info['recognition_inference_ended'] = float(line.split()[0])
                elif ' postprrocessing started' in line:
                    process_info['postprocessing_started'] = float(line.split()[0])
                elif ' postprrocessing ended' in line:
                    process_info['postprocessing_ended'] = float(line.split()[0])
                elif 'process length' in line:
                    process_info['process_length'] = float(line.split()[0])
                elif 'preprocessing length' in line:
                    process_info['preprocessing_length'] = float(line.split()[0])
                elif 'detection inference length' in line:
                    process_info['detection_inference_length'] = float(line.split()[0])
                elif 'cropping length' in line:
                    process_info['cropping_length'] = float(line.split()[0])
                elif 'recognition inference length' in line:
                    process_info['recognition_inference_length'] = float(line.split()[0])
                elif 'postprrocessing length' in line:
                    process_info['postprocessing_length'] = float(line.split()[0])
                elif 'waiting for preprocessing time' in line:
                    process_info['waiting_for_preprocessing_time'] = float(line.split()[0])
                elif 'waiting for cropping time' in line:
                    process_info['waiting_for_cropping_time'] = float(line.split()[0])
                elif 'waiting for postprrocessing time' in line:
                    process_info['waiting_for_postprocessing_time'] = float(line.split()[0])
    return process_info

def read_and_extract_info(directory:str,system_type:str):
    files = read_files_in_directory(directory,system_type)
    process_infos = []
    for file in files:
        process_info = extract_info_from_file(file)
        process_infos.append(process_info)
    return process_infos

if __name__ == "__main__":
    directory = '../log_image/'
    naive_process_infos = read_and_extract_info(directory,"naive")
    print("naive sequential all process time:",naive_process_infos[-1]['postprocessing_ended']-naive_process_infos[0]['process_created'])

    non_coordinate_process_infos = read_and_extract_info(directory,"non-coordinate")
    print("non-coordinate batch all process time:",non_coordinate_process_infos[-1]['postprocessing_ended']-non_coordinate_process_infos[0]['process_created'])

    pipeline_process_infos = read_and_extract_info(directory,"pipeline")
    print("pipeline all process time:",pipeline_process_infos[-1]['postprocessing_ended']-pipeline_process_infos[0]['process_created'])