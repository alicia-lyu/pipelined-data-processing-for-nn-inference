# TODO: send images one by one to image_client.py, data arrival ~ uniform distribution
import os
import time
import random
import subprocess
import argparse

CLIENT = "image_client.py"
IMAGE_FOLDER = "../../datasets/SceneTrialTrain"

def run_subprocess(image_paths, id):
    subprocess.run(["python", CLIENT] + image_paths)

def run_subprocess(image_paths, id):
    subprocess.run(["python", CLIENT] + image_paths)

def batch_arrival(min_interval, max_interval, batch_size, target):
    image_paths = read_images_from_folder(IMAGE_FOLDER)

    for i in range(0, len(image_paths), batch_size):
    # for i in range(0, batch_size*2, batch_size):
        batch = image_paths[i: i + batch_size]
        target(batch, i / batch_size)
        interval = random.uniform(min_interval, max_interval)
        time.sleep(interval)
    
    return i % batch_size + 1 # number of clients

def read_images_from_folder(root_folder):
    image_paths = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    return image_paths

def get_batch_args():
    parser = argparse.ArgumentParser(description="set pipeline data arrival interval")

    parser.add_argument("--min", type=int, default=1, help="Minimum data arrival interval")
    parser.add_argument("--max", type=int, default=5, help="Maximum data arrival interval")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = get_batch_args()

    batch_arrival(args.min, args.max, args.batch_size, run_subprocess)