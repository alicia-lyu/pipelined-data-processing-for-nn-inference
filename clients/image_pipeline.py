# TODO: send images one by one to image_client.py, data arrival ~ uniform distribution
import os
import time
import random
import subprocess
import argparse

def send_images(folder_path, client_script,min_interval,max_interval,batch_size):
    image_paths = read_images_from_folder(folder_path)

    for i in range(0, len(image_paths), batch_size):
    # for i in range(0, batch_size*2, batch_size):
        batch = image_paths[i:i + batch_size]
        batch_argument = " ".join(batch)
        subprocess.run(["python", client_script] + batch)

        interval = random.uniform(min_interval, max_interval)
        time.sleep(interval)

def read_images_from_folder(root_folder):
    image_paths = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".jpg"):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    return image_paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="set pipeline data arrival interval")

    parser.add_argument("--min", type=int, default=1, help="Minimum data arrival interval")
    parser.add_argument("--max", type=int, default=5, help="Maximum data arrival interval")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    args = parser.parse_args()

    folder_path = "../../datasets/SceneTrialTrain"
    client_script = "image_client.py"

    send_images(folder_path, client_script,args.min,args.max,args.batch_size)