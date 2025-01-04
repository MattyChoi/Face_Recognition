import os
import sys

import json
from tqdm import tqdm

def save_img_paths_to_file(data_dir, filename):
    """
    Save a list of image_paths to a text file.

    Args:
        data_dir (str): path to the directory of images to write to the file.
        filename (str): The name of the file to save the strings to.
    """
    if os.path.exists(filename):
        os.remove(filename)
    
    data = {
        "img_paths": {},
    }
    num_imgs = 0
    
    for ids in tqdm(os.listdir(data_dir)):
        data["img_paths"][ids] = []
        for img_path in os.listdir(os.path.join(data_dir, ids)):
            data["img_paths"][ids].append(os.path.join(data_dir, ids, img_path))
            num_imgs += 1
    data["num_imgs"] = num_imgs
            
    # Dump the dictionary into the JSON file
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4) 
        

if __name__ == "__main__":
    data_dir = "/Users/matthewchoi/Projects/FaceDetection/FaceRecognitionModel/data/face_imgs"

    # Output file name
    output_file = "/Users/matthewchoi/Projects/FaceDetection/FaceRecognitionModel/data/img_paths.json"

    # Save strings to the file
    save_img_paths_to_file(data_dir, output_file)
