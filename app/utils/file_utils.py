import os

def create_directories(data_dir, images_dir):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
